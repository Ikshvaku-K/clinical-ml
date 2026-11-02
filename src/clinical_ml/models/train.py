"""M1 baselines for 30-day readmission on the Diabetes-130 OMOP cohort.

Protocol:
* Split by patient: 60% train / 20% calibration / 20% test. No patient appears in two splits.
* Hyperparameters chosen by 5-fold GroupKFold (grouped by patient) on train only.
* The calibration split is reserved for M2 (calibration, conformal, threshold); its
  predictions are stored but no metrics on it are reported here.
* The test split is scored once, after model selection, with patient-level bootstrap CIs.

The GBM is scikit-learn's HistGradientBoostingClassifier: LightGBM/XGBoost wheels need a
Homebrew OpenMP runtime on macOS, which would break the project's isolated environment.
"""

import itertools
import json
import os
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from clinical_ml import config
from clinical_ml.cohort.features import CATEGORICAL
from clinical_ml.eval.metrics import bootstrap
from clinical_ml.omop import schema

ID_COLS = ["visit_occurrence_id", "person_id", "readmit_30d", "split"]
SPLIT_FRACTIONS = {"train": 0.6, "calibration": 0.2, "test": 0.2}

TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]

LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0)]
# A first grid of (0.05, 0.1) x (15, 31) x (150, 300) selected its smallest corner, so the
# grid was extended toward simpler models until the optimum sat inside it.
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": iters}
    for lr, leaves, iters in itertools.product((0.02, 0.05, 0.1), (7, 15, 31), (100, 200, 400))
]


def patient_split(person_ids: pd.Series, seed: int = config.SEED) -> pd.Series:
    persons = np.sort(person_ids.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(persons)
    n = len(persons)
    cut1 = int(round(SPLIT_FRACTIONS["train"] * n))
    cut2 = cut1 + int(round(SPLIT_FRACTIONS["calibration"] * n))
    assign = pd.Series("test", index=persons)
    assign.iloc[:cut1] = "train"
    assign.iloc[cut1:cut2] = "calibration"
    return person_ids.map(assign)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in ID_COLS if c in df]).copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    drug_cols = [c for c in X.columns if c.startswith("drug_")]
    for c in CATEGORICAL + drug_cols:
        X[c] = X[c].astype("category")
    return X


def make_lr(X: pd.DataFrame, C: float) -> Pipeline:
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    counts = [c for c in X.columns if c.startswith(("prior_", "n_")) or c == "length_of_stay"]
    rest = [c for c in X.columns if c not in cats and c not in counts]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cats),
        ("cnt", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), counts),
        ("num", StandardScaler(), rest),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=3000))])


def make_hgb(_X: pd.DataFrame, **params) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", early_stopping=False, min_samples_leaf=50,
        random_state=config.SEED, **params)


def grouped_cv(make, grid, X, y, groups) -> tuple[dict, list]:
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    results = []
    for params in grid:
        aucs = []
        for tr, va in folds:
            m = make(X, **params).fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        results.append({"params": params, "cv_auroc_mean": float(np.mean(aucs)),
                        "cv_auroc_sd": float(np.std(aucs))})
    best = max(results, key=lambda r: r["cv_auroc_mean"])
    return best, results


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def run() -> dict:
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}")
    with schema.connect() as con:
        df = con.execute(
            f"SELECT * FROM {config.RESULTS}.d130_features ORDER BY visit_occurrence_id").df()
        df["split"] = patient_split(df["person_id"])
        con.register("_s", df[["visit_occurrence_id", "person_id", "split"]])
        con.execute(f"CREATE OR REPLACE TABLE {config.RESULTS}.d130_split AS SELECT * FROM _s")
        con.unregister("_s")

    assert df.groupby("person_id")["split"].nunique().max() == 1, "patient in two splits"
    X, y = feature_frame(df), df["readmit_30d"].astype(int).to_numpy()
    parts = {s: (df["split"] == s).to_numpy() for s in SPLIT_FRACTIONS}
    tr, groups_tr = parts["train"], df.loc[parts["train"], "person_id"].to_numpy()
    manifest = json.loads((config.RAW / "manifest.json").read_text())

    exp = "readmission-d130"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(exp)

    summary = {"split_sizes": {s: {"encounters": int(m.sum()),
                                   "patients": int(df.loc[m, "person_id"].nunique()),
                                   "readmit_rate": float(y[m].mean())}
                               for s, m in parts.items()},
               "n_features": X.shape[1], "models": {}}
    preds = {}
    for name, make, grid in (("logistic_regression", make_lr, LR_GRID),
                             ("hist_gradient_boosting", make_hgb, HGB_GRID)):
        with mlflow.start_run(run_name=f"m1-{name}") as run:
            t0 = time.time()
            best, cv = grouped_cv(make, grid, X[tr], y[tr], groups_tr)
            model = make(X, **best["params"]).fit(X[tr], y[tr])
            p_all = model.predict_proba(X)[:, 1]
            preds[name] = p_all
            mlflow.log_params({"model": name, "selection": "GroupKFold(5) AUROC on train",
                               **best["params"]})
            mlflow.log_metrics({"cv_auroc_mean": best["cv_auroc_mean"],
                                "cv_auroc_sd": best["cv_auroc_sd"],
                                "fit_seconds": time.time() - t0})
            mlflow.set_tags({"git_commit": _git_commit(), "milestone": "M1",
                             "d130_zip_sha256": manifest["diabetes130"]["zip_sha256"]})
            mlflow.log_dict({"cv_grid": cv}, "cv_grid.json")
            mlflow.log_dict({c: str(t) for c, t in X.dtypes.items()}, "features.json")
            # skops audits pickled types; these are the ones sklearn's own HGB/LR objects
            # use, trusted because this process just created the model.
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)
            summary["models"][name] = {"run_id": run.info.run_id, "best": best}

    te = parts["test"]
    test = bootstrap(y[te], {k: v[te] for k, v in preds.items()},
                     df.loc[te, "person_id"].to_numpy(), n_boot=2000, seed=config.SEED)
    summary["test"] = test
    for name, info in summary["models"].items():
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics({f"test_{k}": test[name][k]["value"]
                                for k in ("auroc", "auprc", "brier", "log_loss")})

    with schema.connect() as con:
        out = df[["visit_occurrence_id", "person_id", "split", "readmit_30d"]].copy()
        for k, v in preds.items():
            out[f"p_{k}"] = v
        con.register("_p", out)
        con.execute(f"CREATE OR REPLACE TABLE {config.RESULTS}.d130_predictions AS "
                    "SELECT * FROM _p")
        con.unregister("_p")
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "metrics_m1.json").write_text(json.dumps(summary, indent=2))
    return summary
