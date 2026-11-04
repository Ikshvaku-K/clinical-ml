"""Benchmark: logistic regression vs HistGradientBoosting vs XGBoost, in ONE run on ONE
machine, so the paired bootstrap differences are valid.

XGBoost's macOS wheel needs Homebrew's libomp, which the isolated local environment
deliberately avoids, so this runs on Linux: GitHub Actions / the Docker image, or the
GPU node with `--device cuda` (install with `uv sync --extra benchmark`).

Fairness of the comparison:
* the persisted patient-level split (results.d130_split) is reused, and re-derived from
  the seed as a check;
* identical OMOP features (models.train.feature_frame);
* identical selection: grouped 5-fold CV AUROC on train only, grids of similar size;
* identical evaluation: test scored once, paired patient-level bootstrap (2,000).
This is another read of the test split; the README says so.
"""

import itertools
import json
import os
import platform
import subprocess
import time

import mlflow

from clinical_ml import config
from clinical_ml.eval.metrics import bootstrap
from clinical_ml.models import train
from clinical_ml.omop import schema

XGB_GRID = [
    {"learning_rate": lr, "max_depth": depth, "n_estimators": n}
    for lr, depth, n in itertools.product((0.02, 0.05, 0.1), (3, 4, 6), (100, 200, 400))
]


def xgboost_available() -> tuple[bool, str]:
    try:
        import xgboost  # noqa: F401
    except Exception as e:  # ImportError, or XGBoostError when libomp is missing
        return False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else e}"
    return True, ""


def make_xgb(_X, device: str = "cpu", **params):
    import xgboost

    return xgboost.XGBClassifier(
        tree_method="hist", enable_categorical=True, device=device, n_jobs=-1,
        random_state=config.SEED, eval_metric="logloss", min_child_weight=5, **params)


def _environment(device: str) -> dict:
    import sklearn
    import xgboost

    env = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "device": device,
           "xgboost_version": xgboost.__version__, "sklearn_version": sklearn.__version__}
    if device.startswith("cuda"):
        try:
            env["gpu"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            env["gpu"] = "unknown"
    return env


def _xgb_device_used(model) -> str:
    """The device XGBoost actually trained on, read back from the fitted booster's config."""
    cfg = json.loads(model.get_booster().save_config())
    return cfg["learner"]["generic_param"]["device"]


def run(device: str | None = None) -> dict:
    ok, why = xgboost_available()
    if not ok:
        raise SystemExit(f"XGBoost unavailable on this machine ({why}). Run the benchmark on "
                         "Linux (CI/Docker) or the GPU node; see SCOPE.md, M2.")
    device = device or os.environ.get("XGB_DEVICE", "cpu")
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}")
    with schema.connect(read_only=True) as con:
        df = con.execute(f"""
            SELECT f.*, s.split FROM {config.RESULTS}.d130_features f
            JOIN {config.RESULTS}.d130_split s USING (visit_occurrence_id)
            ORDER BY visit_occurrence_id""").df()
    assert (train.patient_split(df["person_id"]) == df["split"]).all(), "split drifted"

    X, y = train.feature_frame(df), df["readmit_30d"].astype(int).to_numpy()
    tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    groups_tr = df.loc[tr, "person_id"].to_numpy()

    candidates = (
        ("logistic_regression", train.make_lr, train.LR_GRID),
        ("hist_gradient_boosting", train.make_hgb, train.HGB_GRID),
        ("xgboost", lambda X_, **p: make_xgb(X_, device=device, **p), XGB_GRID),
    )
    mlflow.set_experiment("readmission-d130")
    env = _environment(device)
    out = {"environment": env, "models": {}}
    preds = {}
    for name, make, grid in candidates:
        with mlflow.start_run(run_name=f"benchmark-{name}") as run:
            t0 = time.time()
            best, cv = train.grouped_cv(make, grid, X[tr], y[tr], groups_tr)
            model = make(X, **best["params"]).fit(X[tr], y[tr])
            preds[name] = model.predict_proba(X[te])[:, 1]
            seconds = time.time() - t0
            mlflow.set_tags({"milestone": "M2-benchmark", **env})
            mlflow.log_params({"model": name, **best["params"]})
            mlflow.log_metrics({"cv_auroc_mean": best["cv_auroc_mean"],
                                "fit_and_select_seconds": seconds})
            mlflow.log_dict({"cv_grid": cv}, "cv_grid.json")
            out["models"][name] = {"run_id": run.info.run_id, "best": best,
                                   "fit_and_select_seconds": seconds, "cv_grid": cv}
            if name == "xgboost":
                out["models"][name]["device_used"] = _xgb_device_used(model)

    test = bootstrap(y[te], preds, df.loc[te, "person_id"].to_numpy(), n_boot=2000,
                     seed=config.SEED)
    out["test"] = test
    for name, info in out["models"].items():
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics({f"test_{k}": test[name][k]["value"]
                                for k in ("auroc", "auprc", "brier")})
    (config.REPORTS / "benchmark.json").write_text(json.dumps(out, indent=2))
    return out


def summary(out: dict) -> dict:
    return {m: {k: out["test"][m][k] for k in ("auroc", "auprc", "brier")}
            for m in out["models"]} | {
        k: v for k, v in out["test"].items() if k.endswith("_auroc")} | {
        "environment": out["environment"]}
