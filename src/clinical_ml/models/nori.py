"""M2c: Nori, a pretrained tabular foundation model, as one pre-declared benchmark row.

Nori (Synthefy, Apache-2.0) predicts by in-context learning: `fit` stores the train rows and
`predict` reads them in a forward pass, with no training. It is regression-only, so it is
given the 0/1 readmission label as a number. The protocol is fixed in the README
("Modelling protocol (M2c)") before the test split is read:
* nori-1-5 from synthefy-nori 0.23.0, weights pinned to one Hugging Face revision, its
  stored inference config; nothing is tuned;
* the context is a fixed sample of 10,000 train encounters drawn with the project seed (from
  the other folds in CV). Amended before any CV score existed: nori-1-5 has no context cache,
  and a pass over all of train does not fit an 8 GB GPU. Nori may not quantise or drop rows
  itself, so a run that does not fit fails instead;
* the risk score is the predictive mean or P(y = 1) from the predicted distribution,
  whichever has the higher grouped 5-fold CV AUROC on train (the folds of the M1 grids);
* recalibration is chosen on the calibration split by grouped CV Brier, as in M2;
* test is scored once: the M1/M2b patient bootstrap, paired with the stored M1 predictions.

Two environments, as for the GPU work (Nori pins its own torch and huggingface-hub < 2):
  export    .venv       features, split and CV folds -> data/nori/features.parquet
  predict   .venv-nori  probe | cv | calibration | test -> data/nori/scores_<step>.parquet
  choose    .venv       score and recalibration, from train CV and calibration only
                        -> reports/nori_choices.json (written before test is predicted)
  evaluate  .venv       test metrics -> reports/benchmark_nori.json
Only `NoriRegressor` is used: the `synthefy` client's default mode sends rows to a hosted
API.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from clinical_ml import config

MODEL = "nori-1-5"
PACKAGE_VERSION = "0.23.0"
REVISION = "79afc5a65b2775d0c62977211a63620ce87846c7"     # Synthefy/Nori-1-5 on the Hub
MEMORY = {"allow_quantization": False, "allow_subsample": False}
CONTEXT_ROWS = 10_000
# Hardware placement only: where the context cache lives and how its build is chunked, set
# per machine through NORI_PLACEMENT (JSON) and logged with every run. It cannot quantise the
# cache or drop rows; Nori documents these paths as changing only floating-point rounding.
PLACEMENT_KEYS = {"offload_to_host", "host_budget_frac", "host_budget_absolute_gb",
                  "gpu_budget_frac", "gpu_budget_absolute_gb", "context_row_chunk",
                  "elements_budget"}
SCORES = {"mean": "predictive mean", "p1": "P(y = 1) from the predicted distribution"}
N_FOLDS = 5
N_BOOT, N_BOOT_CAL = 2000, 1000                # as M1/M2b discrimination, as M2 calibration
ID_COLS = ["visit_occurrence_id", "person_id", "readmit_30d", "split"]   # = train.ID_COLS
META = ID_COLS + ["fold"]
OUT = config.DATA / "nori"
STEPS = ("probe", "cv", "calibration", "test")


# ---- export (.venv) ---------------------------------------------------------------------

def assign_folds(split: pd.Series, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Grouped 5-fold CV folds over the train rows, built exactly as train.grouped_cv builds
    them (same row order, same scikit-learn); -1 outside train."""
    from sklearn.model_selection import GroupKFold

    tr = np.flatnonzero((split == "train").to_numpy())
    fold = np.full(len(split), -1)
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(tr, y[tr], groups[tr])):
        fold[tr[va]] = k
    return fold


def export() -> dict:
    from clinical_ml.models import train
    from clinical_ml.omop import schema

    with schema.connect(read_only=True) as con:
        df = con.execute(f"""
            SELECT f.*, s.split FROM {config.RESULTS}.d130_features f
            JOIN {config.RESULTS}.d130_split s USING (visit_occurrence_id)
            ORDER BY visit_occurrence_id""").df()
    assert (train.patient_split(df["person_id"]) == df["split"]).all(), "split drifted"
    X = train.feature_frame(df)
    out = pd.concat([df[ID_COLS], X], axis=1)
    out["readmit_30d"] = out["readmit_30d"].astype(int)
    out["fold"] = assign_folds(out["split"], out["readmit_30d"].to_numpy(),
                               out["person_id"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "features.parquet", index=False)
    return {"rows": len(out), "features": X.shape[1], "categorical": len(categoricals(X)),
            "splits": out["split"].value_counts().to_dict(),
            "train_rows_per_fold": out.loc[out["fold"] >= 0, "fold"].value_counts()
            .sort_index().tolist()}


def categoricals(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


# ---- predict (.venv-nori) ---------------------------------------------------------------

def p_one(quantiles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """P(y = 1) for a 0/1 target: the predicted distribution's mass above 0.5, i.e.
    1 - CDF(0.5), with the CDF interpolated over each row's quantile bank. This is
    synthefy_nori.discretize.cell_masses(Q, taus, [0, 1])[:, 1] (checked at predict time)."""
    Q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=1)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    cdf = np.array([np.interp(0.5, q, taus, left=0.0, right=1.0) for q in Q])
    cdf = np.where(Q[:, -1] <= 0.5, 1.0, cdf)
    return 1.0 - cdf


def memory_policy() -> dict:
    """The protocol's MEMORY plus this machine's placement settings from NORI_PLACEMENT."""
    extra = json.loads(os.environ.get("NORI_PLACEMENT") or "{}")
    if bad := set(extra) - PLACEMENT_KEYS:
        raise SystemExit(f"NORI_PLACEMENT may set placement keys only, not {sorted(bad)}")
    return MEMORY | extra


def _regressor(cats: list[str]):
    from synthefy_nori import NoriRegressor
    from synthefy_nori.hf import download_checkpoint

    path = download_checkpoint(model=MODEL, revision=REVISION)
    return NoriRegressor(model=MODEL, model_path=path, categorical_columns=cats,
                         memory_policy=memory_policy())


def context_sample(pool: pd.DataFrame) -> pd.DataFrame:
    """The context: CONTEXT_ROWS encounters from `pool`, drawn with the project seed and kept
    in their original order. In CV the pool is the other four folds, so no patient is both
    context and query."""
    if len(pool) <= CONTEXT_ROWS:
        return pool
    return pool.sample(n=CONTEXT_ROWS, random_state=config.SEED).sort_index()


def _predict_once(ctx: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from synthefy_nori.discretize import cell_masses

    X_ctx, X_q = ctx.drop(columns=META), query.drop(columns=META)
    reg = _regressor(categoricals(X_ctx))
    t0 = time.time()
    reg.fit(X_ctx, ctx["readmit_30d"].to_numpy(dtype=float))
    dist = reg.predict(X_q, output_type="full")
    seconds = time.time() - t0
    p1 = p_one(dist["quantiles"], dist["taus"])
    ref = cell_masses(dist["quantiles"], dist["taus"], np.array([0.0, 1.0]))[:, 1]
    assert np.allclose(p1, ref, atol=1e-12), "P(y = 1) disagrees with Nori's cell_masses"
    scores = pd.DataFrame({"visit_occurrence_id": query["visit_occurrence_id"].to_numpy(),
                           "mean": np.asarray(dist["mean"], dtype=float), "p1": p1})
    memory = getattr(reg, "memory_report_", None)
    info = {"context_rows": len(ctx), "query_rows": len(query), "seconds": round(seconds, 1),
            "device": str(getattr(reg, "device_", "unknown")),
            "memory_report": json.loads(json.dumps(memory, default=str))}
    return scores, info


def check_test_allowed() -> None:
    """The test split is predicted once, and only after the score and recalibration have
    been chosen from train CV and the calibration split."""
    if (OUT / "scores_test.parquet").exists():
        raise SystemExit("test scores already exist: the test split is scored once")
    if not (config.REPORTS / "nori_choices.json").exists():
        raise SystemExit("run `nori choose` first: choices are fixed before test is read")


def predict(step: str, context_rows: int | None = None, query_rows: int | None = None) -> dict:
    """Nori scores for one step. `probe` uses train rows only (fold 0 as the query), with
    optional subsampling, to measure speed and memory before the real runs."""
    import synthefy_nori
    import torch

    df = pd.read_parquet(OUT / "features.parquet")
    tr = df[df["split"] == "train"]
    if step == "probe":
        rng = np.random.default_rng(config.SEED)

        def sub(d, n):
            return d if n is None or n >= len(d) else d.iloc[np.sort(rng.choice(len(d), n,
                                                                                 replace=False))]

        parts = [(0, sub(tr[tr["fold"] != 0], context_rows), sub(tr[tr["fold"] == 0],
                                                                  query_rows))]
    elif step == "cv":
        parts = [(k, tr[tr["fold"] != k], tr[tr["fold"] == k]) for k in range(N_FOLDS)]
    else:
        if step == "test":
            check_test_allowed()
        parts = [(-1, tr, df[df["split"] == step])]
    rows, calls = [], []
    for k, pool, query in parts:
        ctx = pool if step == "probe" else context_sample(pool)
        s, info = _predict_once(ctx, query)
        rows.append(s.assign(fold=k))
        ids = np.sort(ctx["visit_occurrence_id"].to_numpy(dtype=np.int64))
        calls.append({"fold": k, "context_pool_rows": len(pool),
                      "context_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), **info})
        print(json.dumps(calls[-1], default=str), flush=True)
    scores = pd.concat(rows, ignore_index=True)
    log = {"step": step, "model": MODEL, "revision": REVISION,
           "versions": {"synthefy_nori": synthefy_nori.__version__, "torch": torch.__version__,
                        "python": sys.version.split()[0]},
           "memory_policy": memory_policy(), "calls": calls,
           "score_range": {c: [float(scores[c].min()), float(scores[c].max())] for c in SCORES},
           "mean_outside_unit_interval": int(((scores["mean"] < 0) | (scores["mean"] > 1)).sum())}
    name = f"probe_{len(parts[0][1])}x{len(parts[0][2])}" if step == "probe" else step
    if step != "probe":
        scores.to_parquet(OUT / f"scores_{name}.parquet", index=False)
    (OUT / f"scores_{name}.json").write_text(json.dumps(log, indent=2))
    return log


# ---- choose and evaluate (.venv) --------------------------------------------------------

def _scores(step: str) -> pd.DataFrame:
    meta = pd.read_parquet(OUT / "features.parquet", columns=META)
    s = pd.read_parquet(OUT / f"scores_{step}.parquet").drop(columns="fold")
    out = meta.merge(s, on="visit_occurrence_id", validate="1:1")
    assert len(out) == len(s), "scores do not match the exported rows"
    return out.sort_values("visit_occurrence_id", ignore_index=True)


def to_probability(score: np.ndarray) -> tuple[np.ndarray, int]:
    """The chosen score as a probability: values outside [0, 1] (possible only for the
    predictive mean) are clipped, and the count is reported."""
    return np.clip(score, 0.0, 1.0), int(((score < 0) | (score > 1)).sum())


def choose() -> dict:
    """The two pre-declared choices, from train CV and the calibration split only."""
    from sklearn.metrics import roc_auc_score

    from clinical_ml.eval import calibration

    cv = _scores("cv")
    auc = {s: [roc_auc_score(g["readmit_30d"], g[s]) for _, g in cv.groupby("fold")]
           for s in SCORES}
    cv_auroc = {s: {"mean": float(np.mean(a)), "sd": float(np.std(a)), "folds": a}
                for s, a in auc.items()}
    score = max(cv_auroc, key=lambda s: cv_auroc[s]["mean"])
    cal = _scores("calibration")
    p_c, clipped = to_probability(cal[score].to_numpy())
    method, cv_brier = calibration.select_method(p_c, cal["readmit_30d"].to_numpy(),
                                                 cal["person_id"].to_numpy())
    out = {"score": score, "cv_auroc_train": cv_auroc,
           "recalibration": {"selected": method, "cv_brier_on_calibration_split": cv_brier},
           "calibration_split_clipped": clipped,
           "rules": {"score": "higher mean grouped 5-fold CV AUROC on train",
                     "recalibration": "lowest grouped 5-fold CV Brier on the calibration "
                                      "split among none, platt, isotonic"}}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "nori_choices.json").write_text(json.dumps(out, indent=2))
    return out


def _m1_predictions() -> pd.DataFrame:
    from clinical_ml.omop import schema

    with schema.connect(read_only=True) as con:
        return con.execute(f"""
            SELECT visit_occurrence_id, split, p_logistic_regression, p_hist_gradient_boosting
            FROM {config.RESULTS}.d130_predictions
            WHERE split IN ('calibration', 'test') ORDER BY visit_occurrence_id""").df()


M1_MODELS = ("logistic_regression", "hist_gradient_boosting")


def compare(choices: dict, cal: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test metrics for Nori beside the stored M1 models. `cal` and `test` hold the Nori
    scores, the label, person_id and the M1 predictions `p_<model>`, one row per encounter."""
    from clinical_ml.eval import calibration
    from clinical_ml.eval.metrics import bootstrap, bootstrap_fn

    score, method = choices["score"], choices["recalibration"]["selected"]
    y_c, y_t = cal["readmit_30d"].to_numpy(), test["readmit_30d"].to_numpy()
    g_c, g_t = cal["person_id"].to_numpy(), test["person_id"].to_numpy()
    p_c, _ = to_probability(cal[score].to_numpy())
    p_t, clipped = to_probability(test[score].to_numpy())
    raw = {m: test[f"p_{m}"].to_numpy() for m in M1_MODELS} | {"nori": p_t}
    disc = bootstrap(y_t, raw, g_t, n_boot=N_BOOT, seed=config.SEED)
    vs_hgb = bootstrap(y_t, {m: raw[m] for m in ("hist_gradient_boosting", "nori")}, g_t,
                       n_boot=N_BOOT, seed=config.SEED)
    disc["nori_minus_hist_gradient_boosting_auroc"] = \
        vs_hgb["nori_minus_hist_gradient_boosting_auroc"]

    # M2's recalibration of each M1 model (method chosen on the calibration split by the
    # same rule), so the recalibrated Brier scores can be compared on paired resamples.
    recal, methods = {}, {}
    for m in M1_MODELS:
        pc = cal[f"p_{m}"].to_numpy()
        methods[m], _ = calibration.select_method(pc, y_c, g_c)
        recal[m] = calibration.Calibrator(methods[m]).fit(pc, y_c).predict(test[f"p_{m}"]
                                                                           .to_numpy())
    methods["nori"] = method
    recal["nori"] = calibration.Calibrator(method).fit(p_c, y_c).predict(p_t)
    nori_cal = {name: bootstrap_fn(lambda y, p: calibration.calibration_metrics(y, p),
                                   {"y": y_t, "p": p}, g_t, N_BOOT_CAL, config.SEED)
                for name, p in (("raw", p_t), ("recalibrated", recal["nori"]))}

    def brier_gaps(y, **p):
        b = {m: float(np.mean((p[m] - y) ** 2)) for m in p}
        return b | {f"nori_minus_{m}": b["nori"] - b[m] for m in M1_MODELS}

    recal_brier = bootstrap_fn(brier_gaps, {"y": y_t, **recal}, g_t, N_BOOT_CAL, config.SEED)
    return {"discrimination": disc, "calibration_nori": nori_cal,
            "recalibrated_brier": recal_brier, "recalibration_methods": methods,
            "test_clipped": clipped}


def evaluate() -> dict:
    import mlflow

    choices = json.loads((config.REPORTS / "nori_choices.json").read_text())
    m1 = _m1_predictions()
    frames = {s: _scores(s).merge(m1[m1["split"] == s].drop(columns="split"),
                                  on="visit_occurrence_id", validate="1:1")
              for s in ("calibration", "test")}
    assert len(frames["test"]) == (m1["split"] == "test").sum(), "test rows do not match M1"
    out = {"choices": choices} | compare(choices, frames["calibration"], frames["test"])
    m1_metrics = json.loads((config.REPORTS / "metrics_m1.json").read_text())["test"]
    m2 = json.loads((config.REPORTS / "metrics_m2.json").read_text())["models"]
    for m in M1_MODELS:   # the stored predictions and recalibrations are the M1/M2 ones
        assert abs(out["discrimination"][m]["auroc"]["value"]
                   - m1_metrics[m]["auroc"]["value"]) < 1e-12, m
        assert out["recalibration_methods"][m] == m2[m]["recalibration"]["selected"], m
        assert abs(out["recalibrated_brier"][m]["value"]
                   - m2[m]["calibration_test"]["recalibrated"]["brier"]["value"]) < 1e-12, m
    logs = {s: json.loads((OUT / f"scores_{s}.json").read_text())
            for s in ("cv", "calibration", "test")}
    out["runs"] = {s: {"calls": v["calls"], "versions": v["versions"],
                       "memory_policy": v["memory_policy"]} for s, v in logs.items()}
    out["model"] = {"name": MODEL, "revision": REVISION, "memory_policy": MEMORY}

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("readmission-d130")
    with mlflow.start_run(run_name="benchmark-nori") as run:
        mlflow.set_tags({"milestone": "M2c-benchmark", **logs["test"]["versions"]})
        mlflow.log_params({"model": MODEL, "revision": REVISION, "score": choices["score"],
                           "recalibration": choices["recalibration"]["selected"]})
        nori = out["discrimination"]["nori"]
        mlflow.log_metrics({f"test_{k}": nori[k]["value"] for k in ("auroc", "auprc", "brier")}
                           | {"cv_auroc_mean": choices["cv_auroc_train"][choices["score"]]
                              ["mean"]})
        out["run_id"] = run.info.run_id
    (config.REPORTS / "benchmark_nori.json").write_text(json.dumps(out, indent=2))
    return out


def summary(out: dict) -> dict:
    d = out["discrimination"]
    return {m: {k: d[m][k] for k in ("auroc", "auprc", "brier")}
            for m in (*M1_MODELS, "nori")} | {
        k: v for k, v in d.items() if k.endswith("_auroc")} | {
        "recalibrated_brier": out["recalibrated_brier"]}


if __name__ == "__main__":   # predict runs in .venv-nori:  python -m clinical_ml.models.nori
    import argparse

    from clinical_ml.rag.models import use_project_hf_cache

    p = argparse.ArgumentParser(prog="clinical_ml.models.nori")
    p.add_argument("cmd", choices=["predict"])
    p.add_argument("step", choices=STEPS)
    p.add_argument("--context", type=int, help="probe only: subsample the context rows")
    p.add_argument("--query", type=int, help="probe only: subsample the query rows")
    a = p.parse_args()
    use_project_hf_cache()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    log = predict(a.step, a.context, a.query)
    print(json.dumps({k: v for k, v in log.items() if k != "calls"}, indent=2))
