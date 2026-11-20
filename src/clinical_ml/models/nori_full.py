"""M2d: Nori with every train row (added after the M2c results and the learning curve).

On the 8 GB GPU one Nori pass reads at most 10,000 context rows. Here the train pool is cut
into disjoint windows of at most 10,000 rows that together hold every row; Nori scores each
query against each window, and the score is the mean of the windows' P(y = 1). This is an
average of Nori passes, not one pass over all of train. (Nori's own large-table policies
return point predictions only, so they cannot give the M2c score.)

Fixed before any window score existed:
* windows: the pool in the learning curve's seeded order. The first window is the M2c context
  (its M2c scores are reused, checked by hash); the rest of the pool is split into equal
  windows of at most 10,000 rows. A CV pool (the other four folds) gives 5 windows, all of
  train gives 6. Every pool row is in exactly one window;
* score: the unweighted mean of the windows' P(y = 1); recalibration chosen on the
  calibration split by grouped CV Brier, as in M2c;
* gate: calibration and test are scored only if the train-CV AUROC (mean of the 5 fold
  AUROCs) is higher than XGBoost's on the same folds with every row (the learning curve's
  `xgboost@all`);
* test, once: as M2c, on the M1 bootstrap resamples, paired with LR, HGB, XGBoost and the
  M2c (10,000-row) Nori.

  predict   .venv-nori  cv | calibration | test -> data/nori_full/scores_<step>.parquet
  choose    .venv       gate (after cv) and recalibration (after calibration)
                        -> reports/nori_full_choices.json
  evaluate  .venv       -> reports/benchmark_nori_full.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.models import curve, nori

OUT = config.DATA / "nori_full"
CHOICES = config.REPORTS / "nori_full_choices.json"
STEPS = ("cv", "calibration", "test")


def windows(order: pd.DataFrame, size: int = nori.CONTEXT_ROWS) -> list[pd.DataFrame]:
    """Disjoint windows covering `order`: its first `size` rows, then the rest in equal parts
    of at most `size` rows; each window back in its original row order."""
    rest = order.iloc[size:]
    k = -(-len(rest) // size)
    cuts = np.linspace(0, len(rest), k + 1).astype(int)
    return [order.iloc[:size].sort_index()] + [rest.iloc[a:b].sort_index()
                                               for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def parts(step: str):
    """(fold, ordered pool, query) per Nori call group: the 5 CV folds, or all of train."""
    df = pd.read_parquet(nori.OUT / "features.parquet")
    tr = df[df["split"] == "train"]
    if step == "cv":
        yield from curve.folds(tr)
    else:
        yield -1, curve.row_order(tr), df[df["split"] == step]


# ---- predict (.venv-nori) ---------------------------------------------------------------

def check_allowed(step: str) -> None:
    if step == "cv":
        return
    if not CHOICES.exists() or not json.loads(CHOICES.read_text())["gate"]["passed"]:
        raise SystemExit("the CV gate has not been passed: calibration and test stay unread")
    if step == "test":
        if (OUT / "scores_test.parquet").exists():
            raise SystemExit("test scores already exist: the test split is scored once")
        if "recalibration" not in json.loads(CHOICES.read_text()):
            raise SystemExit("choose the recalibration (after calibration) before test")


def predict(step: str) -> dict:
    import synthefy_nori
    import torch

    check_allowed(step)
    m2c = json.loads((nori.OUT / f"scores_{step}.json").read_text())["calls"]
    m2c_scores = pd.read_parquet(nori.OUT / f"scores_{step}.parquet")
    done = OUT / "parts"
    done.mkdir(parents=True, exist_ok=True)
    calls = []
    for i, (k, order, query) in enumerate(parts(step)):
        for w, win in enumerate(windows(order)):
            sha = curve.ids_sha256(win)
            path = done / f"{step}_fold{k}_window{w}.parquet"
            if w == 0:
                assert sha == m2c[i]["context_ids_sha256"], "window 0 is not the M2c context"
                s = m2c_scores[m2c_scores["fold"] == k] if step == "cv" else m2c_scores
                info = {"reused": f"data/nori/scores_{step}.parquet",
                        "seconds": m2c[i]["seconds"], "device": m2c[i]["device"]}
                s = s[["visit_occurrence_id", "p1"]]
            elif path.exists():                       # resume after an interrupted run
                s = pd.read_parquet(path)
                info = json.loads(path.with_suffix(".json").read_text())
            else:
                s, info = nori._predict_once(win, query)
                s = s[["visit_occurrence_id", "p1"]]
                s.to_parquet(path, index=False)
                path.with_suffix(".json").write_text(json.dumps(info, default=str))
            assert len(s) == len(query)
            calls.append({"fold": k, "window": w, "context_rows": len(win),
                          "context_ids_sha256": sha, "query_rows": len(query),
                          **{c: v for c, v in info.items() if c not in ("context_rows",
                                                                        "query_rows")}})
            print(json.dumps({c: v for c, v in calls[-1].items() if c != "memory_report"},
                             default=str), flush=True)
    scores = pd.concat([pd.read_parquet(done / f"{step}_fold{c['fold']}_window{c['window']}"
                                        ".parquet").assign(fold=c["fold"], window=c["window"])
                        if c["window"] else
                        (m2c_scores[m2c_scores["fold"] == c["fold"]] if step == "cv"
                         else m2c_scores)[["visit_occurrence_id", "p1"]]
                        .assign(fold=c["fold"], window=0) for c in calls], ignore_index=True)
    log = {"step": step, "model": nori.MODEL, "revision": nori.REVISION,
           "versions": {"synthefy_nori": synthefy_nori.__version__, "torch": torch.__version__,
                        "python": sys.version.split()[0]},
           "memory_policy": nori.memory_policy(), "calls": calls}
    scores.to_parquet(OUT / f"scores_{step}.parquet", index=False)
    (OUT / f"scores_{step}.json").write_text(json.dumps(log, indent=2, default=str))
    return log


# ---- choose and evaluate (.venv) --------------------------------------------------------

def averaged(step: str) -> pd.DataFrame:
    """One row per encounter: the mean P(y = 1) over windows, with label, patient and fold."""
    s = pd.read_parquet(OUT / f"scores_{step}.parquet")
    n_win = s.groupby("visit_occurrence_id")["window"].nunique()
    expected = s.groupby("fold")["window"].nunique()
    assert (s.groupby("visit_occurrence_id")["fold"].first().map(expected) == n_win).all(), \
        "an encounter is missing a window"
    p = s.groupby("visit_occurrence_id")["p1"].mean().rename("p_nori_full")
    meta = pd.read_parquet(nori.OUT / "features.parquet", columns=nori.META)
    out = meta.join(p, on="visit_occurrence_id", how="inner")
    assert len(out) == len(p)
    return out.sort_values("visit_occurrence_id", ignore_index=True)


def cv_comparison(cv: pd.DataFrame, n_boot: int = curve.N_BOOT) -> dict:
    """Train-CV AUROC of the window average beside the learning curve's whole-pool models and
    the 10,000-row Nori, paired on the same patient resamples of train."""
    from clinical_ml.eval.metrics import bootstrap_fn

    wide = curve.load_scores().merge(cv[["visit_occurrence_id", "p_nori_full"]],
                                     on="visit_occurrence_id", validate="1:1")
    cols = ["p_nori_full", "nori@10000", *(f"{m}@all" for m in curve.TREES)]

    def stats(readmit_30d, fold, **p):
        res = {c: curve.fold_mean_auroc(readmit_30d, fold, p[c]) for c in cols}
        return res | {f"nori_full - {c}": res["p_nori_full"] - res[c] for c in cols[1:]}

    arrays = {"readmit_30d": wide["readmit_30d"].to_numpy(), "fold": wide["fold"].to_numpy(),
              **{c: wide[c].to_numpy() for c in cols}}
    out = bootstrap_fn(stats, arrays, wide["person_id"].to_numpy(), n_boot=n_boot,
                       seed=config.SEED)
    folds = {c: [curve.auroc(g["readmit_30d"].to_numpy(), g[c].to_numpy())
                 for _, g in wide.groupby("fold")] for c in cols}
    for c in cols:
        out[c]["folds"] = folds[c]
    for c in cols[1:]:
        d = [a - b for a, b in zip(folds["p_nori_full"], folds[c], strict=True)]
        out[f"nori_full - {c}"] |= {"folds": d, "folds_full_ahead": sum(x > 0 for x in d)}
    return out


def choose() -> dict:
    """After cv: the gate. After calibration (if the gate passed): the recalibration."""
    from clinical_ml.eval import calibration

    cv = averaged("cv")
    comp = cv_comparison(cv)
    full, xgb = comp["p_nori_full"]["value"], comp["xgboost@all"]["value"]
    out = {"cv": comp,
           "gate": {"rule": "train-CV AUROC of the window average > XGBoost's with every row "
                            "(learning curve xgboost@all)",
                    "nori_full": full, "xgboost_all": xgb, "passed": bool(full > xgb)}}
    if out["gate"]["passed"] and (OUT / "scores_calibration.parquet").exists():
        cal = averaged("calibration")
        method, cv_brier = calibration.select_method(
            cal["p_nori_full"].to_numpy(), cal["readmit_30d"].to_numpy(),
            cal["person_id"].to_numpy())
        out["recalibration"] = {"selected": method, "cv_brier_on_calibration_split": cv_brier}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    CHOICES.write_text(json.dumps(out, indent=2))
    return out


def evaluate() -> dict:
    """Test metrics, once: M2c's comparison with the window average as the Nori score, then
    paired differences against XGBoost and the M2c Nori on the same resamples."""
    from clinical_ml.eval.metrics import bootstrap
    from clinical_ml.models.benchmark import XGB_PREDICTIONS

    choices = json.loads(CHOICES.read_text())
    m1 = nori._m1_predictions()
    xgb = pd.read_parquet(XGB_PREDICTIONS)

    def frame(step):
        f = averaged(step).rename(columns={"p_nori_full": "p1"})
        f = f.merge(m1[m1["split"] == step].drop(columns="split"), on="visit_occurrence_id",
                    validate="1:1")
        return f.merge(nori._scores(step)[["visit_occurrence_id", "p1"]]
                       .rename(columns={"p1": "p_nori_10k"}), on="visit_occurrence_id",
                       validate="1:1")

    cal, test = frame("calibration"), frame("test")
    test = test.merge(xgb[xgb["split"] == "test"].drop(columns="split"),
                      on="visit_occurrence_id", validate="1:1")
    assert len(test) == len(averaged("test")), "test rows do not line up"
    out = nori.compare({"score": "p1", "recalibration": choices["recalibration"]}, cal, test)
    y, g = test["readmit_30d"].to_numpy(), test["person_id"].to_numpy()
    for base in ("xgboost", "nori_10k"):
        pair = bootstrap(y, {base: test[f"p_{base}"].to_numpy(), "nori": test["p1"].to_numpy()},
                         g, n_boot=nori.N_BOOT, seed=config.SEED)
        out["discrimination"][f"nori_minus_{base}_auroc"] = pair[f"nori_minus_{base}_auroc"]
        out["discrimination"].setdefault(base, pair[base])
    m2c = json.loads((config.REPORTS / "benchmark_nori.json").read_text())["discrimination"]
    for m in nori.M1_MODELS:
        assert out["discrimination"][m] == m2c[m], "not the M1 resamples"
    assert out["discrimination"]["nori_10k"] == m2c["nori"], "not the M2c Nori scores"
    out["runs"] = {s: {k: v for k, v in json.loads((OUT / f"scores_{s}.json").read_text())
                       .items() if k != "calls"} for s in STEPS}
    out["windows"] = {s: sorted({c["context_rows"] for c in json.loads(
        (OUT / f"scores_{s}.json").read_text())["calls"]}) for s in STEPS}
    out["test_seconds"] = sum(c["seconds"] for c in json.loads(
        (OUT / "scores_test.json").read_text())["calls"] if c["window"])
    (config.REPORTS / "benchmark_nori_full.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":   # predict runs in .venv-nori: python -m clinical_ml.models.nori_full
    import argparse

    from clinical_ml.rag.models import use_project_hf_cache

    p = argparse.ArgumentParser(prog="clinical_ml.models.nori_full")
    p.add_argument("cmd", choices=["predict"])
    p.add_argument("step", choices=STEPS)
    a = p.parse_args()
    use_project_hf_cache()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    log = predict(a.step)
    print(json.dumps({k: v for k, v in log.items() if k != "calls"}, indent=2, default=str))
