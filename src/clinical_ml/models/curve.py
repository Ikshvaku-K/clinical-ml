"""Learning curve (exploratory, added after the M2c results): train-CV AUROC against the
number of training rows, for Nori and the three trained models given the same rows.

Why: on the 8 GB GPU Nori could read a 10,000-row sample of train as context, while the
trained models used every row. This asks whether that, rather than the model, is the gap.

Design, fixed before any curve score existed:
* the M2c grouped 5-fold CV on train. For fold k the pool is the other four folds, put in
  one nested seeded order: Nori's M2c context (10,000 rows) shuffled, then the rest of the
  pool shuffled. Size n is the first n rows of that order, so every model sees the same rows
  at each size, and the 10,000-row set is exactly the M2c context. The query is always the
  whole of fold k;
* Nori: nori-1-5 with the M2c settings and score (P(y = 1)), at 1,000, 2,500, 5,000 and
  10,000 rows (the 8 GB limit). The 10,000 point is the M2c CV run, checked by context hash;
* LR, HGB and XGBoost: their M1/M2b grids, re-selected at every size by grouped 5-fold CV
  inside the given rows and refit on them, at the same sizes plus 20,000 and the whole pool;
* per size: the mean over folds of the fold AUROC, with patient-level bootstrap 95% CIs and
  the paired Nori − model differences on the same resamples. The test split is not read.

  nori      .venv-nori  -> data/curve/nori.parquet (+ .json log)
  trees     .venv       -> data/curve/<model>.parquet (+ .json log); XGBoost needs Linux
  evaluate  .venv       -> reports/learning_curve.json, reports/figures/learning_curve.png
"""

import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.models import nori

OUT = config.DATA / "curve"
NORI_SIZES = (1000, 2500, 5000, nori.CONTEXT_ROWS)
TREE_SIZES = (*NORI_SIZES, 20_000, None)          # None: the whole pool
TREES = ("logistic_regression", "hist_gradient_boosting", "xgboost")
N_BOOT = 1000


def size_label(n: int | None) -> str:
    return "all" if n is None else str(n)


def train_rows() -> pd.DataFrame:
    df = pd.read_parquet(nori.OUT / "features.parquet")
    return df[df["split"] == "train"]


def row_order(pool: pd.DataFrame) -> pd.DataFrame:
    """The pool in one nested seeded order: the M2c context first, then everything else."""
    ctx = nori.context_sample(pool)
    rest = pool.drop(index=ctx.index)
    return pd.concat([ctx.sample(frac=1, random_state=config.SEED),
                      rest.sample(frac=1, random_state=config.SEED)])


def rows(order: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """The first n rows of the order (all of it for None), back in their original order."""
    return (order if n is None else order.iloc[:n]).sort_index()


def folds(tr: pd.DataFrame):
    for k in range(nori.N_FOLDS):
        yield k, row_order(tr[tr["fold"] != k]), tr[tr["fold"] == k]


def ids_sha256(d: pd.DataFrame) -> str:
    ids = np.sort(d["visit_occurrence_id"].to_numpy(dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()


def _write(name: str, scores: pd.DataFrame, log: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(log, indent=2, default=str))


# ---- nori (.venv-nori) ------------------------------------------------------------------

def predict_nori() -> dict:
    import synthefy_nori
    import torch

    m2c = json.loads((nori.OUT / "scores_cv.json").read_text())["calls"]
    m2c_scores = pd.read_parquet(nori.OUT / "scores_cv.parquet")
    out, calls = [], []
    for k, order, query in folds(train_rows()):
        for n in NORI_SIZES:
            ctx = rows(order, n)
            sha = ids_sha256(ctx)
            if n == nori.CONTEXT_ROWS:
                assert sha == m2c[k]["context_ids_sha256"], "not the M2c context"
                s = m2c_scores[m2c_scores["fold"] == k].drop(columns="fold")
                info = {"reused": "data/nori/scores_cv.parquet",
                        "seconds": m2c[k]["seconds"], "device": m2c[k]["device"]}
            else:
                s, info = nori._predict_once(ctx, query)
            out.append(s[["visit_occurrence_id", "p1"]].rename(columns={"p1": "p"})
                       .assign(fold=k, size=size_label(n), rows=len(ctx)))
            calls.append({"fold": k, "size": n, "context_ids_sha256": sha, **info})
            print(json.dumps({c: v for c, v in calls[-1].items() if c != "memory_report"},
                             default=str), flush=True)
    log = {"model": nori.MODEL, "revision": nori.REVISION, "score": "p1",
           "versions": {"synthefy_nori": synthefy_nori.__version__, "torch": torch.__version__,
                        "python": sys.version.split()[0]},
           "memory_policy": nori.memory_policy(), "calls": calls}
    _write("nori", pd.concat(out, ignore_index=True), log)
    return log


# ---- trees (.venv) ----------------------------------------------------------------------

def _features(d: pd.DataFrame) -> pd.DataFrame:
    from clinical_ml.models import train

    return train.feature_frame(d.drop(columns="fold"))


def fit_trees(models=TREES, device: str | None = None) -> dict:
    import sklearn

    from clinical_ml.models import benchmark, train

    device = device or "cpu"
    makers = {"logistic_regression": (train.make_lr, train.LR_GRID),
              "hist_gradient_boosting": (train.make_hgb, train.HGB_GRID),
              "xgboost": (lambda X_, **p: benchmark.make_xgb(X_, device=device, **p),
                          benchmark.XGB_GRID)}
    tr = train_rows()
    logs = {}
    for model in models:
        make, grid = makers[model]
        env = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
               "sklearn_version": sklearn.__version__}
        if model == "xgboost":
            import xgboost

            env |= {"xgboost_version": xgboost.__version__, "device": device}
        out, fits, t0 = [], [], time.time()
        for k, order, query in folds(tr):
            Xq = _features(query)
            for n in TREE_SIZES:
                fit = rows(order, n)
                X, y = _features(fit), fit["readmit_30d"].to_numpy()
                best, _ = train.grouped_cv(make, grid, X, y, fit["person_id"].to_numpy())
                m = make(X, **best["params"]).fit(X, y)
                out.append(pd.DataFrame({"visit_occurrence_id": query["visit_occurrence_id"]
                                         .to_numpy(), "p": m.predict_proba(Xq)[:, 1]})
                           .assign(fold=k, size=size_label(n), rows=len(fit)))
                fits.append({"fold": k, "size": n, "rows": len(fit),
                             "rows_sha256": ids_sha256(fit), **best})
                print(json.dumps({"model": model, **fits[-1]}), flush=True)
        logs[model] = {"environment": env, "seconds": round(time.time() - t0, 1),
                       "fits": fits}
        _write(model, pd.concat(out, ignore_index=True), logs[model])
    return logs


# ---- evaluate (.venv) -------------------------------------------------------------------

def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC from average ranks (ties count one half), equal to roc_auc_score."""
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rankdata(p)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fold_mean_auroc(y: np.ndarray, fold: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([auroc(y[fold == k], p[fold == k]) for k in np.unique(fold)]))


def load_scores() -> pd.DataFrame:
    """One row per train encounter: label, patient, fold, and a column `<model>@<size>`."""
    wide = train_rows()[["visit_occurrence_id", "person_id", "readmit_30d", "fold"]]
    for model in ("nori", *TREES):
        s = pd.read_parquet(OUT / f"{model}.parquet")
        for size, g in s.groupby("size", sort=False):
            col = g.set_index("visit_occurrence_id")["p"].rename(f"{model}@{size}")
            assert col.index.is_unique and len(col) == len(wide), f"{col.name}: rows differ"
            wide = wide.join(col, on="visit_occurrence_id", validate="1:1")
    assert not wide.isna().any().any()
    return wide


def summarise(wide: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    from clinical_ml.eval.metrics import bootstrap_fn

    cols = [c for c in wide.columns if "@" in c]
    pairs = [(f"nori@{s}", f"{m}@{s}") for s in map(size_label, NORI_SIZES) for m in TREES
             if f"{m}@{s}" in cols]

    def stats(readmit_30d, fold, **p):
        res = {c: fold_mean_auroc(readmit_30d, fold, p[c]) for c in cols}
        return res | {f"{a} - {b}": res[a] - res[b] for a, b in pairs}

    arrays = {"readmit_30d": wide["readmit_30d"].to_numpy(), "fold": wide["fold"].to_numpy(),
              **{c: wide[c].to_numpy() for c in cols}}
    boot = bootstrap_fn(stats, arrays, wide["person_id"].to_numpy(), n_boot=n_boot,
                        seed=config.SEED)
    per_fold = {c: [auroc(g["readmit_30d"].to_numpy(), g[c].to_numpy())
                    for _, g in wide.groupby("fold")] for c in cols}
    for c in cols:
        boot[c]["folds"] = per_fold[c]
    for a, b in pairs:
        d = [x - y for x, y in zip(per_fold[a], per_fold[b], strict=True)]
        boot[f"{a} - {b}"] |= {"folds": d, "folds_nori_ahead": sum(x > 0 for x in d)}
    return boot


def evaluate() -> dict:
    wide = load_scores()
    res = summarise(wide)
    logs = {m: json.loads((OUT / f"{m}.json").read_text()) for m in ("nori", *TREES)}
    rows_by_size = {}
    for m in ("nori", *TREES):
        s = pd.read_parquet(OUT / f"{m}.parquet", columns=["size", "rows", "fold"])
        for size, g in s.drop_duplicates(["size", "fold"]).groupby("size"):
            rows_by_size[size] = float(g["rows"].mean())
    devices = {c["device"] for c in logs["nori"]["calls"]}
    assert len(devices) == 1, f"Nori ran on several devices: {devices}"
    out = {"nori_device": devices.pop(),
           "design": {"nori_sizes": list(NORI_SIZES),
                      "tree_sizes": [size_label(n) for n in TREE_SIZES],
                      "n_boot": N_BOOT, "unit": "patient", "score": "mean of 5 fold AUROCs",
                      "n_patients": int(wide["person_id"].nunique()), "rows": len(wide)},
           "mean_rows": rows_by_size,
           "results": res,
           "runs": {"nori": {k: v for k, v in logs["nori"].items() if k != "calls"}
                    | {"seconds": {f"{c['fold']}@{size_label(c['size'])}": c["seconds"]
                                   for c in logs["nori"]["calls"]}},
                    **{m: {"environment": logs[m]["environment"],
                           "seconds": logs[m]["seconds"],
                           "selected": [{k: f[k] for k in ("fold", "size", "params",
                                                           "cv_auroc_mean")}
                                        for f in logs[m]["fits"]]} for m in TREES}}}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "learning_curve.json").write_text(json.dumps(out, indent=2))
    figure(out, config.REPORTS / "figures" / "learning_curve.png")
    return out


def summary(out: dict) -> dict:
    r = out["results"]
    return {k: {"value": round(v["value"], 4), "ci95": [round(x, 4) for x in v["ci95"]]}
            | ({"folds_nori_ahead": v["folds_nori_ahead"]} if "folds_nori_ahead" in v else {})
            for k, v in r.items()}


# ---- figure -----------------------------------------------------------------------------

LABELS = {"nori": "Nori 1.5 (context)", "xgboost": "XGBoost",
          "hist_gradient_boosting": "Histogram gradient boosting",
          "logistic_regression": "Logistic regression"}
# Validated (all pairs, light surface): CVD dE 9.2, normal-vision dE 24.0. Logistic
# regression is a grey dashed reference line, not a fourth hue; every line has its own
# marker and a direct label, which also covers the aqua's 2.7:1 contrast.
STYLE = {"xgboost": ("#2a78d6", "o", "-"), "nori": ("#eb6834", "D", "-"),
         "hist_gradient_boosting": ("#1baf7a", "s", "-"),
         "logistic_regression": ("#52514e", "^", "--")}


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so none sit closer than `gap`, keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:], strict=True):
        out[b] = max(out[b], out[a] + gap)
    return list(out + (np.array(ys)[order].mean() - out[order].mean()))


def figure(out: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from clinical_ml.eval.m2 import GRID, INK, INK_2, SURFACE, _style

    r, mean_rows = out["results"], out["mean_rows"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    nudge = {"logistic_regression": 0.955, "hist_gradient_boosting": 0.985, "xgboost": 1.015,
             "nori": 1.045}                          # dodge the CI bars apart on the log axis
    ends = []
    for m in ("logistic_regression", "hist_gradient_boosting", "xgboost", "nori"):
        sizes = [str(s) for s in (out["design"]["nori_sizes"] if m == "nori"
                                  else out["design"]["tree_sizes"])]
        x =np.array([mean_rows[s] for s in sizes]) * nudge[m]
        v = np.array([r[f"{m}@{s}"]["value"] for s in sizes])
        lo = v - np.array([r[f"{m}@{s}"]["ci95"][0] for s in sizes])
        hi = np.array([r[f"{m}@{s}"]["ci95"][1] for s in sizes]) - v
        color, marker, ls = STYLE[m]
        ax.errorbar(x, v, yerr=[lo, hi], color=color, lw=2, ls=ls, marker=marker, ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1, capsize=0,
                    label=LABELS[m], zorder=3 if m == "nori" else 2)
        ends.append((m, x[-1], v[-1]))
    ax.set_xscale("log")
    ticks = [mean_rows[size_label(s)] for s in TREE_SIZES]
    ax.set_xticks(ticks, [f"{t / 1000:g}k" if i < len(ticks) - 1 else f"all\n(~{t / 1000:.0f}k)"
                          for i, t in enumerate(ticks)])
    ax.minorticks_off()
    limit = mean_rows[str(nori.CONTEXT_ROWS)]
    ax.axvline(limit, color=GRID, lw=6, zorder=0)
    ax.text(limit * 0.93, ax.get_ylim()[1], "Nori's context limit\non the 8 GB GPU",
            color=INK_2, fontsize=8, ha="right", va="top")
    trees = [e for e in ends if e[0] != "nori"]          # these end together at "all"
    ys = _spread([e[2] for e in trees], gap=0.004)
    x_end = max(e[1] for e in trees)
    for (m, _, _), y in zip(trees, ys, strict=True):
        ax.annotate(LABELS[m], (x_end, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    m, x, y = next(e for e in ends if e[0] == "nori")   # Nori ends alone, at its limit
    ax.annotate(LABELS[m], (x, y), xytext=(-10, 10), textcoords="offset points", ha="right",
                va="bottom", fontsize=8, color=INK)
    ax.set_xlim(ticks[0] * 0.8, ticks[-1] * 2.6)
    ax.set_xlabel("Training rows given to the model (log scale; Nori reads them as context)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("AUROC, mean of 5 grouped CV folds", color=INK_2, fontsize=9)
    ax.set_title("Learning curve on train: the same rows for every model", color=INK,
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":   # nori runs in .venv-nori:  python -m clinical_ml.models.curve nori
    from clinical_ml.rag.models import use_project_hf_cache

    if sys.argv[1:] != ["nori"]:
        raise SystemExit("usage: python -m clinical_ml.models.curve nori")
    use_project_hf_cache()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    log = predict_nori()
    print(json.dumps({k: v for k, v in log.items() if k != "calls"}, indent=2))
