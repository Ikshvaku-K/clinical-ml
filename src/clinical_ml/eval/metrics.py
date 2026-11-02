"""Discrimination metrics with patient-level (cluster) bootstrap confidence intervals.

Patients contribute several encounters, so resampling rows would understate the
uncertainty; we resample patients and keep all of their encounters.
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

METRICS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
    "log_loss": log_loss,
}


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {name: float(fn(y, p)) for name, fn in METRICS.items()}
    out["prevalence"] = float(np.mean(y))
    out["n"] = int(len(y))
    return out


def _cluster_indices(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap_fn(fn, arrays: dict[str, np.ndarray], groups, n_boot=1000, seed=0,
                 alpha=0.05) -> dict:
    """Patient-level bootstrap of any fn(**arrays) -> dict[str, float]. Returns, per key,
    the point estimate and percentile CI."""
    groups = np.asarray(groups)
    clusters = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    point = fn(**arrays)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        res = fn(**{k: v[idx] for k, v in arrays.items()})
        for k in draws:
            draws[k].append(res[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"value": float(point[k]),
                "ci95": [float(np.nanpercentile(draws[k], lo)),
                         float(np.nanpercentile(draws[k], hi))]}
            for k in point}


def bootstrap(y, preds: dict[str, np.ndarray], groups, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """Point estimates and percentile CIs for each model, plus paired differences of every
    later model vs the first one (same resamples, so the comparison is paired)."""
    y, groups = np.asarray(y), np.asarray(groups)
    clusters = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    names = list(preds)
    draws = {m: {k: [] for k in METRICS} for m in names}
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m in names:
            pb = preds[m][idx]
            for k, fn in METRICS.items():
                draws[m][k].append(fn(yb, pb))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for m in names:
        point = point_metrics(y, preds[m])
        out[m] = {k: {"value": point[k],
                      "ci95": [float(np.percentile(draws[m][k], lo)),
                               float(np.percentile(draws[m][k], hi))]}
                  for k in METRICS}
        out[m]["prevalence"], out[m]["n"] = point["prevalence"], point["n"]
    base = names[0]
    for m in names[1:]:
        d = np.asarray(draws[m]["auroc"]) - np.asarray(draws[base]["auroc"])
        out[f"{m}_minus_{base}_auroc"] = {
            "value": out[m]["auroc"]["value"] - out[base]["auroc"]["value"],
            "ci95": [float(np.percentile(d, lo)), float(np.percentile(d, hi))],
        }
    out["bootstrap"] = {"n_boot": n_boot, "unit": "patient", "n_patients": len(clusters)}
    return out
