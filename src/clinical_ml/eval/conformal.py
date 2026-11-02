"""Split conformal prediction sets for a binary outcome (LAC score: 1 - p_hat(y)).

Calibrated on the calibration split with the *raw* model probabilities from the M1 model
(trained on train only), so the calibration split is untouched by the model and the
conformal quantile is valid. Three procedures:

* marginal:  one quantile; guarantees P(y in set) >= 1 - alpha on average over everyone.
* class:     Mondrian by true label; the guarantee holds separately for readmitted and
             non-readmitted patients (important at ~11% prevalence).
* group:     Mondrian by a demographic group; the guarantee holds within each group.

Caveat: the guarantee assumes exchangeable units. Patients were split at random, but a
patient can contribute several encounters, so encounter-level coverage is close to, not
exactly, guaranteed. Measured test coverage is reported with patient-bootstrap CIs.
"""

import numpy as np


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _scores(p1: np.ndarray) -> np.ndarray:
    """Nonconformity of label 0 and label 1 for each row: shape (n, 2)."""
    return np.column_stack([p1, 1 - p1])  # 1 - p(0) = p1 ; 1 - p(1) = 1 - p1


def fit(p1_cal, y_cal, alpha: float, by: np.ndarray | None = None) -> dict:
    """Return quantile(s). `by` is the Mondrian key per calibration row (label or group)."""
    s = _scores(p1_cal)[np.arange(len(y_cal)), y_cal]
    if by is None:
        return {"_all": _quantile(s, alpha)}
    return {k: _quantile(s[by == k], alpha) for k in np.unique(by)}


def predict_sets(p1, q: dict, by_label: bool = False, groups=None) -> np.ndarray:
    """Boolean array (n, 2): is label 0 / label 1 in the set."""
    s = _scores(p1)
    if by_label:
        return np.column_stack([s[:, 0] <= q[0], s[:, 1] <= q[1]])
    if groups is not None:
        thr = np.array([q.get(g, max(q.values())) for g in groups])
        return s <= thr[:, None]
    return s <= q["_all"]


def summarize(sets: np.ndarray, y: np.ndarray, groups=None) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    out = {
        "coverage": float(covered.mean()),
        "coverage_readmitted": float(covered[y == 1].mean()),
        "coverage_not_readmitted": float(covered[y == 0].mean()),
        "mean_set_size": float(size.mean()),
        "share_empty": float((size == 0).mean()),
        "share_singleton_no_readmit": float((sets[:, 0] & ~sets[:, 1]).mean()),
        "share_singleton_readmit": float((sets[:, 1] & ~sets[:, 0]).mean()),
        "share_both": float((size == 2).mean()),
    }
    if groups is not None:
        out["coverage_by_group"] = {
            str(g): {"coverage": float(covered[groups == g].mean()),
                     "n": int((groups == g).sum())}
            for g in np.unique(groups)
        }
    return out
