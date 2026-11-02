"""Calibration: recalibration methods fitted on the calibration split, and the metrics
TRIPOD+AI asks for (calibration-in-the-large, slope, intercept, ECE, Brier).

The recalibration method is chosen *before* touching test, by grouped (patient) 5-fold
CV Brier score on the calibration split, among: none, Platt (logistic on logit p), and
isotonic regression.
"""

import numpy as np
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str):
        assert method in ("none", "platt", "isotonic")
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.model.predict_proba(logit(p)[:, None])[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), EPS, 1 - EPS)
        return p


def select_method(p, y, groups, methods=("none", "platt", "isotonic")) -> tuple[str, dict]:
    """Grouped 5-fold CV Brier on the calibration split; lowest wins."""
    folds = list(GroupKFold(n_splits=5).split(p, y, groups))
    scores = {}
    for m in methods:
        briers = []
        for tr, va in folds:
            q = Calibrator(m).fit(p[tr], y[tr]).predict(p[va])
            briers.append(np.mean((q - y[va]) ** 2))
        scores[m] = float(np.mean(briers))
    return min(scores, key=scores.get), scores


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Slope from y ~ a + b*logit(p); intercept from the offset model y ~ a + logit(p),
    i.e. calibration-in-the-large on the log-odds scale (0 and 1 are ideal)."""
    lp = logit(p)
    slope = float(LogisticRegression(C=1e6).fit(lp[:, None], y).coef_[0, 0])
    target = y.mean()

    def gap(a):
        return np.mean(1 / (1 + np.exp(-(a + lp)))) - target

    intercept = float(brentq(gap, -10, 10))
    return slope, intercept


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over equal-frequency bins."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins))


def calibration_metrics(y, p) -> dict:
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "observed_over_expected": float(y.mean() / p.mean()),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece": ece(y, p),
    }


def reliability_curve(y, p, n_bins: int = 10) -> list[dict]:
    order = np.argsort(p, kind="stable")
    return [{"mean_predicted": float(p[b].mean()), "observed": float(y[b].mean()),
             "n": int(len(b))} for b in np.array_split(order, n_bins)]
