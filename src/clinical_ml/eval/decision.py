"""Operating point and clinical utility.

* Capacity threshold: a transitional-care team can follow up a fixed share of discharges
  (20%). The threshold is the matching quantile of calibrated risk on the calibration
  split, frozen before test is scored.
* Decision-curve analysis (Vickers & Elkin, 2006): net benefit of acting on the model at
  threshold probability t, vs. treating everyone or no one.
"""

import numpy as np


def capacity_threshold(p_cal: np.ndarray, capacity: float) -> float:
    return float(np.quantile(p_cal, 1 - capacity))


def confusion_metrics(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "flag_rate": div(tp + fp, len(y)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    flagged = p >= t
    n = len(y)
    tp = np.sum(flagged & (y == 1)) / n
    fp = np.sum(flagged & (y == 0)) / n
    return float(tp - fp * t / (1 - t))


def treat_all_net_benefit(y: np.ndarray, t: float) -> float:
    prev = float(np.mean(y))
    return prev - (1 - prev) * t / (1 - t)


def decision_curve(y, p, thresholds) -> list[dict]:
    return [{"threshold": float(t), "model": net_benefit(y, p, t),
             "treat_all": treat_all_net_benefit(y, t), "treat_none": 0.0}
            for t in thresholds]
