"""Subgroup audit with Fairlearn and patient-level bootstrap CIs.

Pre-specified (fixed before the audit was run on test):
* Population: the test split. Risk is the primary model's recalibrated probability, and
  an encounter is flagged when risk >= the M2 capacity threshold (fixed on calibration).
* Attributes: race, ethnicity, sex, age band and payer. Age band and payer are coarsened
  (below), because their raw levels leave many groups with a handful of readmissions.
* A group with fewer than MIN_ENCOUNTERS test encounters or MIN_EVENTS readmissions is
  listed with its size but gets no metrics: its estimates would be noise.
* Each group is compared with the attribute's largest group (the reference): the signed
  difference is recomputed on every bootstrap resample, so its CI is paired and can
  include zero.
* The largest gap (highest minus lowest group, Fairlearn's between-groups difference) is
  reported as a point estimate only. A max-minus-min statistic is never below zero, so a
  percentile CI for it cannot include zero even when groups do not differ.

Point estimates come from Fairlearn's MetricFrame. A MetricFrame costs ~0.2 s, too slow
for 1,000 resamples x 5 attributes, so the bootstrap uses a vectorised equivalent
(group_metrics), which must match MetricFrame on the full test split or the run fails.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from scipy.stats import rankdata
from sklearn.metrics import precision_score, roc_auc_score

from clinical_ml.eval.metrics import bootstrap_fn

MIN_ENCOUNTERS = 100
MIN_EVENTS = 10
ATTRIBUTES = ("race", "ethnicity", "sex", "age_group", "payer_group")
METRICS = ("auroc", "observed_over_expected", "sensitivity", "false_positive_rate", "ppv",
           "alert_rate")
AGE_GROUPS = {"[0-10)": "<40", "[10-20)": "<40", "[20-30)": "<40", "[30-40)": "<40",
              "[40-50)": "40-59", "[50-60)": "40-59", "[60-70)": "60-79", "[70-80)": "60-79",
              "[80-90)": "80+", "[90-100)": "80+"}
# The source does not document payer codes. MC, MD and SP are the three largest named
# codes; every other recorded code is pooled.
PAYERS = ("MC", "MD", "SP")


def coarsen(audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()
    out["age_group"] = out["age_band"].map(AGE_GROUPS)
    assert out["age_group"].notna().all(), "unmapped age band"
    out["payer_group"] = np.where(out["payer"].isin(PAYERS), out["payer"],
                                  np.where(out["payer"] == "missing", "missing", "other codes"))
    return out


def _auroc(y, flag, risk):
    return roc_auc_score(y, risk) if 0 < y.sum() < len(y) else np.nan


FUNCS = {"auroc": _auroc,
         "observed_over_expected": lambda y, f, risk: y.mean() / risk.mean(),
         "sensitivity": lambda y, f: true_positive_rate(y, f),
         "false_positive_rate": lambda y, f: false_positive_rate(y, f),
         "ppv": lambda y, f: precision_score(y, f, zero_division=np.nan),
         "alert_rate": lambda y, f: selection_rate(y, f)}
NEEDS_RISK = ("auroc", "observed_over_expected")


def metric_frame(y, risk, flag, groups) -> MetricFrame:
    return MetricFrame(
        metrics=FUNCS, y_true=y, y_pred=flag, sensitive_features=groups,
        sample_params={m: {"risk": risk} for m in NEEDS_RISK})


def _ratio(a, b):
    return a / b if b else np.nan


def group_metrics(y, risk, flag) -> dict:
    """The METRICS for one group, in numpy (AUROC as Mann-Whitney U with average ranks,
    which equals roc_auc_score including ties)."""
    pos, n, nf = int(y.sum()), len(y), int(flag.sum())
    tp = int(np.sum(flag & (y == 1)))
    auroc = np.nan
    if 0 < pos < n:
        r = rankdata(risk)
        auroc = (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (n - pos))
    return {"auroc": auroc, "observed_over_expected": y.mean() / risk.mean(),
            "sensitivity": _ratio(tp, pos), "false_positive_rate": _ratio(nf - tp, n - pos),
            "ppv": _ratio(tp, nf), "alert_rate": nf / n}


def reported_groups(y, groups) -> tuple[list[str], dict]:
    """Groups large enough to report, largest first (the first is the reference)."""
    sizes = (pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"]
             .agg(encounters="size", readmissions="sum"))
    keep = sizes[(sizes.encounters >= MIN_ENCOUNTERS) & (sizes.readmissions >= MIN_EVENTS)]
    return sorted(keep.index, key=lambda g: -sizes.loc[g, "encounters"]), {
        g: {"encounters": int(r.encounters), "readmissions": int(r.readmissions)}
        for g, r in sizes.iterrows()}


def _flat(y, risk, flag, groups, keep) -> dict:
    by = {g: group_metrics(y[groups == g], risk[groups == g], flag[groups == g]) for g in keep}
    out = {}
    for m in METRICS:
        out |= {f"{g}|{m}": by[g][m] for g in keep}
        out |= {f"{g}-ref|{m}": by[g][m] - by[keep[0]][m] for g in keep[1:]}
    return out


def _excludes_zero(ci) -> bool:
    return ci[0] > 0 or ci[1] < 0


def audit(y, risk, flag, persons, attrs: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Per-group metrics, differences from the reference group and largest gaps, per
    attribute, with patient-level bootstrap CIs."""
    y, risk, flag = np.asarray(y).astype(int), np.asarray(risk), np.asarray(flag).astype(bool)
    overall = {m: float(v) for m, v in group_metrics(y, risk, flag).items()}
    out = {"overall": overall, "attributes": {}}
    for attr in ATTRIBUTES:
        groups = attrs[attr].to_numpy()
        keep, sizes = reported_groups(y, groups)
        rows = np.isin(groups, keep)
        mf = metric_frame(y[rows], risk[rows], flag[rows], groups[rows])
        ci = bootstrap_fn(lambda y, risk, flag, g, keep=keep: _flat(y, risk, flag, g, keep),
                          {"y": y, "risk": risk, "flag": flag, "g": groups}, persons,
                          n_boot, seed)
        for m in METRICS:
            assert abs(mf.overall[m] - group_metrics(y[rows], risk[rows], flag[rows])[m]) \
                < 1e-12, (attr, m)
            for g in keep:
                assert abs(mf.by_group.loc[g, m] - ci[f"{g}|{m}"]["value"]) < 1e-12, (attr, g, m)

        gaps = {}
        diff = mf.difference(method="between_groups")
        for m in METRICS:
            vals = {g: ci[f"{g}|{m}"]["value"] for g in keep}
            lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
            gaps[m] = {"value": vals[hi] - vals[lo], "lowest": lo, "highest": hi}
            assert abs(gaps[m]["value"] - diff[m]) < 1e-12, (attr, m)
        eod = equalized_odds_difference(y[rows], flag[rows], sensitive_features=groups[rows])
        gaps["equalized_odds"] = {"value": float(eod)}
        assert abs(eod - max(gaps["sensitivity"]["value"],
                             gaps["false_positive_rate"]["value"])) < 1e-12, attr

        vs_ref = {g: {m: ci[f"{g}-ref|{m}"] for m in METRICS} for g in keep[1:]}
        out["attributes"][attr] = {
            "reference": keep[0],
            "reported": keep,
            "too_small": sorted(set(sizes) - set(keep)),
            "sizes": sizes,
            "by_group": {g: {m: ci[f"{g}|{m}"] for m in METRICS} for g in keep},
            "vs_reference": vs_ref,
            "differs_from_reference": [[g, m] for g, d in vs_ref.items() for m, v in d.items()
                                       if _excludes_zero(v["ci95"])],
            "gaps": gaps,
        }
    return out
