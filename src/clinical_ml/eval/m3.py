"""M3: explanations (TreeSHAP) and a subgroup fairness audit for the primary model.

Pre-specified protocol (fixed before running on test):
* The explained model is the M1 artifact logged in MLflow. It is loaded back and must
  reproduce the stored predictions exactly.
* The recalibration and threshold come from M2. The recalibrator is refitted on the
  calibration split and must reproduce M2's threshold exactly.
* Explanations are on the log-odds of the recalibrated risk. Platt scaling is affine in
  log-odds (a + b * raw), so every SHAP value is the raw one times b: the ranking is
  unchanged and the explanation adds up to the risk the user sees.
* Global importance is the mean |SHAP| over test encounters.
* Three example encounters are chosen by rule: the highest-risk encounter, the
  median-risk flagged encounter, and the median-risk readmission the threshold missed.
* Fairness audit: see eval/fairness.py.
"""

import json
import os

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clinical_ml import config  # noqa: E402
from clinical_ml.eval import calibration, decision, fairness, treeshap  # noqa: E402
from clinical_ml.eval.m2 import GRID, INK, INK_2, SURFACE, _style  # noqa: E402
from clinical_ml.models.train import feature_frame  # noqa: E402
from clinical_ml.omop import schema  # noqa: E402

N_BOOT = 1000
TOP_K = 15
EXAMPLE_TOP = 6
RAISES, LOWERS, NEUTRAL, MARK = "#eb6834", "#2a78d6", "#a3a29d", "#2a78d6"
AUDIT_COLS = {"race": "race", "ethnicity": "ethnicity", "audit_sex": "sex",
              "audit_age_band": "age_band", "payer": "payer"}
ATTR_LABELS = {"race": "Race", "ethnicity": "Ethnicity", "sex": "Sex",
               "age_group": "Age", "payer_group": "Payer code"}

LABELS = {
    "sex": "Sex", "age_band": "Age band", "admission_type": "Admission type",
    "admission_source": "Admission source", "discharge_disposition": "Discharge disposition",
    "specialty": "Admitting specialty", "length_of_stay": "Length of stay (days)",
    "prior_inpatient": "Inpatient visits, prior year",
    "prior_emergency": "Emergency visits, prior year",
    "prior_outpatient": "Outpatient visits, prior year", "n_lab_tests": "Lab tests",
    "n_procedures": "Procedures", "n_medications": "Distinct medications",
    "n_diagnoses": "Diagnoses recorded", "n_dose_changes": "Diabetes dose changes",
    "n_diabetes_drugs": "Diabetes drugs", "on_insulin": "On insulin",
    "primary_dx_group": "Primary diagnosis group", "a1c": "HbA1c result",
    "glucose": "Max serum glucose",
}


def label(feature: str) -> str:
    if feature in LABELS:
        return LABELS[feature]
    if feature.startswith("drug_"):
        return f"{feature[5:].capitalize()} (dose status)"
    if feature.startswith("sec_dx_"):
        return f"Secondary dx: {feature[7:]}"
    return feature


def _display(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _short(text: str, n: int = 32) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _load() -> pd.DataFrame:
    audit = ", ".join(f"a.{src} AS {dst}" if src != dst else f"a.{src}" for src, dst in
                      (("race", "race"), ("ethnicity", "ethnicity"), ("sex", "audit_sex"),
                       ("age_band", "audit_age_band"), ("payer", "payer")))
    with schema.connect(read_only=True) as con:
        return con.execute(f"""
            SELECT f.*, p.split, p.p_hist_gradient_boosting AS p_stored, {audit}
            FROM {config.RESULTS}.d130_features f
            JOIN {config.RESULTS}.d130_predictions p USING (visit_occurrence_id, person_id)
            JOIN {config.RESULTS}.d130_audit a USING (visit_occurrence_id)
            ORDER BY visit_occurrence_id""").df()


def _examples(q, y, flag, phi, base, X_raw: pd.DataFrame) -> dict:
    def median_of(idx):
        idx = idx[np.argsort(q[idx], kind="stable")]
        return int(idx[len(idx) // 2])

    picks = {"highest_risk": int(np.argmax(q)),
             "median_flagged": median_of(np.flatnonzero(flag)),
             "median_missed_readmission": median_of(np.flatnonzero((y == 1) & ~flag))}
    out = {}
    for name, i in picks.items():
        order = np.argsort(-np.abs(phi[i]), kind="stable")
        top = order[:EXAMPLE_TOP]
        out[name] = {
            "row": i, "risk": float(q[i]), "readmitted": bool(y[i]), "flagged": bool(flag[i]),
            "base_log_odds": base,
            "top": [{"feature": X_raw.columns[j], "label": label(X_raw.columns[j]),
                     "value": _display(X_raw.iat[i, j]), "shap": float(phi[i, j])}
                    for j in top],
            "other_features_shap": float(phi[i, order[EXAMPLE_TOP:]].sum()),
            "n_other_features": int(len(order) - EXAMPLE_TOP),
        }
    return out


def importance_figure(glob: list[dict], path) -> None:
    top = glob[:TOP_K][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh([g["label"] for g in top], [g["mean_abs_shap"] for g in top], height=0.6,
            color=MARK)
    ax.set_xlabel("Mean |SHAP value| (log-odds of recalibrated risk)", color=INK_2, fontsize=9)
    ax.set_title(f"Top {TOP_K} features by mean impact, held-out test encounters", color=INK,
                 fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def examples_figure(examples: dict, path) -> None:
    titles = {"highest_risk": "Highest-risk encounter",
              "median_flagged": "Typical flagged encounter",
              "median_missed_readmission": "Typical missed readmission"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), facecolor=SURFACE)
    for ax, (name, e) in zip(axes, examples.items(), strict=True):
        # each panel gets its own symmetric axis: one dominant feature would flatten the rest
        lim = max(abs(v) for v in [t["shap"] for t in e["top"]] + [e["other_features_shap"]])
        lim *= 1.15
        _style(ax)
        ax.grid(axis="y", visible=False)
        rows = [(f"{t['label']} = {_short(t['value'])}", t["shap"],
                 RAISES if t["shap"] > 0 else LOWERS) for t in e["top"]]
        rows.append((f"Other {e['n_other_features']} features", e["other_features_shap"],
                     NEUTRAL))
        rows = rows[::-1]
        colors = [r[2] for r in rows]
        ax.barh([r[0] for r in rows], [r[1] for r in rows], height=0.6, color=colors)
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_xlim(-lim, lim)
        ax.tick_params(axis="y", labelsize=8)
        outcome = "readmitted" if e["readmitted"] else "not readmitted"
        ax.set_title(f"{titles[name]}: risk {e['risk']:.3f}, "
                     f"{'flagged' if e['flagged'] else 'not flagged'}, {outcome}",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("SHAP value (log-odds)", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RAISES, LOWERS, NEUTRAL)]
    fig.legend(handles, ["Raises risk", "Lowers risk", "Sum of remaining features"],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Per-encounter explanations: each adds up from the average log-odds\n"
                 "to that encounter's recalibrated risk", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fairness_figure(aud: dict, path) -> None:
    panels = (("auroc", "AUROC"), ("observed_over_expected", "Observed / expected"),
              ("sensitivity", "Sensitivity at threshold"), ("alert_rate", "Alert rate"))
    rows = [(attr, g) for attr, a in aud["attributes"].items() for g in a["reported"]]
    ypos, labels, y, prev = [], [], 0.0, None
    for attr, g in rows:
        if prev is not None and attr != prev:
            y += 0.8
        ypos.append(y)
        labels.append(f"{ATTR_LABELS[attr]}: {g}")
        prev, y = attr, y + 1
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.32 * len(rows) + 1.6),
                             sharey=True, facecolor=SURFACE)
    for ax, (m, title) in zip(axes, panels, strict=True):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(aud["overall"][m], color=INK_2, lw=1, ls="--")
        for yy, (attr, g) in zip(ypos, rows, strict=True):
            v = aud["attributes"][attr]["by_group"][g][m]
            ax.plot(v["ci95"], [yy, yy], color=MARK, lw=1.5, solid_capstyle="round")
            ax.plot(v["value"], yy, "o", color=MARK, ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
        if m == "observed_over_expected":
            ax.axvline(1, color=GRID, lw=1)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(ypos, labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Subgroup audit, held-out test encounters (95% patient-level bootstrap CI; "
                 "dashed line = overall)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run() -> dict:
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    m1 = json.loads((config.REPORTS / "metrics_m1.json").read_text())
    m2 = json.loads((config.REPORTS / "metrics_m2.json").read_text())
    primary = m2["primary_model_for_m3"]
    assert primary == "hist_gradient_boosting", "TreeSHAP here is for the gradient-boosting model"
    m2p = m2["models"][primary]

    df = _load()
    feat_cols = [c for c in df.columns if c not in ("p_stored", *AUDIT_COLS)]
    X = feature_frame(df[feat_cols])
    model = mlflow.sklearn.load_model(f"runs:/{m1['models'][primary]['run_id']}/model")
    p = model.predict_proba(X)[:, 1]
    assert np.max(np.abs(p - df["p_stored"])) < 1e-12, "logged model != stored predictions"

    cal, te = (df["split"] == "calibration").to_numpy(), (df["split"] == "test").to_numpy()
    y = df["readmit_30d"].astype(int).to_numpy()
    method = m2p["recalibration"]["selected"]
    assert method in ("none", "platt"), "explanations need an affine recalibration"
    recal = calibration.Calibrator(method).fit(p[cal], y[cal])
    t = decision.capacity_threshold(recal.predict(p[cal]), m2["protocol"]["capacity"])
    assert abs(t - m2p["operating_point"]["threshold"]) < 1e-12, "recalibration drifted from M2"
    a, b = ((float(recal.model.intercept_[0]), float(recal.model.coef_[0, 0]))
            if method == "platt" else (0.0, 1.0))

    Xt, yt, qt = X[te], y[te], recal.predict(p[te])
    raw = model.decision_function(Xt)
    assert np.max(np.abs(calibration.logit(p[te]) - raw)) < 1e-8
    phi_raw, ev = treeshap.shap_values(model, Xt)
    phi, base = b * phi_raw, a + b * ev
    ours_err = float(np.max(np.abs(base + phi.sum(1) - calibration.logit(qt))))

    import shap

    Z, _ = treeshap._encode(model, Xt)
    lib = shap.TreeExplainer(model)
    lib_err = float(np.max(np.abs(float(np.ravel(lib.expected_value)[0])
                                  + lib.shap_values(Z, check_additivity=False).sum(1) - raw)))

    mean_abs = np.abs(phi).mean(0)
    glob = [{"feature": c, "label": label(c), "mean_abs_shap": float(v)}
            for c, v in sorted(zip(X.columns, mean_abs, strict=True), key=lambda r: -r[1])]
    flag = qt >= t
    raw_vals = df.loc[te, list(X.columns)].reset_index(drop=True)
    examples = _examples(qt, yt, flag, phi, base, raw_vals)

    attrs = fairness.coarsen(df.loc[te, list(AUDIT_COLS)].rename(columns=AUDIT_COLS)
                             .reset_index(drop=True))
    aud = fairness.audit(yt, qt, flag, df.loc[te, "person_id"].to_numpy(), attrs, N_BOOT,
                         config.SEED)

    out = {
        "protocol": {"n_boot": N_BOOT, "bootstrap_unit": "patient",
                     "min_group_encounters": fairness.MIN_ENCOUNTERS,
                     "min_group_readmissions": fairness.MIN_EVENTS,
                     "test_encounters": int(te.sum()),
                     "test_patients": int(df.loc[te, "person_id"].nunique())},
        "model": {"name": primary, "m1_run_id": m1["models"][primary]["run_id"],
                  "recalibration": method, "platt_intercept": a, "platt_slope": b,
                  "threshold": t, "test_alert_rate": float(flag.mean())},
        "shap": {"scale": "log-odds of recalibrated risk", "base_log_odds": base,
                 "global": glob, "examples": examples,
                 "checks": {"treeshap_max_additivity_error": ours_err,
                            "shap_library_version": shap.__version__,
                            "shap_library_max_additivity_error": lib_err}},
        "fairness": aud,
    }

    with schema.connect() as con:
        sv = pd.DataFrame(phi, columns=[f"shap_{c}" for c in X.columns])
        sv.insert(0, "base_log_odds", base)
        sv.insert(0, "visit_occurrence_id", df.loc[te, "visit_occurrence_id"].to_numpy())
        con.register("_sv", sv)
        con.execute(f"CREATE OR REPLACE TABLE {config.RESULTS}.d130_shap_test AS "
                    "SELECT * FROM _sv")
        con.unregister("_sv")

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {"importance": config.FIGURES / "shap_importance.png",
             "examples": config.FIGURES / "shap_examples.png",
             "fairness": config.FIGURES / "fairness.png"}
    importance_figure(glob, paths["importance"])
    examples_figure(examples, paths["examples"])
    fairness_figure(aud, paths["fairness"])
    (config.REPORTS / "metrics_m3.json").write_text(json.dumps(out, indent=2))

    mlflow.set_experiment("readmission-d130")
    with mlflow.start_run(run_name="m3-shap-fairness") as run:
        mlflow.set_tags({"milestone": "M3", "primary_model": primary,
                         "m1_run": out["model"]["m1_run_id"]})
        mlflow.log_params({"n_boot": N_BOOT, "min_group_encounters": fairness.MIN_ENCOUNTERS,
                           "min_group_readmissions": fairness.MIN_EVENTS})
        mlflow.log_metrics({"treeshap_max_additivity_error": ours_err,
                            "shap_library_max_additivity_error": lib_err}
                           | {f"{attr}.gap_{m}": a_["gaps"][m]["value"]
                              for attr, a_ in aud["attributes"].items()
                              for m in ("sensitivity", "observed_over_expected", "alert_rate")
                              if np.isfinite(a_["gaps"][m]["value"])})
        for path in (*paths.values(), config.REPORTS / "metrics_m3.json"):
            mlflow.log_artifact(str(path))
        out["run_id"] = run.info.run_id
    (config.REPORTS / "metrics_m3.json").write_text(json.dumps(out, indent=2))
    return out
