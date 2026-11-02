"""M2: calibration, conformal uncertainty, a capacity-based threshold and decision curves.

Pre-specified protocol (fixed before test was scored for M2):
* Everything is *fitted or chosen* on the calibration split: the recalibration method
  (grouped-CV Brier), the conformal quantiles (alpha = 0.1), and the operating threshold
  (top 20% of calibrated risk = care-management capacity).
* The test split only *reports* those choices, with patient-level bootstrap CIs.
* The model carried into M3 is the one with the lower calibration-split CV Brier after
  its selected recalibration (a calibration-split criterion, not a test result).
"""

import json
import os

import matplotlib
import mlflow
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clinical_ml import config  # noqa: E402
from clinical_ml.eval import calibration, conformal, decision  # noqa: E402
from clinical_ml.eval.metrics import bootstrap_fn  # noqa: E402
from clinical_ml.omop import schema  # noqa: E402

ALPHA = 0.1
CAPACITY = 0.20
N_BOOT = 1000
DCA_THRESHOLDS = np.round(np.arange(0.02, 0.41, 0.01), 2)
DCA_REPORTED = (0.10, 0.15, 0.20, 0.25, 0.30)
MODELS = {"logistic_regression": "Logistic regression",
          "hist_gradient_boosting": "Histogram gradient boosting"}

# Reference palette (validated: CVD dE 24.7, normal-vision dE 33.6, >= 3:1 on surface)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAW, CALIBRATED = "#2a78d6", "#eb6834"


def _load():
    with schema.connect(read_only=True) as con:
        df = con.execute(f"""
            SELECT p.*, a.race FROM {config.RESULTS}.d130_predictions p
            JOIN {config.RESULTS}.d130_audit a USING (visit_occurrence_id)
            WHERE p.split IN ('calibration', 'test') ORDER BY p.visit_occurrence_id
        """).df()
    return {s: df[df.split == s].reset_index(drop=True) for s in ("calibration", "test")}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def reliability_figure(curves: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (model, c) in zip(axes, curves.items(), strict=True):
        _style(ax)
        hi = max(max(b["mean_predicted"] for b in c[k]) for k in ("raw", "calibrated")) * 1.1
        hi = max(hi, max(max(b["observed"] for b in c[k]) for k in ("raw", "calibrated")) * 1.1)
        ax.plot([0, hi], [0, hi], ls="--", lw=1, color=INK_2, label="Perfect calibration")
        lines = [("raw", RAW, "o", "Raw model output")]
        if c["method"] == "none":
            lines[0] = ("raw", RAW, "o", "Raw (no recalibration selected)")
        else:
            lines.append(("calibrated", CALIBRATED, "s", f"Recalibrated ({c['method']})"))
        for key, color, marker, label in lines:
            xs = [b["mean_predicted"] for b in c[key]]
            ys = [b["observed"] for b in c[key]]
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(MODELS[model], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Mean predicted 30-day readmission risk (decile)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axes[0].set_ylabel("Observed readmission rate", color=INK_2, fontsize=9)
    fig.suptitle("Reliability on held-out test patients (equal-frequency deciles)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def decision_figure(curves: dict, threshold: float, primary: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    _style(ax)
    first = next(iter(curves.values()))
    t = [r["threshold"] for r in first]
    ax.axhline(0, color=INK_2, lw=1)
    ax.text(t[-1], -0.007, "Treat none", color=INK_2, fontsize=8, ha="right")
    ax.plot(t, [r["treat_all"] for r in first], ls="--", lw=1.5, color=INK_2)
    # label "Treat all" just right of its line, below the model curves
    i = next(k for k, r in enumerate(first) if r["treat_all"] < 0.6 * first[k]["model"])
    ax.text(t[i] + 0.006, first[i]["treat_all"], "Treat all", color=INK_2, fontsize=8)
    for (model, curve), color, marker in zip(curves.items(), (RAW, CALIBRATED), ("o", "s"),
                                             strict=True):
        # Markers every 5th point give each model a non-colour cue as well.
        ax.plot(t, [r["model"] for r in curve], color=color, lw=2, marker=marker, ms=6,
                markevery=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODELS[model])
    top = max(max(r["model"] for r in c) for c in curves.values())
    ax.set_ylim(-0.01, max(top, first[0]["treat_all"]) * 1.15)
    ax.axvline(threshold, color=INK, lw=1, ls=":")
    ax.text(threshold + 0.004, ax.get_ylim()[1] * 0.80,
            f"20%-capacity threshold\n({MODELS[primary]}) = {threshold:.3f}",
            color=INK, fontsize=8)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Threshold probability (risk at which follow-up is worth it)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("Net benefit", color=INK_2, fontsize=9)
    ax.set_title("Decision curve, held-out test patients (recalibrated risk)", color=INK,
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run() -> dict:
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}")
    d = _load()
    cal, test = d["calibration"], d["test"]
    y_c, y_t = cal.readmit_30d.astype(int).to_numpy(), test.readmit_30d.astype(int).to_numpy()
    g_c, g_t = cal.person_id.to_numpy(), test.person_id.to_numpy()
    race_c, race_t = cal.race.to_numpy(), test.race.to_numpy()
    m1 = json.loads((config.REPORTS / "metrics_m1.json").read_text())

    out = {"protocol": {"alpha": ALPHA, "capacity": CAPACITY, "n_boot": N_BOOT,
                        "bootstrap_unit": "patient",
                        "calibration_split": {"encounters": len(cal),
                                              "patients": int(cal.person_id.nunique())},
                        "test_split": {"encounters": len(test),
                                       "patients": int(test.person_id.nunique())}},
           "models": {}}
    reliability, dca, thresholds = {}, {}, {}
    for model in MODELS:
        p_c, p_t = cal[f"p_{model}"].to_numpy(), test[f"p_{model}"].to_numpy()

        method, cv_brier = calibration.select_method(p_c, y_c, g_c)
        cal_model = calibration.Calibrator(method).fit(p_c, y_c)
        q_c, q_t = cal_model.predict(p_c), cal_model.predict(p_t)

        calib = {name: bootstrap_fn(lambda y, p: calibration.calibration_metrics(y, p),
                                    {"y": y_t, "p": p}, g_t, N_BOOT, config.SEED)
                 for name, p in (("raw", p_t), ("recalibrated", q_t))}
        reliability[model] = {"method": method,
                              "raw": calibration.reliability_curve(y_t, p_t),
                              "calibrated": calibration.reliability_curve(y_t, q_t)}

        # Conformal sets use the raw scores, which the calibration split never touched.
        conf = {}
        for proc, key_c, key_t in (("marginal", None, None), ("class_conditional", y_c, None),
                                   ("race_conditional", race_c, race_t)):
            q = conformal.fit(p_c, y_c, ALPHA, by=key_c)
            sets = conformal.predict_sets(p_t, q, by_label=proc == "class_conditional",
                                          groups=key_t)
            summary = conformal.summarize(sets, y_t, groups=race_t)
            ci = bootstrap_fn(
                lambda s, y: {k: v for k, v in conformal.summarize(s, y).items()
                              if k in ("coverage", "coverage_readmitted", "share_both")},
                {"s": sets, "y": y_t}, g_t, N_BOOT, config.SEED)
            conf[proc] = {"quantiles": {str(k): v for k, v in q.items()}, "test": summary,
                          "ci95": ci}

        t = decision.capacity_threshold(q_c, CAPACITY)
        thresholds[model] = t
        op = bootstrap_fn(
            lambda y, p, t=t: {k: v for k, v in decision.confusion_metrics(y, p >= t).items()
                          if k in ("sensitivity", "specificity", "ppv", "npv", "flag_rate")},
            {"y": y_t, "p": q_t}, g_t, N_BOOT, config.SEED)
        dca[model] = decision.decision_curve(y_t, q_t, DCA_THRESHOLDS)
        nb = bootstrap_fn(
            lambda y, p: {f"nb_model@{x}": decision.net_benefit(y, p, x) for x in DCA_REPORTED}
            | {f"nb_treat_all@{x}": decision.treat_all_net_benefit(y, x) for x in DCA_REPORTED},
            {"y": y_t, "p": q_t}, g_t, N_BOOT, config.SEED)
        useful = [r["threshold"] for r in dca[model]
                  if r["model"] > max(r["treat_all"], 0)]

        out["models"][model] = {
            "m1_run_id": m1["models"][model]["run_id"],
            "recalibration": {"selected": method, "cv_brier_on_calibration_split": cv_brier},
            "calibration_test": calib,
            "conformal_test": conf,
            "operating_point": {"capacity": CAPACITY, "threshold": t,
                                "calibration_split_flag_rate": float(np.mean(q_c >= t)),
                                "test": op, "test_prevalence": float(y_t.mean())},
            "decision_curve": {"net_benefit": nb,
                               "model_beats_both_defaults_from_to":
                                   [min(useful), max(useful)] if useful else None},
        }

    best = min(MODELS, key=lambda m: min(
        out["models"][m]["recalibration"]["cv_brier_on_calibration_split"].values()))
    out["primary_model_for_m3"] = best
    out["primary_rule"] = "lowest calibration-split grouped-CV Brier after recalibration"

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    rel_path, dca_path = config.FIGURES / "reliability.png", config.FIGURES / "decision_curve.png"
    reliability_figure(reliability, rel_path)
    decision_figure(dca, thresholds[best], best, dca_path)
    (config.REPORTS / "metrics_m2.json").write_text(json.dumps(out, indent=2))
    (config.REPORTS / "decision_curve.json").write_text(json.dumps(dca, indent=2))

    mlflow.set_experiment("readmission-d130")
    with mlflow.start_run(run_name="m2-calibration-conformal-threshold") as run:
        mlflow.set_tags({"milestone": "M2", "primary_model": best,
                         **{f"m1_run_{m}": v["m1_run_id"] for m, v in out["models"].items()}})
        mlflow.log_params({"alpha": ALPHA, "capacity": CAPACITY, "n_boot": N_BOOT})
        for m, v in out["models"].items():
            mlflow.log_metrics({
                f"{m}.test_ece_recal": v["calibration_test"]["recalibrated"]["ece"]["value"],
                f"{m}.test_slope_recal":
                    v["calibration_test"]["recalibrated"]["calibration_slope"]["value"],
                f"{m}.test_conformal_coverage":
                    v["conformal_test"]["marginal"]["test"]["coverage"],
                f"{m}.test_ppv_at_capacity": v["operating_point"]["test"]["ppv"]["value"],
                f"{m}.test_sensitivity_at_capacity":
                    v["operating_point"]["test"]["sensitivity"]["value"],
            })
        for p in (rel_path, dca_path, config.REPORTS / "metrics_m2.json",
                  config.REPORTS / "decision_curve.json"):
            mlflow.log_artifact(str(p))
        out["run_id"] = run.info.run_id
    (config.REPORTS / "metrics_m2.json").write_text(json.dumps(out, indent=2))
    return out
