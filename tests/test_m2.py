import numpy as np
import pandas as pd
import pytest

from clinical_ml.eval import calibration, conformal, decision
from clinical_ml.eval.metrics import bootstrap_fn
from clinical_ml.models import benchmark

RNG = np.random.default_rng(7)


def _calibrated(n=20000):
    p = RNG.beta(1.2, 8, n)
    return p, (RNG.random(n) < p).astype(int)


def test_calibration_metrics_on_calibrated_data():
    p, y = _calibrated()
    m = calibration.calibration_metrics(y, p)
    assert abs(m["calibration_slope"] - 1) < 0.1
    assert abs(m["calibration_intercept"]) < 0.05
    assert abs(m["observed_over_expected"] - 1) < 0.05
    assert m["ece"] < 0.01


def test_platt_repairs_overconfident_scores():
    p, y = _calibrated()
    lp = calibration.logit(p)
    over = 1 / (1 + np.exp(-2 * lp))           # logit scaled by 2 = overconfident
    assert calibration.calibration_metrics(y, over)["calibration_slope"] < 0.6
    fixed = calibration.Calibrator("platt").fit(over, y).predict(over)
    assert abs(calibration.calibration_metrics(y, fixed)["calibration_slope"] - 1) < 0.1
    groups = np.arange(len(y)) // 2
    method, scores = calibration.select_method(over, y, groups)
    assert method in ("platt", "isotonic") and scores["none"] > scores[method]


def test_conformal_marginal_and_class_coverage():
    p, y = _calibrated(40000)
    cal, test = slice(0, 20000), slice(20000, None)
    alpha = 0.1
    q = conformal.fit(p[cal], y[cal], alpha)
    s = conformal.summarize(conformal.predict_sets(p[test], q), y[test])
    assert s["coverage"] >= 1 - alpha - 0.01
    qc = conformal.fit(p[cal], y[cal], alpha, by=y[cal])
    sc = conformal.summarize(conformal.predict_sets(p[test], qc, by_label=True), y[test])
    assert sc["coverage_readmitted"] >= 1 - alpha - 0.02
    assert sc["coverage_not_readmitted"] >= 1 - alpha - 0.02
    # class-conditional buys minority-class coverage with bigger sets
    assert sc["coverage_readmitted"] > s["coverage_readmitted"]
    assert sc["mean_set_size"] >= s["mean_set_size"]


def test_group_conditional_unseen_group_is_conservative():
    p, y = _calibrated(4000)
    g = np.where(np.arange(4000) % 2, "a", "b")
    q = conformal.fit(p, y, 0.1, by=g)
    sets = conformal.predict_sets(p[:5], q, groups=np.array(["z"] * 5))
    assert sets.shape == (5, 2)


def test_decision_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .6, .7, .1, .1, .1, .1, .1, .1, .2])
    m = decision.confusion_metrics(y, p >= 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["ppv"] == pytest.approx(2 / 3) and m["sensitivity"] == pytest.approx(2 / 3)
    assert m["flag_rate"] == pytest.approx(0.3)
    # net benefit at t=0.5: TP/n - FP/n * 1 = 0.2 - 0.1
    assert decision.net_benefit(y, p, 0.5) == pytest.approx(0.1)
    assert decision.treat_all_net_benefit(y, 0.5) == pytest.approx(0.3 - 0.7)
    t = decision.capacity_threshold(np.linspace(0, 1, 1001), 0.2)
    assert t == pytest.approx(0.8)


def test_bootstrap_fn_brackets_point():
    p, y = _calibrated(5000)
    out = bootstrap_fn(lambda y, p: {"mean_p": p.mean(), "prev": y.mean()},
                       {"y": y, "p": p}, np.arange(5000) // 3, n_boot=200, seed=0)
    for v in out.values():
        assert v["ci95"][0] <= v["value"] <= v["ci95"][1]


available, _ = benchmark.xgboost_available()


def test_benchmark_refuses_cleanly_without_xgboost():
    if available:
        pytest.skip("xgboost is importable here")
    with pytest.raises(SystemExit, match="Linux"):
        benchmark.run()


@pytest.mark.skipif(not available, reason="xgboost runs on Linux (CI/Docker or GPU node)")
def test_xgboost_handles_categorical_features():
    X = pd.DataFrame({"a": pd.Categorical(RNG.choice(["x", "y", "z"], 500)),
                      "b": RNG.random(500)})
    y = ((X["a"] == "x") | (X["b"] > 0.8)).astype(int).to_numpy()
    m = benchmark.make_xgb(X, learning_rate=0.1, max_depth=3, n_estimators=50).fit(X, y)
    assert m.predict_proba(X).shape == (500, 2)
