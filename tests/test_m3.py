import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import MetricFrame
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from clinical_ml.eval import fairness, m3, treeshap

RNG = np.random.default_rng(11)


def _numeric_model(n=3000):
    X = pd.DataFrame({"a": RNG.random(n), "b": RNG.normal(size=n),
                      "c": RNG.integers(0, 5, n).astype(float), "d": RNG.random(n)})
    X.loc[RNG.random(n) < 0.05, "b"] = np.nan          # exercises missing-value routing
    logit = 2 * X.a - X.b.fillna(0) + (X.c == 2) + X.a * X.d
    y = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=15,
                                          random_state=0).fit(X, y), X


def _categorical_model(n=3000):
    X = pd.DataFrame({"a": RNG.random(n),
                      "cat": pd.Categorical(RNG.choice(list("abcdefg"), n)),
                      "cat2": pd.Categorical(RNG.choice(["u", "v", "w"], n))})
    # a non-ordinal categorical effect, so the trees need bitset splits
    logit = X.a + np.where(X.cat.isin(["a", "d", "g"]), 1.2, -0.8) + (X.cat2 == "v")
    y = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=40,
                                           max_leaf_nodes=15, random_state=0).fit(X, y)
    assert any(p.nodes["is_categorical"].any() for row in model._predictors for p in row)
    return model, X


def test_treeshap_matches_shap_library_on_numeric_trees():
    shap = pytest.importorskip("shap")
    model, X = _numeric_model()
    phi, ev = treeshap.shap_values(model, X)
    explainer = shap.TreeExplainer(model)
    assert np.abs(phi - explainer.shap_values(X, check_additivity=False)).max() < 1e-10
    assert ev == pytest.approx(float(np.ravel(explainer.expected_value)[0]), abs=1e-10)


def test_treeshap_is_exact_with_categorical_splits():
    model, X = _categorical_model()
    assert np.abs(treeshap.raw_predict(model, X) - model.decision_function(X)).max() == 0
    phi, ev = treeshap.shap_values(model, X)            # asserts additivity internally
    assert phi.shape == X.shape
    assert np.abs(phi[:, 1]).mean() > np.abs(phi[:, 0]).mean()   # cat drives the outcome


def test_treeshap_handles_unseen_and_missing_categories():
    model, X = _categorical_model()
    Xu = X.iloc[:40].copy()
    Xu["cat"] = Xu["cat"].cat.add_categories(["zz"])
    Xu.iloc[:10, Xu.columns.get_loc("cat")] = "zz"
    Xu.iloc[10:20, Xu.columns.get_loc("cat2")] = np.nan
    assert np.abs(treeshap.raw_predict(model, Xu) - model.decision_function(Xu)).max() == 0
    treeshap.shap_values(model, Xu)


def test_unused_feature_gets_zero_attribution():
    model, X = _numeric_model()
    X = X.assign(unused=0.0)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(
        X, (X.a > 0.5).astype(int))
    phi, _ = treeshap.shap_values(model, X)
    assert np.all(phi[:, X.columns.get_loc("unused")] == 0)


def _audit_data(n=6000):
    g = RNG.choice(["big", "mid", "tiny"], n, p=[0.6, 0.39, 0.01])
    risk = np.round(RNG.beta(1.2, 8, n), 3)              # rounding creates ties
    y = (RNG.random(n) < risk).astype(int)
    flag = risk >= np.quantile(risk, 0.8)
    # "mid" loses half its true positives: lower sensitivity than the reference
    drop = (g == "mid") & flag & (y == 1) & (RNG.random(n) < 0.5)
    return y, risk, flag & ~drop, g


def test_group_metrics_match_fairlearn_and_sklearn():
    y, risk, flag, g = _audit_data()
    mf = fairness.metric_frame(y, risk, flag, g)
    for grp in ("big", "mid"):
        m = fairness.group_metrics(y[g == grp], risk[g == grp], flag[g == grp])
        assert m["auroc"] == pytest.approx(roc_auc_score(y[g == grp], risk[g == grp]), abs=1e-12)
        for k, v in m.items():
            assert v == pytest.approx(mf.by_group.loc[grp, k], abs=1e-12)
    assert isinstance(mf, MetricFrame)


def test_audit_suppresses_small_groups_and_finds_designed_gap():
    y, risk, flag, g = _audit_data()
    attrs = pd.DataFrame({a: g for a in fairness.ATTRIBUTES})
    out = fairness.audit(y, risk, flag, np.arange(len(y)) // 2, attrs, n_boot=200, seed=0)
    a = out["attributes"]["race"]
    assert a["reference"] == "big" and a["reported"] == ["big", "mid"]
    assert a["too_small"] == ["tiny"]
    assert ["mid", "sensitivity"] in a["differs_from_reference"]
    d = a["vs_reference"]["mid"]["sensitivity"]
    assert d["ci95"][1] < 0 < a["gaps"]["sensitivity"]["value"]
    assert a["gaps"]["sensitivity"]["lowest"] == "mid"


def test_coarsening_covers_every_band_and_pools_payers():
    audit = pd.DataFrame({"age_band": list(fairness.AGE_GROUPS),
                          "payer": ["MC", "MD", "SP", "HM", "missing", "BC", "MC", "OG",
                                    "SP", "UN"]})
    out = fairness.coarsen(audit)
    assert set(out.age_group) == {"<40", "40-59", "60-79", "80+"}
    assert set(out.payer_group) == {"MC", "MD", "SP", "missing", "other codes"}


def test_example_selection_rules():
    q = np.array([0.05, 0.9, 0.3, 0.25, 0.2, 0.1, 0.12])
    y = np.array([1, 0, 1, 0, 0, 1, 1])
    flag = q >= 0.2
    phi = RNG.normal(size=(7, 8))
    X = pd.DataFrame(RNG.integers(0, 3, (7, 8)), columns=[f"f{i}" for i in range(8)])
    ex = m3._examples(q, y, flag, phi, -2.0, X)
    assert ex["highest_risk"]["row"] == 1
    assert ex["median_flagged"]["row"] == 2   # flagged .2 .25 .3 .9: upper median for even n
    assert ex["median_missed_readmission"]["row"] == 5   # missed .05 .1 .12 -> .1
    top = ex["highest_risk"]["top"]
    assert [abs(t["shap"]) for t in top] == sorted((abs(t["shap"]) for t in top), reverse=True)
    total = sum(t["shap"] for t in top) + ex["highest_risk"]["other_features_shap"]
    assert total == pytest.approx(phi[1].sum())
