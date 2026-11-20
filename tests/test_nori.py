import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from clinical_ml import config, report
from clinical_ml.eval import calibration
from clinical_ml.models import nori, train

RNG = np.random.default_rng(5)
TAUS = (np.arange(999) + 1) / 1000


def test_id_columns_match_m1():
    assert nori.ID_COLS == train.ID_COLS


def test_p_one_reads_the_mass_above_one_half():
    step = np.where(TAUS < 0.7, 0.0, 1.0)            # 70% of the mass at 0, 30% at 1
    Q = np.stack([np.zeros(999), np.ones(999), np.linspace(-0.5, 1.5, 999), step,
                  np.full(999, 0.5)])
    p = nori.p_one(Q, TAUS)
    assert p[0] == 0.0 and p[1] == 1.0 and p[4] == 0.0     # bank top at 0.5 -> no mass above
    assert p[2] == pytest.approx(0.5, abs=1e-3)
    assert p[3] == pytest.approx(0.3, abs=1e-3)


def test_p_one_matches_nori_cell_masses():
    discretize = pytest.importorskip("synthefy_nori.discretize")
    Q = np.sort(RNG.normal(0.2, 0.4, (50, 999)), axis=1)
    ref = discretize.cell_masses(Q, TAUS, np.array([0.0, 1.0]))[:, 1]
    np.testing.assert_allclose(nori.p_one(Q, TAUS), ref, atol=1e-12)


def test_folds_are_the_m1_grouped_folds_on_train_only():
    n = 300
    groups = RNG.integers(0, 90, n)
    split = pd.Series(np.where(RNG.random(n) < 0.7, "train", "test"))
    y = RNG.integers(0, 2, n)
    fold = nori.assign_folds(split, y, groups)
    tr = (split == "train").to_numpy()
    assert (fold[~tr] == -1).all() and set(fold[tr]) == set(range(nori.N_FOLDS))
    assert pd.DataFrame({"g": groups[tr], "f": fold[tr]}).groupby("g")["f"].nunique().max() == 1
    X = pd.DataFrame({"a": RNG.random(tr.sum())})         # as train.grouped_cv calls it
    for k, (_, va) in enumerate(GroupKFold(n_splits=5).split(X, y[tr], groups[tr])):
        assert (fold[np.flatnonzero(tr)[va]] == k).all()


def _encounters(n_patients=400, split="test"):
    person = np.repeat(np.arange(n_patients), RNG.integers(1, 4, n_patients))
    n = len(person)
    risk = RNG.random(n)
    y = (RNG.random(n) < 0.05 + 0.3 * risk).astype(int)
    return pd.DataFrame({"visit_occurrence_id": np.arange(n), "person_id": person,
                         "readmit_30d": y, "split": split, "risk": risk})


def _write_scores(tmp_path, monkeypatch):
    """features.parquet meta columns and Nori scores: p1 is informative, mean is noise."""
    monkeypatch.setattr(nori, "OUT", tmp_path)
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    tr, cal = _encounters(300, "train"), _encounters(300, "calibration")
    cal["visit_occurrence_id"] += 10_000
    cal["person_id"] += 10_000
    df = pd.concat([tr, cal], ignore_index=True)
    df["fold"] = nori.assign_folds(df["split"], df["readmit_30d"].to_numpy(),
                                   df["person_id"].to_numpy())
    df[nori.META].to_parquet(tmp_path / "features.parquet", index=False)
    s = pd.DataFrame({"visit_occurrence_id": df["visit_occurrence_id"], "fold": df["fold"],
                      "mean": RNG.normal(0.1, 0.3, len(df)), "p1": 0.05 + 0.3 * df["risk"]})
    s[df["split"] == "train"].to_parquet(tmp_path / "scores_cv.parquet", index=False)
    s[df["split"] == "calibration"].to_parquet(tmp_path / "scores_calibration.parquet",
                                               index=False)


def test_choose_uses_train_cv_and_the_calibration_split_only(tmp_path, monkeypatch):
    _write_scores(tmp_path, monkeypatch)
    out = nori.choose()
    assert out["score"] == "p1"
    assert out["cv_auroc_train"]["p1"]["mean"] > out["cv_auroc_train"]["mean"]["mean"]
    assert len(out["cv_auroc_train"]["p1"]["folds"]) == nori.N_FOLDS
    assert out["recalibration"]["selected"] in ("none", "platt", "isotonic")
    assert json.loads((tmp_path / "nori_choices.json").read_text())["score"] == "p1"


def test_test_split_is_predicted_once_and_only_after_choose(tmp_path, monkeypatch):
    monkeypatch.setattr(nori, "OUT", tmp_path)
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    with pytest.raises(SystemExit, match="choose"):
        nori.check_test_allowed()
    (tmp_path / "nori_choices.json").write_text("{}")
    nori.check_test_allowed()
    (tmp_path / "scores_test.parquet").write_text("")
    with pytest.raises(SystemExit, match="once"):
        nori.check_test_allowed()


def test_out_of_range_means_are_clipped_and_counted():
    p, n = nori.to_probability(np.array([-0.2, 0.0, 0.4, 1.0, 1.3]))
    assert n == 2 and p.min() == 0.0 and p.max() == 1.0


def test_context_is_a_fixed_seeded_sample_of_the_pool(monkeypatch):
    monkeypatch.setattr(nori, "CONTEXT_ROWS", 50)
    tr = _encounters(100, "train")
    tr["fold"] = nori.assign_folds(tr["split"], tr["readmit_30d"].to_numpy(),
                                   tr["person_id"].to_numpy())
    pool = tr[tr["fold"] != 0]                       # CV: the other folds only
    a, b = nori.context_sample(pool), nori.context_sample(pool)
    assert len(a) == 50 and a.index.equals(b.index) and a.index.is_monotonic_increasing
    assert (a["fold"] != 0).all()
    assert nori.context_sample(tr.iloc[:30]).equals(tr.iloc[:30])


def test_placement_settings_cannot_change_the_protocol(monkeypatch):
    monkeypatch.setenv("NORI_PLACEMENT", '{"host_budget_absolute_gb": 10.5}')
    assert nori.memory_policy() == nori.MEMORY | {"host_budget_absolute_gb": 10.5}
    for key in ("allow_subsample", "allow_quantization", "cache_dtype"):
        monkeypatch.setenv("NORI_PLACEMENT", json.dumps({key: True}))
        with pytest.raises(SystemExit, match="placement"):
            nori.memory_policy()
    monkeypatch.delenv("NORI_PLACEMENT")
    assert nori.memory_policy() == nori.MEMORY


def _frames():
    cal, test = _encounters(500, "calibration"), _encounters(500, "test")
    for d in (cal, test):
        d["p_logistic_regression"] = np.clip(d["risk"] * 0.3 + RNG.normal(0, 0.05, len(d)),
                                             0.01, 0.99)
        d["p_hist_gradient_boosting"] = np.clip(0.05 + 0.3 * d["risk"], 0.01, 0.99)
        d["p1"] = d["p_hist_gradient_boosting"]            # Nori identical to HGB
        d["mean"] = d["p1"] - 0.1                          # some values below 0
    return cal, test


def test_compare_pairs_nori_with_the_stored_m1_models(monkeypatch):
    monkeypatch.setattr(nori, "N_BOOT", 200)
    monkeypatch.setattr(nori, "N_BOOT_CAL", 100)
    cal, test = _frames()
    method, _ = calibration.select_method(cal["p1"].to_numpy(), cal["readmit_30d"].to_numpy(),
                                          cal["person_id"].to_numpy())
    out = nori.compare({"score": "p1", "recalibration": {"selected": method}}, cal, test)
    d = out["discrimination"]
    assert d["nori"]["auroc"] == d["hist_gradient_boosting"]["auroc"]
    assert d["nori_minus_hist_gradient_boosting_auroc"] == {"value": 0.0, "ci95": [0.0, 0.0]}
    assert "nori_minus_logistic_regression_auroc" in d
    assert out["recalibration_methods"]["nori"] == out["recalibration_methods"][
        "hist_gradient_boosting"] == method
    assert out["recalibrated_brier"]["nori_minus_hist_gradient_boosting"]["value"] == 0.0
    assert out["test_clipped"] == 0
    clipped = nori.compare({"score": "mean", "recalibration": {"selected": "platt"}}, cal, test)
    assert clipped["test_clipped"] == int((test["mean"] < 0).sum()) > 0


def test_nori_block_renders_from_json(tmp_path, monkeypatch):
    monkeypatch.setattr(nori, "N_BOOT", 100)
    monkeypatch.setattr(nori, "N_BOOT_CAL", 100)
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    cal, test = _frames()
    choices = {"score": "p1", "recalibration": {"selected": "platt"},
               "cv_auroc_train": {"mean": {"mean": 0.61, "sd": 0.01},
                                  "p1": {"mean": 0.65, "sd": 0.008}}}
    call = {"context_rows": 10000, "context_pool_rows": 59341, "query_rows": 20047,
            "seconds": 812.4, "device": "mps",
            "memory_report": {"rung": "resident_bf16", "dropped_context_rows": 0}}
    runs = {"calls": [call], "versions": {"synthefy_nori": "0.23.0", "torch": "2.9.0",
                                          "python": "3.12"}}
    out = {"choices": choices, **nori.compare(choices, cal, test), "run_id": "0123456789ab",
           "runs": {s: runs for s in ("cv", "calibration", "test")},
           "model": {"name": nori.MODEL, "revision": nori.REVISION}}
    (tmp_path / "benchmark_nori.json").write_text(json.dumps(out))
    best = {"best": {"cv_auroc_mean": 0.66, "cv_auroc_sd": 0.007}, "run_id": "abcdef012345"}
    (tmp_path / "metrics_m1.json").write_text(json.dumps(
        {"models": {m: best for m in nori.M1_MODELS}}))
    c = {k: {"value": 0.1, "ci95": [0.09, 0.11]} for k in
         ("brier", "calibration_slope", "calibration_intercept", "ece")}
    (tmp_path / "metrics_m2.json").write_text(json.dumps({"models": {
        m: {"recalibration": {"selected": "platt"}, "calibration_test": {"recalibrated": c}}
        for m in nori.M1_MODELS}}))
    block = report.render_nori()
    assert block.startswith(report.NORI_START) and block.endswith(report.NORI_END)
    assert "a seeded sample of 10,000 of the 59,341 train encounters" in block
    assert "no context rows dropped" in block
    assert "`resident_bf16`" in block and "`79afc5a6`" in block and "`mps`" in block
    assert "| Nori 1.5, no training (P(y = 1) from the predicted distribution) |" in block
    assert "- Nori − Histogram gradient boosting: 0.0000 (0.0000–0.0000)" in block
    assert "Selected: P(y = 1) from the predicted distribution." in block
    assert block.count("| platt |") == 3 and "clipped" not in block
    assert "XGBoost" not in block                   # no refit report yet

    test["p_xgboost"] = test["p1"]
    rerun = {"environment": {"xgboost_version": "3.4.1", "gpu": "NVIDIA GeForce RTX 3060 Ti"},
             "device_used": "cuda:0", "same_hyperparameters": True, "test_auroc_difference": 0.0,
             "best": {"params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100}}}
    (tmp_path / "benchmark_nori_xgboost.json").write_text(json.dumps(
        {"discrimination": nori.pair_with_xgboost("p1", test), "xgboost_rerun": rerun}))
    block = report.render_nori()
    assert "- Nori − XGBoost: 0.0000 (0.0000–0.0000) (added after these results" in block
    assert "the same hyperparameters as M2b (learning_rate=0.1, max_depth=3, " in block
    assert "is identical to M2b's" in block and "RTX 3060 Ti" in block


def test_nori_pairs_with_xgboost_on_the_m1_resamples(monkeypatch):
    monkeypatch.setattr(nori, "N_BOOT", 200)
    cal, test = _frames()
    test["p_xgboost"] = test["p_hist_gradient_boosting"]
    out = nori.pair_with_xgboost("p1", test)
    assert out["nori_minus_xgboost_auroc"] == {"value": 0.0, "ci95": [0.0, 0.0]}
    ref = nori.compare({"score": "p1", "recalibration": {"selected": "platt"}}, cal, test)
    assert out["nori"] == ref["discrimination"]["nori"]      # same resamples as compare()
