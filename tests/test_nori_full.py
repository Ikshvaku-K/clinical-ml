import json

import numpy as np
import pandas as pd
import pytest

from clinical_ml import config, report
from clinical_ml.eval.metrics import bootstrap
from clinical_ml.models import nori, nori_full

RNG = np.random.default_rng(17)


def test_windows_hold_every_row_once_and_start_with_the_m2c_context():
    order = pd.DataFrame({"x": np.arange(47)}, index=RNG.permutation(np.arange(100, 147)))
    ws = nori_full.windows(order, size=10)
    pd.testing.assert_frame_equal(ws[0], order.iloc[:10].sort_index())
    idx = [set(w.index) for w in ws]
    assert sum(map(len, idx)) == 47 and set().union(*idx) == set(order.index)
    assert [len(w) for w in ws] == [10, 9, 9, 9, 10] and all(w.index.is_monotonic_increasing
                                                             for w in ws)


def test_calibration_and_test_wait_for_the_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(nori_full, "OUT", tmp_path)
    monkeypatch.setattr(nori_full, "CHOICES", tmp_path / "choices.json")
    nori_full.check_allowed("cv")
    for step in ("calibration", "test"):
        with pytest.raises(SystemExit, match="gate"):
            nori_full.check_allowed(step)
    nori_full.CHOICES.write_text(json.dumps({"gate": {"passed": False}}))
    with pytest.raises(SystemExit, match="gate"):
        nori_full.check_allowed("calibration")
    nori_full.CHOICES.write_text(json.dumps({"gate": {"passed": True}}))
    nori_full.check_allowed("calibration")
    with pytest.raises(SystemExit, match="recalibration"):
        nori_full.check_allowed("test")
    nori_full.CHOICES.write_text(json.dumps({"gate": {"passed": True},
                                             "recalibration": {"selected": "platt"}}))
    nori_full.check_allowed("test")
    (tmp_path / "scores_test.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="once"):
        nori_full.check_allowed("test")


def test_score_is_the_mean_over_every_window(tmp_path, monkeypatch):
    monkeypatch.setattr(nori, "OUT", tmp_path)
    monkeypatch.setattr(nori_full, "OUT", tmp_path)
    n = 60
    meta = pd.DataFrame({"visit_occurrence_id": np.arange(n), "person_id": np.arange(n) // 2,
                         "readmit_30d": RNG.integers(0, 2, n), "split": "train",
                         "fold": np.arange(n) % 3})
    meta.to_parquet(tmp_path / "features.parquet", index=False)
    s = pd.concat([meta[["visit_occurrence_id", "fold"]].assign(window=w, p1=RNG.random(n))
                   for w in range(4)], ignore_index=True)
    s.to_parquet(tmp_path / "scores_cv.parquet", index=False)
    out = nori_full.averaged("cv")
    np.testing.assert_allclose(out["p_nori_full"],
                               s.groupby("visit_occurrence_id")["p1"].mean().to_numpy())
    s.iloc[1:].to_parquet(tmp_path / "scores_cv.parquet", index=False)   # one window lost
    with pytest.raises(AssertionError, match="missing a window"):
        nori_full.averaged("cv")


def _cv(passed: bool) -> dict:
    keys = [k for k, _ in report.FULL_CV]
    cv = {k: {"value": 0.66 - 0.002 * i, "ci95": [0.65, 0.67]} for i, k in enumerate(keys)}
    for k in keys[1:]:
        cv[f"nori_full - {k}"] = {"value": 0.003, "ci95": [0.001, 0.005],
                                  "folds_full_ahead": 5}
    return {"cv": cv, "gate": {"nori_full": 0.66, "xgboost_all": 0.662 if not passed else 0.65,
                               "passed": passed}}


def test_full_block_renders_the_gate_and_then_the_test(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    monkeypatch.setattr(nori, "N_BOOT", 100)
    monkeypatch.setattr(nori, "N_BOOT_CAL", 50)
    (tmp_path / "nori_full_choices.json").write_text(json.dumps(_cv(False)))
    block = report.render_nori_full()
    assert block.startswith(report.FULL_START) and block.endswith(report.FULL_END)
    assert "not passed: the test split was not read for M2d" in block
    assert "0.6600 vs 0.6620" in block and "**Test" not in block
    assert block.count("| 5/5 |") == 4 and "| — | — |" in block

    person = np.repeat(np.arange(300), 2)
    y = (RNG.random(600) < 0.15).astype(int)
    frames = []
    for _ in range(2):
        f = pd.DataFrame({"person_id": person, "readmit_30d": y,
                          "p1": np.clip(0.1 + 0.2 * y + RNG.normal(0, 0.1, 600), 0.01, 0.99)})
        for m in ("logistic_regression", "hist_gradient_boosting", "xgboost", "nori_10k"):
            f[f"p_{m}"] = np.clip(0.1 + 0.1 * y + RNG.normal(0, 0.1, 600), 0.01, 0.99)
        frames.append(f)
    cal, test = frames
    choices = _cv(True) | {"recalibration": {"selected": "platt"}}
    (tmp_path / "nori_full_choices.json").write_text(json.dumps(choices))
    out = nori.compare({"score": "p1", "recalibration": choices["recalibration"]}, cal, test)
    for base in ("xgboost", "nori_10k"):
        pair = bootstrap(y, {base: test[f"p_{base}"].to_numpy(), "nori": test["p1"].to_numpy()},
                         person, n_boot=100, seed=config.SEED)
        out["discrimination"][f"nori_minus_{base}_auroc"] = pair[f"nori_minus_{base}_auroc"]
        out["discrimination"][base] = pair[base]
    out |= {"windows": {"test": [9868, 9869, 10000]}, "test_seconds": 1234.5}
    (tmp_path / "benchmark_nori_full.json").write_text(json.dumps(out))
    block = report.render_nori_full()
    assert "passed: calibration and test were scored" in block
    assert "windows of 9,868, 9,869, 10,000 rows; the new windows took 1,234 s" in block
    assert "| Nori, every train row (M2d) |" in block and "- M2d − XGBoost:" in block
    assert "- M2d − Nori, 10,000-row context (M2c):" in block
