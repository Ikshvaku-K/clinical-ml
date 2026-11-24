import json

import numpy as np
import pandas as pd
import pytest

from clinical_ml import report
from clinical_ml.models import curve, nori
from clinical_ml.models import tfm_variants as tv

RNG = np.random.default_rng(23)


def _train(n=300):
    y = (RNG.random(n) < 0.3).astype(int)
    return pd.DataFrame({"visit_occurrence_id": np.arange(n), "person_id": np.arange(n) // 2,
                         "readmit_30d": y, "split": "train", "fold": (np.arange(n) // 2) % 5,
                         "x": y + RNG.normal(0, 1, n)})


def test_predict_stops_at_the_first_size_that_does_not_fit(tmp_path, monkeypatch):
    tr = _train()
    monkeypatch.setattr(tv, "OUT", tmp_path)
    monkeypatch.setattr(nori, "CONTEXT_ROWS", 50)
    monkeypatch.setattr(curve, "train_rows", lambda: tr)
    monkeypatch.setitem(tv.VARIANTS, "stub", {"family": "nori", "sizes": (20, 40, 80, None),
                                              "label": "stub"})
    seen = []

    def scorer(spec, ctx, query):
        seen.append(len(ctx))
        if len(ctx) == 40 and query["fold"].iloc[0] == 3:      # fits folds 0-2 only
            raise tv.NotFit("OutOfMemoryError: stub")
        return query["x"].to_numpy(), {"seconds": 1.0, "device": "cpu", "score": "p1"}

    log = tv.predict("stub", scorer)
    assert log["sizes_done"] == ["20"] and 80 not in seen
    assert log["calls"][-1] == {"size": "40", "status": "does not fit",
                                "error": "OutOfMemoryError: stub"}
    s = pd.read_parquet(tmp_path / "stub.parquet")
    assert len(s) == len(tr) and set(s["size"]) == {"20"}
    seen.clear()
    log = tv.predict("stub", scorer)                        # resumes: 20 is not rerun
    assert log["sizes_done"] == ["20"] and seen and 20 not in seen
    (tmp_path / "parts" / "stub@40.attempt").write_text("killed")   # as if the guard hit
    seen.clear()
    log = tv.predict("stub", scorer)
    assert not seen and log["calls"][-1]["error"].startswith("Killed")


def test_evaluate_pairs_each_variant_with_the_learning_curve(tmp_path, monkeypatch):
    tr = _train()
    monkeypatch.setattr(curve, "train_rows", lambda: tr)
    monkeypatch.setattr(curve, "OUT", tmp_path / "curve")
    monkeypatch.setattr(tv, "OUT", tmp_path / "variants")
    monkeypatch.setattr(tv, "REPORT", tmp_path / "tfm_variants.json")
    monkeypatch.setattr(tv.config, "REPORTS", tmp_path)
    (tmp_path / "curve").mkdir()
    (tmp_path / "variants").mkdir()
    base = tr[["visit_occurrence_id", "fold"]]
    y = tr["readmit_30d"].to_numpy()

    def scores(sizes, signal):
        return pd.concat([base.assign(size=s, p=signal * y + RNG.normal(0, 1, len(tr)))
                          for s in sizes], ignore_index=True)

    scores(["1000", "10000"], 0.5).to_parquet(tmp_path / "curve" / "nori.parquet")
    for m in curve.TREES:
        scores(["1000", "10000", "all"], 0.5).to_parquet(tmp_path / "curve" / f"{m}.parquet")
    for name, sizes in (("tabicl", ["1000", "all"]), ("cluster_route", ["all"])):
        scores(sizes, 3.0).to_parquet(tmp_path / "variants" / f"{name}.parquet")
        (tmp_path / "variants" / f"{name}.json").write_text(json.dumps(
            {"variant": name, "sizes_tried": sizes, "sizes_done": sizes,
             "versions": {"torch": "x"}, "calls": [{"seconds": 2.0, "device": "cuda:0"}]}))
    out = tv.evaluate(n_boot=30)
    r = out["results"]
    assert {"tabicl@1000 - nori@1000", "tabicl@1000 - xgboost@1000", "tabicl@all - xgboost@all",
            "cluster_route@all - nori@10000", "cluster_route@all - xgboost@all"} <= set(r)
    assert "tabicl@all - nori@all" not in r                  # Nori 1.5 has no such size
    assert r["tabicl@all - xgboost@all"]["folds_ahead"] == 5
    assert out["test_rule"]["test_candidate"] in ("tabicl@all", "tabicl@1000",
                                                  "cluster_route@all")
    assert out["variants"]["tabicl"]["seconds"] == 2.0

    block = report.render_variants()
    assert block.startswith(report.VARIANTS_START) and block.endswith(report.VARIANTS_END)
    assert "| TabICL v2 classifier | all" in block and "test candidate" in block


def test_variants_have_pinned_weights():
    for spec in tv.VARIANTS.values():
        if spec["family"] == "nori":
            assert len(spec["revision"]) == 40
    assert len(tv.TABICL["revision"]) == 40
    with pytest.raises(KeyError):
        tv.VARIANTS["nori-1-5"]


def test_candidate_steps_follow_the_test_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(tv, "OUT", tmp_path / "variants")
    monkeypatch.setattr(tv, "REPORT", tmp_path / "tfm_variants.json")
    monkeypatch.setattr(tv, "CANDIDATE", tmp_path / "choices.json")
    monkeypatch.setattr(nori, "OUT", tmp_path)
    monkeypatch.setattr(nori, "CONTEXT_ROWS", 50)
    tv.REPORT.write_text(json.dumps({"test_rule": {"test_candidate": None}}))
    with pytest.raises(SystemExit, match="unread"):
        tv.predict_split("calibration")
    tv.REPORT.write_text(json.dumps({"test_rule": {"test_candidate": "tabicl@all"}}))
    parts = []
    for split, n in (("train", 200), ("calibration", 300), ("test", 300)):
        d = _train(n).drop(columns="x")
        d["visit_occurrence_id"] += len(parts) * 1000
        d["person_id"] += len(parts) * 1000
        parts.append(d.assign(split=split, fold=d["fold"] if split == "train" else -1))
    pd.concat(parts, ignore_index=True).to_parquet(tmp_path / "features.parquet", index=False)

    def stub(spec, ctx, query):
        return np.clip(0.1 + 0.3 * query["readmit_30d"].to_numpy(), 0, 1), {
            "seconds": 1.0, "device": "cpu", "score": "p1"}

    with pytest.raises(SystemExit, match="recalibration"):
        tv.predict_split("test", stub)
    log = tv.predict_split("calibration", stub)
    assert log["context_rows"] == 200 and log["candidate"] == "tabicl@all"
    assert tv.choose_candidate()["candidate"] == "tabicl@all"
    tv.predict_split("test", stub)
    with pytest.raises(SystemExit, match="once"):
        tv.predict_split("test", stub)


def test_candidate_test_block_renders(tmp_path, monkeypatch):
    from clinical_ml import config
    from clinical_ml.eval.metrics import bootstrap

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    monkeypatch.setattr(nori, "N_BOOT", 100)
    monkeypatch.setattr(nori, "N_BOOT_CAL", 50)
    (tmp_path / "tfm_variants.json").write_text(json.dumps({
        "results": {}, "columns": [],
        "variants": {"tabicl": {"label": "TabICL v2 classifier", "not_fit": [],
                                "sizes_done": ["all"]}},
        "test_rule": {"best": "tabicl@all", "best_cv_auroc": 0.6655, "xgboost_all": 0.662,
                      "test_candidate": "tabicl@all"}}))
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
    out = nori.compare({"score": "p1", "recalibration": {"selected": "platt"}}, cal, test)
    for base in ("xgboost", "nori_10k"):
        pair = bootstrap(y, {base: test[f"p_{base}"].to_numpy(), "nori": test["p1"].to_numpy()},
                         person, n_boot=100, seed=0)
        out["discrimination"][f"nori_minus_{base}_auroc"] = pair[f"nori_minus_{base}_auroc"]
        out["discrimination"][base] = pair[base]
    out |= {"candidate": "tabicl@all", "runs": {"test": {"context_rows": 59341, "seconds": 41.2,
                                                          "device": "cuda"}}}
    (tmp_path / "benchmark_tfm_candidate.json").write_text(json.dumps(out))
    block = report.render_variants()
    assert "**tabicl@all** is the test candidate" in block
    assert "**Test, scored once: TabICL v2 classifier** with 59,341 train rows" in block
    assert "- TabICL v2 classifier − XGBoost:" in block and "| TabICL v2 classifier |" in block
