import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from clinical_ml import config, report
from clinical_ml.models import curve, nori

RNG = np.random.default_rng(11)


def test_sizes_are_nested_and_10k_is_the_m2c_context(monkeypatch):
    monkeypatch.setattr(nori, "CONTEXT_ROWS", 50)
    pool = pd.DataFrame({"visit_occurrence_id": RNG.permutation(10_000)[:200]},
                        index=np.arange(0, 400, 2))
    order = curve.row_order(pool)
    pd.testing.assert_frame_equal(curve.rows(order, 50), nori.context_sample(pool))
    pd.testing.assert_frame_equal(curve.rows(order, None), pool)
    small, ctx, big = (set(curve.rows(order, n).index) for n in (20, 50, 120))
    assert small < ctx < big and len(big) == 120
    pd.testing.assert_frame_equal(curve.rows(curve.row_order(pool), 20), curve.rows(order, 20))
    assert list(curve.rows(order, 120).index) == sorted(big)      # original order kept


def test_auroc_matches_scikit_learn_with_ties():
    y = RNG.integers(0, 2, 500)
    p = np.round(RNG.random(500) + 0.3 * y, 1)                   # many ties
    assert curve.auroc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def _wide(n=600):
    y = (RNG.random(n) < 0.3).astype(int)
    wide = pd.DataFrame({"visit_occurrence_id": np.arange(n),
                         "person_id": RNG.integers(0, 400, n), "readmit_30d": y,
                         "fold": RNG.integers(0, 5, n)})
    sizes = [curve.size_label(s) for s in curve.TREE_SIZES]
    for m in ("nori", *curve.TREES):
        for s in sizes:
            if m == "nori" and s not in map(str, curve.NORI_SIZES):
                continue
            signal = 2.0 if m == "nori" else 0.5                  # Nori clearly ahead
            wide[f"{m}@{s}"] = signal * y + RNG.normal(0, 1, n)
    return wide


def test_summarise_pairs_nori_with_each_model_at_the_same_size():
    wide = _wide()
    res = curve.summarise(wide, n_boot=30)
    for s in curve.NORI_SIZES:
        for m in curve.TREES:
            d = res[f"nori@{s} - {m}@{s}"]
            assert d["value"] == pytest.approx(res[f"nori@{s}"]["value"]
                                               - res[f"{m}@{s}"]["value"])
            assert d["ci95"][0] <= d["value"] <= d["ci95"][1]
            assert d["folds_nori_ahead"] == 5 and len(d["folds"]) == 5
    assert "nori@all" not in res and "xgboost@all" in res
    assert not any(k.startswith("nori@20000 -") for k in res)


def test_curve_block_and_figure_render_from_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    wide = _wide()
    sizes = [curve.size_label(s) for s in curve.TREE_SIZES]
    rows = {s: (47_472.8 if s == "all" else float(s)) for s in sizes}
    env = {"platform": "Linux-7.0", "sklearn_version": "1.9.1", "cpu_count": 12}
    out = {"nori_device": "cuda:0",
           "design": {"nori_sizes": list(curve.NORI_SIZES), "tree_sizes": sizes,
                      "n_boot": 30, "n_patients": 400, "rows": len(wide)},
           "mean_rows": rows, "results": curve.summarise(wide, n_boot=30),
           "runs": {"nori": {"versions": {"synthefy_nori": "0.23.0", "torch": "2.13.0"},
                             "seconds": {"0@1000": 40.0, "0@10000": 450.0}},
                    **{m: {"environment": env | {"xgboost_version": "3.4.1",
                                                 "device": "cuda"}} for m in curve.TREES}}}
    (tmp_path / "learning_curve.json").write_text(json.dumps(out))
    block = report.render_curve()
    assert block.startswith(report.CURVE_START) and block.endswith(report.CURVE_END)
    assert "took 40 s on `cuda:0`" in block
    assert "| all (~47,473) | — |" in block and "| 20,000 | — |" in block
    assert block.count("; 5/5") == 3 * len(curve.NORI_SIZES)
    curve.figure(out, tmp_path / "figures" / "learning_curve.png")
    assert (tmp_path / "figures" / "learning_curve.png").stat().st_size > 10_000
