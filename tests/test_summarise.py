import numpy as np
import pandas as pd
import pytest

from clinical_ml import config
from clinical_ml.summarise import data, evaluate, pairs


def _asclepius(n=60):
    tasks = ["Summarization", "Question Answering"]
    df = pd.DataFrame({"note": [f"note {i}" for i in range(n)],
                       "question": [f"summarise {i}" for i in range(n)],
                       "answer": ["unanswerable" if i == 4 else f"ref {i}" for i in range(n)],
                       "task": [tasks[i % 2] for i in range(n)]})
    df["note_id"] = [f"n{i}" for i in range(n)]
    return df


def test_split_is_by_note_excludes_2a_and_is_seeded(monkeypatch):
    monkeypatch.setattr(data, "SPLITS", {"sft": 10, "dpo": 6, "dev": 2, "test": 4})
    df = _asclepius()
    out = data.split(df, exclude={"n0", "n2"}, seed=1)
    assert len(out) == 22 and out["note_id"].is_unique
    assert not {"n0", "n2", "n4"} & set(out["note_id"])         # 2a notes, unanswerable
    assert set(out["note_id"]) <= set(df.loc[df.task == "Summarization", "note_id"])
    assert out["split"].value_counts().to_dict() == {"sft": 10, "dpo": 6, "test": 4, "dev": 2}
    pd.testing.assert_frame_equal(out, data.split(df, exclude={"n0", "n2"}, seed=1))


def _group(faith, words):
    return pd.DataFrame({"faith": faith, "words": words})


def test_pair_choice_is_clean_vs_flawed_within_the_length_rule():
    g = _group([1.0, 0.9, 0.2, 1.0], [100, 110, 300, 100])
    # 0 vs 2 is the largest gap but 300 words is > 1.25 x 100; 0 vs 1 qualifies
    assert pairs.choose(g) == (0, 1)


def test_pair_choice_needs_one_clean_and_one_flawed_summary():
    assert pairs.choose(_group([0.9, 0.8, 0.5], [100, 100, 100])) is None   # none clean
    assert pairs.choose(_group([1.0, 1.0, 1.0], [100, 100, 100])) is None   # none flawed


def test_original_gap_rule_is_still_available():
    g = _group([1.0, 0.9, 0.6], [100, 100, 100])
    assert pairs.choose(g, min_gap=0.25) == (0, 2)
    assert pairs.choose(_group([1.0, 0.9], [100, 100]), min_gap=0.25) is None


def test_pair_ties_go_to_the_closest_lengths():
    g = _group([1.0, 0.5, 1.0], [100, 120, 112])
    assert pairs.choose(g) == (2, 1)


def test_rouge_l_bounds():
    r = evaluate._rouge_l(["the patient improved", "nothing alike"], ["the patient improved"] * 2)
    assert r[0] == pytest.approx(1.0) and r[1] == 0.0


def test_metrics_report_pairs_models_by_note(tmp_path, monkeypatch):
    for mod in (data, pairs, evaluate):
        monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate, "N_BOOT", 50)
    notes = [f"n{i}" for i in range(6)]
    items = pd.DataFrame({"split": ["dev"] * 2 + ["test"] * 4, "note_id": notes,
                          "note": "x", "question": "q", "answer": "the patient improved"})
    items.to_parquet(tmp_path / "items.parquet", index=False)
    rows, judged = [], []
    for n, s in zip(notes, items["split"], strict=True):
        for m, faith in (("base", 0.5), ("sft", 0.75), ("dpo", 1.0)):
            rows.append({"note_id": n, "split": s, "model": m, "text": "the patient improved",
                         "finish_reason": "stop"})
            judged.append({"note_id": n, "split": s, "model": m, "n_claims": 4,
                           "n_supported": int(4 * faith), "judge_error": False,
                           "grade": "partial"})
    pd.DataFrame(rows).to_parquet(tmp_path / "eval_outputs.parquet", index=False)
    for role in evaluate.JUDGES:
        pd.DataFrame(judged).to_parquet(tmp_path / f"eval_judged_{role}.parquet", index=False)
    evaluate.metrics_report()
    import json

    res = json.loads((tmp_path / "summ_eval.json").read_text())["splits"]["test"]["results"]
    assert res["eval_judge_faith|dpo-sft"]["value"] == pytest.approx(0.25)
    assert res["eval_judge_faith|dpo-base"]["value"] == pytest.approx(0.5)
    assert res["eval_judge_unsupported|base"]["value"] == pytest.approx(2.0)
    assert res["rouge_l|sft"]["value"] == pytest.approx(1.0)
    assert np.isclose(res["eval_judge_complete|dpo"]["value"], 1.0)
