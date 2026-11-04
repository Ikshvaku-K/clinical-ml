import httpx
import numpy as np
import pandas as pd
import pytest

from clinical_ml.rag import data, pipeline, retrieve

NOTE = """Discharge Summary:

Patient: 60-year-old male with ARDS

Hospital Course:

The patient was admitted with fever and dyspnea. He improved with oxygen.

Follow-up:

Review in clinic in two weeks."""


def _words(t):
    return len(t.split())


def test_chunks_keep_headers_with_text_and_preserve_the_note():
    chunks = data.chunk_note(NOTE)
    assert " ".join(" ".join(chunks).split()) == " ".join(NOTE.split())
    assert all(not c.rstrip().endswith(":") for c in chunks)       # no header-only chunk
    assert "Hospital Course:\nThe patient was admitted" in "\n\n".join(chunks)


def test_long_paragraphs_and_lists_respect_the_word_cap(monkeypatch):
    monkeypatch.setattr(data, "MAX_WORDS", 20)
    sentences = " ".join(f"Sentence number {i} has five words." for i in range(12))
    listing = ", ".join(f"drug{i}" for i in range(70))               # no sentence ends
    note = f"Medications:\n\n{listing}\n\nCourse:\n\n{sentences}"
    chunks = data.chunk_note(note)
    assert max(_words(c) for c in chunks) <= 20
    assert " ".join(" ".join(chunks).split()) == " ".join(note.split())


def _frame(n=400):
    rng = np.random.default_rng(0)
    tasks = list(data.TASKS) + ["Paraphrasing", "Abbreviation Expansion"]
    df = pd.DataFrame({
        "patient_id": [str(i) for i in range(n)],
        "note": [f"Note {i}:\n\nBody text for patient {i}." for i in range(n)],
        "question": [f"q{i}" for i in range(n)],
        "answer": ["This is unanswerable." if i % 25 == 0 else f"a{i}" for i in range(n)],
        "task": rng.choice(tasks, n)})
    df["note_id"] = df["note"].map(data.note_id)
    return df


def test_question_sample_follows_the_protocol(monkeypatch):
    monkeypatch.setattr(data, "N_DEV", 20)
    monkeypatch.setattr(data, "N_TEST", 50)
    df = _frame()
    q = data.sample_questions(df, seed=1)
    assert len(q) == 70 and q["note_id"].is_unique
    assert set(q["task"]) <= set(data.TASKS)
    assert not q["answer"].str.contains("unanswerable").any()
    held = (~q["in_index"]).groupby(q["split"]).sum().to_dict()
    assert held == {"dev": 2, "test": 5}
    pd.testing.assert_frame_equal(q, data.sample_questions(df, seed=1))   # deterministic
    chunks = data.build_chunks(df, set(q.loc[~q["in_index"], "note_id"]))
    assert not set(q.loc[~q["in_index"], "note_id"]) & set(chunks["note_id"])
    assert set(q.loc[q["in_index"], "note_id"]) <= set(chunks["note_id"])


def test_rrf_rewards_agreement_between_retrievers():
    a = np.array([[1, 2, 3]])
    b = np.array([[3, 1, 4]])
    fused = retrieve.rrf(a, b, depth=3)[0]
    assert fused[0] == 1                      # ranks 1 and 2 beat 3's ranks 3 and 1
    assert fused[:2] == [1, 3] and set(fused) == {1, 2, 3, 4}


def test_hit_metrics_and_first_hit():
    chunk_note = np.array(["a", "b", "a", "c"])
    assert pipeline.first_hit([1, 3, 2], "a", chunk_note) == 3
    assert pipeline.first_hit([1, 3], "a", chunk_note) is None
    m = pipeline.hit_metrics(np.array([1.0, 3.0, np.inf, 9.0]))
    assert m["hit@1"] == 0.25 and m["hit@5"] == 0.5 and m[f"hit@{pipeline.K_CONTEXT}"] == 0.5
    assert m["hit@20"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 9) / 4)


def test_exact_search_matches_brute_force():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:7] + 0.01
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    got = retrieve.search_exact(v, q, k=10)
    want = np.argsort(-(q @ v.T), axis=1)[:, :10]
    assert (got == want).all() and (got[:, 0] == np.arange(7)).all()


def test_bm25_finds_the_matching_chunk():
    texts = ["warfarin was stopped after the bleed", "knee replacement recovery",
             "the patient received insulin for hyperglycemia", "routine follow-up"]
    ids = retrieve.BM25(texts).search(["why was warfarin stopped", "insulin dose"], k=2)
    assert ids[0][0] == 0 and ids[1][0] == 2


def test_pgvector_hnsw_returns_nearest_neighbours(tmp_path, monkeypatch):
    pytest.importorskip("pgserver")
    monkeypatch.setattr(retrieve, "PG_DIR", tmp_path / "pg")
    rng = np.random.default_rng(1)
    v = rng.normal(size=(300, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    srv = retrieve.pg_server()
    try:
        info = retrieve.build_pgvector(srv.get_uri(), v)
        assert info["rows"] == 300 and info["pgvector"]
        got = retrieve.search_pgvector(srv.get_uri(), v[:5], k=5)
        assert (got[:, 0] == np.arange(5)).all()
    finally:
        srv.cleanup()


def test_rerank_orders_by_relevance_score(monkeypatch):
    def handler(request):
        docs = request.read().decode()
        assert '"query"' in docs
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5}]})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    out = retrieve.rerank(["q"], [[10, 11, 12]], {10: "a", 11: "b", 12: "c"}, pool=3)
    assert out == [[11, 12, 10]]


def test_contexts_give_topk_for_retrieval_and_whole_note_for_oracle():
    from clinical_ml.rag import generate

    chunks = pd.DataFrame({"note_id": ["a", "a", "b", "c", "a"]})
    qs = pd.DataFrame({"qid": ["q1", "q2"], "note_id": ["a", "z"], "in_index": [True, False]})
    rankings = pd.DataFrame({"qid": ["q1", "q2", "q1"], "config": ["hybrid", "hybrid", "bm25"],
                             "ranked": [list(range(20)), list(range(20, 40)), [9]]})
    ctx = generate.contexts(qs, chunks, rankings, "hybrid")
    assert list(zip(ctx.qid, ctx.condition, strict=True)) == [
        ("q1", "retrieval"), ("q1", "oracle"), ("q2", "retrieval")]   # no oracle if held out
    assert ctx.chunk_rows[0] == list(range(pipeline.K_CONTEXT))
    assert ctx.chunk_rows[1] == [0, 1, 4]


def test_citations_outside_the_context_are_invalid():
    from clinical_ml.rag.judge import valid_citations

    assert valid_citations([2, 2, 0, 9, 1], 8) == [2, 1]


def test_answer_metrics_definitions():
    from clinical_ml.rag.judge import metrics

    df = pd.DataFrame({
        "abstain": [False, False, False, True],
        "grade": ["correct", "partial", "incorrect", None],
        "n_claims": [2, 4, 2, 0], "n_supported": [2, 2, 0, 0],
        "cited_gold_share": [1.0, 0.5, np.nan, np.nan], "n_cited": [1, 2, 0, 0]})
    m = metrics(df)
    assert m["abstention_rate"] == 0.25
    assert m["correct"] == 0.25 and m["correct_or_partial"] == 0.5
    assert m["correct_when_answered"] == pytest.approx(1 / 3)
    assert m["faithfulness"] == pytest.approx(4 / 8)
    assert m["citation_precision"] == pytest.approx(0.75)
    assert m["answers_without_valid_citation"] == pytest.approx(1 / 3)


def test_decision_accuracy_balances_answering_and_abstaining():
    from clinical_ml.rag.generate import decision_accuracy

    df = pd.DataFrame({"condition": ["retrieval"] * 5 + ["oracle"],
                       "gold_in_context": [True, True, False, False, False, True],
                       "abstain": [False, True, True, False, True, True]})
    d = decision_accuracy(df)                         # the oracle row is ignored
    assert d["answer_when_note_present"] == 0.5
    assert d["abstain_when_note_absent"] == pytest.approx(2 / 3)
    assert d["balanced"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_judge_failures_are_left_out_not_counted_wrong():
    from clinical_ml.rag.judge import metrics

    base = {"abstain": [False, False], "grade": ["correct", "judge_error"],
            "n_claims": [1, 0], "n_supported": [1, 0], "cited_gold_share": [1.0, 1.0],
            "n_cited": [1, 1]}
    assert metrics(pd.DataFrame(base))["correct"] == 1.0
