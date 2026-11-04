"""Phase 2a steps, run on the GPU pod (the vLLM servers for each step must be up):

  data      download Asclepius, chunk notes, sample the dev/test questions
  index     embed every chunk, load pgvector + HNSW, keep the vectors for exact search
  retrieve  rank chunks for every question with each retriever; score retrieval

Retrieval protocol (fixed before running):
* A hit@k is a question whose own note supplies at least one of the top-k chunks;
  held-out questions (note removed from the index) have no hit to find and are scored
  separately, later, for abstention.
* The generator reads the top K_CONTEXT chunks, so the retriever used for generation is
  the one with the highest dev hit@K_CONTEXT (ties go to the simpler one). Test results
  are reported for every retriever, but only that one feeds generation.
"""

import json

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.eval.metrics import bootstrap_fn
from clinical_ml.rag import data, retrieve

K_CONTEXT = 8
HITS_AT = (1, 5, K_CONTEXT, 20)
CONFIGS = ("bm25", "dense", "hybrid", "hybrid_rerank")      # simplest first
N_BOOT = 1000
DEPTH = 50                                                  # ranks kept per question


def _chunks() -> pd.DataFrame:
    return pd.read_parquet(data.OUT_DIR / "chunks.parquet")


def _questions() -> pd.DataFrame:
    return pd.read_parquet(data.OUT_DIR / "questions.parquet")


def index() -> dict:
    chunks = _chunks()
    vectors = retrieve.embed(chunks["text"].tolist())
    np.save(data.OUT_DIR / "chunk_vectors.npy", vectors)
    srv = retrieve.pg_server()
    info = retrieve.build_pgvector(srv.get_uri(), vectors)
    n_tok = retrieve.token_counts(chunks["text"].tolist())
    info |= {"embedding_dim": int(vectors.shape[1]), "chunks": int(len(chunks)),
             "chunks_truncated_for_dense": int((n_tok > retrieve.MAX_TOKENS).sum()),
             "chunk_tokens": {"median": float(np.median(n_tok)), "max": int(n_tok.max())}}
    (config.REPORTS / "rag_index.json").write_text(json.dumps(info, indent=2))
    return info


def first_hit(ranked: list[int], gold: str, chunk_note: np.ndarray) -> int | None:
    """1-based rank of the first chunk from the gold note, or None."""
    for r, i in enumerate(ranked, 1):
        if chunk_note[i] == gold:
            return r
    return None


def hit_metrics(ranks: np.ndarray) -> dict:
    """ranks: first-hit rank per question, np.inf when not found within DEPTH."""
    return {f"hit@{k}": float(np.mean(ranks <= k)) for k in HITS_AT} | {
        "mrr": float(np.mean(np.where(np.isfinite(ranks), 1 / ranks, 0.0)))}


def retrieve_all() -> dict:
    chunks, qs = _chunks(), _questions()
    texts, chunk_note = chunks["text"].tolist(), chunks["note_id"].to_numpy()
    queries = qs["question"].tolist()

    bm25 = retrieve.BM25(texts).search(queries)
    qvec = retrieve.embed(queries, query=True)
    srv = retrieve.pg_server()
    dense = retrieve.search_pgvector(srv.get_uri(), qvec)
    exact = retrieve.search_exact(np.load(data.OUT_DIR / "chunk_vectors.npy"), qvec)
    hybrid = retrieve.rrf(bm25, dense)
    ranked = {"bm25": [list(r) for r in bm25], "dense": [list(r) for r in dense],
              "hybrid": hybrid, "hybrid_rerank": retrieve.rerank(queries, hybrid, texts)}

    rows = []
    for name, lists in ranked.items():
        for qid, gold, lst in zip(qs["qid"], qs["note_id"], lists, strict=True):
            lst = [int(i) for i in lst[:DEPTH]]
            r = first_hit(lst, gold, chunk_note)
            rows.append({"qid": qid, "config": name, "ranked": lst,
                         "first_hit": float(r) if r else np.inf})
    rank_df = pd.DataFrame(rows)
    rank_df.to_parquet(data.OUT_DIR / "rankings.parquet", index=False)

    ann = float(np.mean([len(set(a[:20]) & set(b[:20])) / 20 for a, b in
                         zip(dense, exact, strict=True)]))
    out = {"protocol": {"k_context": K_CONTEXT, "pool_per_retriever": retrieve.POOL,
                        "rrf_k": retrieve.RRF_K, "rerank_pool": retrieve.RERANK_POOL,
                        "hnsw_ef_search": retrieve.EF_SEARCH, "n_boot": N_BOOT},
           "hnsw_vs_exact_overlap@20": ann, "splits": {}}
    for split in ("dev", "test"):
        q = qs[(qs["split"] == split) & qs["in_index"]]
        res = {"n_questions": int(len(q)), "configs": {}}
        for name in CONFIGS:
            fh = rank_df[rank_df["config"] == name].set_index("qid").loc[q["qid"],
                                                                          "first_hit"].to_numpy()
            res["configs"][name] = {
                "overall": bootstrap_fn(lambda r: hit_metrics(r), {"r": fh},
                                        q["qid"].to_numpy(), N_BOOT, config.SEED),
                "by_task": {t: hit_metrics(fh[(q["task"] == t).to_numpy()])
                            for t in sorted(q["task"].unique())}}
        out["splits"][split] = res
    dev = out["splits"]["dev"]["configs"]
    best = max(CONFIGS, key=lambda c: (dev[c]["overall"][f"hit@{K_CONTEXT}"]["value"],
                                       -CONFIGS.index(c)))
    out["selected_for_generation"] = best
    out["selection_rule"] = f"highest dev hit@{K_CONTEXT}; ties to the simpler retriever"
    (config.REPORTS / "rag_retrieval.json").write_text(json.dumps(out, indent=2))
    return out
