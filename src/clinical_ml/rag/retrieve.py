"""Retrievers for the known-item benchmark.

* BM25: bm25s with English stop words and Snowball stemming.
* Dense: bge-base embeddings from the vLLM embedding server, called through LangChain's
  OpenAIEmbeddings and stored in pgvector (HNSW, cosine). Chunks over the model's
  512-token limit are truncated for this index only (BM25 sees the full text).
  Postgres is the embedded `pgserver` build, so it lives inside the project folder.
* Hybrid: reciprocal-rank fusion (Cormack et al., 2009) of the BM25 and dense lists.
* Rerank: the bge cross-encoder rescores the top RERANK_POOL fused chunks (vLLM /rerank).

Chunks are referred to by their row number in chunks.parquet.
"""

import httpx
import numpy as np

from clinical_ml import config
from clinical_ml.rag.models import BGE_QUERY_PREFIX, MODELS, use_project_hf_cache

POOL = 100          # candidates taken from each retriever before fusion
RRF_K = 60          # the constant from Cormack et al.
RERANK_POOL = 50
EF_SEARCH = 200
PG_DIR = config.DATA / "rag" / "pg"


class BM25:
    def __init__(self, texts: list[str]):
        import bm25s
        import Stemmer

        self._bm25s, self._stem = bm25s, Stemmer.Stemmer("english")
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize(texts, stopwords="en", stemmer=self._stem,
                                        show_progress=False), show_progress=False)

    def search(self, queries: list[str], k: int = POOL) -> np.ndarray:
        q = self._bm25s.tokenize(queries, stopwords="en", stemmer=self._stem,
                                 show_progress=False)
        ids, _ = self.index.retrieve(q, k=k, show_progress=False, n_threads=-1)
        return np.asarray(ids)


MAX_TOKENS = 512    # bge-base input limit; longer chunks are truncated for the dense index


def embedder():
    from langchain_openai import OpenAIEmbeddings

    m = MODELS["embed"]
    return OpenAIEmbeddings(model="embed", base_url=m.base_url, api_key="EMPTY",
                            check_embedding_ctx_length=False, chunk_size=256,
                            model_kwargs={"extra_body": {"truncate_prompt_tokens": MAX_TOKENS}})


def token_counts(texts: list[str]) -> np.ndarray:
    """bge token counts (with [CLS]/[SEP]), to report how many chunks get truncated."""
    from tokenizers import Tokenizer

    use_project_hf_cache()
    m = MODELS["embed"]
    tok = Tokenizer.from_pretrained(m.name, revision=m.revision)
    return np.array([len(e.ids) for e in tok.encode_batch(texts)])


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    texts = [BGE_QUERY_PREFIX + t for t in texts] if query else texts
    v = np.asarray(embedder().embed_documents(texts), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pg_server():
    import pgserver

    PG_DIR.mkdir(parents=True, exist_ok=True)
    srv = pgserver.get_server(str(PG_DIR), cleanup_mode="stop")
    srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    return srv


def _connect(uri: str):
    import psycopg
    from pgvector.psycopg import register_vector

    con = psycopg.connect(uri, autocommit=True)
    register_vector(con)
    return con


def build_pgvector(uri: str, vectors: np.ndarray) -> dict:
    """Load all chunk vectors and build the HNSW index. Returns the pgvector version."""
    with _connect(uri) as con:
        con.execute("DROP TABLE IF EXISTS chunk_vec")
        dim = vectors.shape[1]
        con.execute(f"CREATE TABLE chunk_vec (i integer PRIMARY KEY, v vector({dim}))")
        copy_sql = "COPY chunk_vec (i, v) FROM STDIN WITH (FORMAT BINARY)"
        with con.cursor() as cur, cur.copy(copy_sql) as copy:
            copy.set_types(["int4", "vector"])
            for i, v in enumerate(vectors):
                copy.write_row((i, v))
        con.execute("SET maintenance_work_mem = '16GB'")
        con.execute("SET max_parallel_maintenance_workers = 16")
        con.execute("CREATE INDEX ON chunk_vec USING hnsw (v vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 200)")
        version = con.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    return {"pgvector": version, "rows": int(len(vectors)),
            "index": "hnsw m=16 ef_construction=200"}


def search_pgvector(uri: str, queries: np.ndarray, k: int = POOL) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    with _connect(uri) as con:
        con.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        for j, q in enumerate(queries):
            rows = con.execute("SELECT i FROM chunk_vec ORDER BY v <=> %s LIMIT %s",
                               (q, k)).fetchall()
            out[j] = [r[0] for r in rows]
    return out


def search_exact(vectors: np.ndarray, queries: np.ndarray, k: int = POOL) -> np.ndarray:
    """Brute-force cosine top-k (vectors are unit length), to measure HNSW's recall."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), 256):
        sims = queries[s:s + 256] @ vectors.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.take_along_axis(sims, top, 1).argsort(axis=1)[:, ::-1]
        out[s:s + 256] = np.take_along_axis(top, order, 1)
    return out


def rrf(*rankings: np.ndarray, k: int = RRF_K, depth: int = POOL) -> list[list[int]]:
    """Reciprocal-rank fusion of several (n_queries, depth) rankings."""
    fused = []
    for rows in zip(*rankings, strict=True):
        score: dict[int, float] = {}
        for row in rows:
            for rank, doc in enumerate(row[:depth]):
                score[int(doc)] = score.get(int(doc), 0.0) + 1.0 / (k + rank + 1)
        fused.append(sorted(score, key=lambda d: (-score[d], d)))
    return fused


def rerank(queries: list[str], candidates: list[list[int]], texts: list[str],
           pool: int = RERANK_POOL) -> list[list[int]]:
    m = MODELS["rerank"]
    out = []
    with httpx.Client(timeout=120) as client:
        for q, cand in zip(queries, candidates, strict=True):
            cand = cand[:pool]
            r = client.post(f"{m.base_url}/rerank", json={
                "model": "rerank", "query": q, "documents": [texts[i] for i in cand]})
            r.raise_for_status()
            order = sorted(r.json()["results"], key=lambda x: -x["relevance_score"])
            out.append([cand[x["index"]] for x in order])
    return out
