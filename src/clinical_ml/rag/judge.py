"""LLM-as-judge (phi-4, a different model family from the generator) and the 2a metrics.

The judge grades two things, each with a fixed JSON schema and greedy decoding:
* correctness of an answer against the Asclepius reference answer: correct / partial /
  incorrect;
* faithfulness: the answer is split into atomic claims, and each is checked against the
  excerpts the answer *cited*. An answer that cites nothing has no support.

Everything else is computed, not judged: abstention, whether the context contained the
question's own note, and citation precision (share of cited excerpts that come from the
question's own note).
"""

import json
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from clinical_ml import config
from clinical_ml.eval.metrics import bootstrap_fn
from clinical_ml.rag import data
from clinical_ml.rag.generate import format_excerpts
from clinical_ml.rag.models import MODELS

N_BOOT = 1000

CORRECTNESS = """You grade a candidate answer to a question about one patient's discharge \
summary. A reference answer, written from that summary, is given.
- "correct": the candidate conveys the reference's key facts and contradicts nothing in it.
- "partial": the candidate conveys some of the key facts and contradicts nothing.
- "incorrect": the candidate misses the key facts, is about a different patient, or \
contradicts the reference.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}"""

FAITHFULNESS = """Split the answer below into short, atomic factual claims. For each claim, \
decide whether the excerpts state or directly imply it. Use only the excerpts, not outside \
knowledge.

Excerpts:
{excerpts}

Answer: {answer}"""


class Grade(BaseModel):
    label: Literal["correct", "partial", "incorrect"]


class Claim(BaseModel):
    claim: str
    supported: bool


class Claims(BaseModel):
    claims: list[Claim]


def _llm(schema):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    m = MODELS["judge"]
    llm = ChatOpenAI(model="judge", base_url=m.base_url, api_key="EMPTY", temperature=0,
                     max_tokens=2048, timeout=300)
    template = CORRECTNESS if schema is Grade else FAITHFULNESS
    return (ChatPromptTemplate.from_messages([("human", template)])
            | llm.with_structured_output(schema, method="json_schema"))


def valid_citations(cites: list[int], n_context: int) -> list[int]:
    return [c for c in dict.fromkeys(cites) if 1 <= c <= n_context]


def run_judge(split: str) -> pd.DataFrame:
    ans = pd.read_parquet(data.OUT_DIR / f"answers_{split}.parquet")
    chosen = json.loads((config.REPORTS / "rag_prompt_selection.json").read_text())["selected"]
    ans = ans[ans["prompt"] == chosen].reset_index(drop=True)
    qs = pd.read_parquet(data.OUT_DIR / "questions.parquet").set_index("qid")
    chunks = pd.read_parquet(data.OUT_DIR / "chunks.parquet")
    texts, chunk_note = chunks["text"].tolist(), chunks["note_id"].to_numpy()

    cited = [[int(rows[c - 1]) for c in valid_citations(list(cites), len(rows))]
             for cites, rows in zip(ans["citations"], ans["chunk_rows"], strict=True)]
    ans["n_cited"] = [len(c) for c in cited]
    ans["n_invalid_citations"] = [len(list(c)) - len(valid_citations(list(c), len(r)))
                                  for c, r in zip(ans["citations"], ans["chunk_rows"],
                                                  strict=True)]
    ans["cited_gold_share"] = [
        float(np.mean(chunk_note[c] == qs.loc[q, "note_id"])) if c else np.nan
        for q, c in zip(ans["qid"], cited, strict=True)]

    answered = ~ans["abstain"] & (ans["answer"].str.strip() != "")
    todo = ans[answered]
    grades = _llm(Grade).batch(
        [{"question": qs.loc[q, "question"], "reference": qs.loc[q, "answer"], "answer": a}
         for q, a in zip(todo["qid"], todo["answer"], strict=True)],
        config={"max_concurrency": 64}, return_exceptions=True)
    claims = _llm(Claims).batch(
        [{"excerpts": format_excerpts([texts[i] for i in c]) if c else "(none cited)",
          "answer": a} for c, a in zip([cited[i] for i in np.flatnonzero(answered)],
                                       todo["answer"], strict=True)],
        config={"max_concurrency": 64}, return_exceptions=True)
    # A judge call that fails (e.g. a runaway claim list) is recorded, not guessed: the
    # grade becomes "judge_error" and is left out of correctness; failed claim lists add
    # no claims to faithfulness.
    ans["grade"] = None
    ans.loc[answered, "grade"] = [g.label if isinstance(g, Grade) else "judge_error"
                                  for g in grades]
    ok = [isinstance(c, Claims) for c in claims]
    ans["n_claims"], ans["n_supported"], ans["claims_error"] = 0, 0, False
    ans.loc[answered, "n_claims"] = [len(c.claims) if k else 0 for c, k in
                                     zip(claims, ok, strict=True)]
    ans.loc[answered, "n_supported"] = [sum(x.supported for x in c.claims) if k else 0
                                        for c, k in zip(claims, ok, strict=True)]
    ans.loc[answered, "claims_error"] = [not k for k in ok]
    ans.to_parquet(data.OUT_DIR / f"judged_{split}.parquet", index=False)
    return ans


def _rate(mask) -> float:
    return float(np.mean(mask)) if len(mask) else np.nan


def metrics(df: pd.DataFrame) -> dict:
    """Per-question metrics for one condition; NaN where a metric has no denominator."""
    graded = df["grade"].to_numpy() != "judge_error"
    df = df[graded]
    answered = ~df["abstain"].to_numpy()
    grade = df["grade"].to_numpy()
    claims, supported = df["n_claims"].to_numpy(float), df["n_supported"].to_numpy(float)
    out = {
        "abstention_rate": _rate(~answered),
        "correct": _rate(grade == "correct"),                # abstentions count as not correct
        "correct_or_partial": _rate(np.isin(grade, ["correct", "partial"])),
        "correct_when_answered": _rate(grade[answered] == "correct"),
        "faithfulness": float(supported[answered].sum() / claims[answered].sum())
        if claims[answered].sum() else np.nan,
        "citation_precision": float(np.nanmean(df["cited_gold_share"]))
        if df["cited_gold_share"].notna().any() else np.nan,
        "answers_without_valid_citation": _rate((df["n_cited"].to_numpy() == 0)[answered]),
    }
    return out


def summarize(split: str) -> dict:
    df = pd.read_parquet(data.OUT_DIR / f"judged_{split}.parquet")
    qs = pd.read_parquet(data.OUT_DIR / "questions.parquet").set_index("qid")
    df["in_index"] = qs.loc[df["qid"], "in_index"].to_numpy()

    def boot(sub: pd.DataFrame) -> dict:
        cols = {c: sub[c].to_numpy() for c in ("abstain", "grade", "n_claims", "n_supported",
                                                 "cited_gold_share", "n_cited")}
        return bootstrap_fn(lambda **a: metrics(pd.DataFrame(a)), cols, sub["qid"].to_numpy(),
                            N_BOOT, config.SEED) | {"n": int(len(sub))}

    ret = df[df["condition"] == "retrieval"]
    # The same in-index questions under both conditions, so the gap is not confounded by
    # which questions retrieval happened to find.
    ok = df["grade"].isin(["correct", "partial"])
    r_ok = ok[(df["condition"] == "retrieval") & df["in_index"]].set_axis(
        ret.loc[ret["in_index"], "qid"])
    o_ok = ok[df["condition"] == "oracle"].set_axis(df.loc[df["condition"] == "oracle", "qid"])
    hit = ret.loc[ret["in_index"]].set_index("qid")["gold_in_context"]
    matched = {k: {"n": int(m.sum()), "retrieval": float(r_ok[m.index[m]].mean()),
                   "oracle": float(o_ok[m.index[m]].mean())}
               for k, m in (("gold_in_context", hit), ("gold_not_in_context", ~hit))}
    gap = float(o_ok[hit.index].mean() - r_ok.mean())
    miss = hit.index[~hit]
    matched["gap_share_from_missed_notes"] = float(
        (o_ok[miss].sum() - r_ok[miss].sum()) / len(hit) / gap) if gap else None
    out = {
        "retriever": str(df["retriever"].iloc[0]), "prompt": str(df["prompt"].iloc[0]),
        "answerable": {c: boot(df[(df["condition"] == c) & df["in_index"]])
                       for c in ("retrieval", "oracle")},
        "matched_correct_or_partial": matched,
        "retrieval_by_context": {
            "gold_in_context": boot(ret[ret["in_index"] & ret["gold_in_context"]]),
            "gold_not_in_context": boot(ret[ret["in_index"] & ~ret["gold_in_context"]]),
            "held_out_note": boot(ret[~ret["in_index"]]),
        },
        "invalid_citation_share": float(df["n_invalid_citations"].sum()
                                        / max(1, df["citations"].map(len).sum())),
        "generation_errors": {c: int(df.loc[df["condition"] == c, "error"].notna().sum())
                              for c in ("retrieval", "oracle")},
        "judge_errors": {"grade": int((df["grade"] == "judge_error").sum()),
                         "claims": int(df["claims_error"].sum())},
    }
    (config.REPORTS / f"rag_answers_{split}.json").write_text(json.dumps(out, indent=2))
    return out
