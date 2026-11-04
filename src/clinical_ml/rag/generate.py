"""Grounded answers with citations, through LangChain against the vLLM generator.

Prompt choice (dev only, rule fixed before comparing): two system prompts are run on dev,
and the one with the higher balanced answer/abstain accuracy in the retrieval condition
is used on test. Balanced accuracy is the mean of (answers when the question's own note
is among the excerpts) and (abstains when it is not). It needs no judge.

Two conditions per question (the prompt is the same):
* retrieval: the top K_CONTEXT chunks from the retriever chosen on dev;
* oracle: every chunk of the question's own note (in-index questions only). This is the
  upper bound that separates retrieval misses from generation errors.

The model must answer in a fixed JSON schema (vLLM enforces it): abstain or not, the
answer, and the numbers of the excerpts that support it. Decoding is greedy.
"""

import json

import pandas as pd
from pydantic import BaseModel, Field

from clinical_ml import config
from clinical_ml.rag import data
from clinical_ml.rag.models import MODELS
from clinical_ml.rag.pipeline import K_CONTEXT

SYSTEM = """You answer questions about hospital patients using only the numbered excerpts \
from discharge summaries that you are given.
- The excerpts may come from several different patients. Use only excerpts about the \
patient the question describes.
- If the excerpts do not contain the answer, set "abstain" to true and leave "answer" empty.
- Otherwise answer concisely, and list in "citations" the numbers of every excerpt that \
supports your answer. Do not state anything the cited excerpts do not say."""

STRICT = """You answer questions about hospital patients using only the numbered excerpts \
from discharge summaries that you are given.
- The excerpts come from many different patients, and most are probably about other \
patients. Before using an excerpt, check that it describes the patient the question is \
about: the specific details in the question (conditions, procedures, medications, \
events, timing) must match the excerpt.
- If no excerpt clearly matches those details, or the matching excerpts do not contain \
the answer, set "abstain" to true and leave "answer" empty.
- Otherwise answer concisely, and list in "citations" the numbers of every excerpt that \
supports your answer. Do not state anything the cited excerpts do not say."""
PROMPTS = {"base": SYSTEM, "strict": STRICT}

USER = """Excerpts:
{excerpts}

Question: {question}"""


MAX_TOKENS = 1024


class Answer(BaseModel):
    abstain: bool = Field(description="true if the excerpts do not contain the answer")
    answer: str = Field(description="the answer, or empty when abstaining")
    citations: list[int] = Field(description="numbers of the excerpts that support the answer")


def format_excerpts(texts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))


def chain(prompt_name: str):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    m = MODELS["generate"]
    llm = ChatOpenAI(model="generate", base_url=m.base_url, api_key="EMPTY", temperature=0,
                     max_tokens=MAX_TOKENS, timeout=300,
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    prompt = ChatPromptTemplate.from_messages([("system", PROMPTS[prompt_name]),
                                               ("human", USER)])
    return prompt | llm.with_structured_output(Answer, method="json_schema")


def contexts(qs: pd.DataFrame, chunks: pd.DataFrame, rankings: pd.DataFrame,
             retriever: str) -> pd.DataFrame:
    """One row per (question, condition) with the chunk rows shown to the model."""
    by_note = chunks.groupby("note_id").indices
    ranked = rankings[rankings["config"] == retriever].set_index("qid")["ranked"]
    rows = []
    for q in qs.itertuples():
        rows.append({"qid": q.qid, "condition": "retrieval",
                     "chunk_rows": [int(i) for i in ranked[q.qid][:K_CONTEXT]]})
        if q.in_index:
            rows.append({"qid": q.qid, "condition": "oracle",
                         "chunk_rows": sorted(int(i) for i in by_note[q.note_id])})
    return pd.DataFrame(rows)


def decision_accuracy(df: pd.DataFrame) -> dict:
    """Answer/abstain decisions in the retrieval condition against whether the question's
    own note was among the excerpts (held-out notes never are)."""
    r = df[df["condition"] == "retrieval"]
    present, absent = r["gold_in_context"], ~r["gold_in_context"]
    answer_when_present = float((~r.loc[present, "abstain"]).mean())
    abstain_when_absent = float(r.loc[absent, "abstain"].mean())
    return {"answer_when_note_present": answer_when_present,
            "abstain_when_note_absent": abstain_when_absent,
            "balanced": (answer_when_present + abstain_when_absent) / 2,
            "n_present": int(present.sum()), "n_absent": int(absent.sum())}


def _generate(ctx: pd.DataFrame, texts: list[str], prompt_name: str) -> pd.DataFrame:
    ctx = ctx.copy()
    inputs = [{"excerpts": format_excerpts([texts[i] for i in rows]), "question": question}
              for rows, question in zip(ctx["chunk_rows"], ctx["question"], strict=True)]
    outputs = chain(prompt_name).batch(inputs, config={"max_concurrency": 64},
                                       return_exceptions=True)
    # A failed generation (e.g. hitting MAX_TOKENS) is kept as an empty, non-abstained
    # answer with its error, so it counts as not correct rather than vanishing.
    ok = [not isinstance(o, Exception) for o in outputs]
    ctx["prompt"] = prompt_name
    ctx["error"] = [None if k else f"{type(o).__name__}: {str(o)[:200]}"
                    for o, k in zip(outputs, ok, strict=True)]
    ctx["abstain"] = [o.abstain if k else False for o, k in zip(outputs, ok, strict=True)]
    ctx["answer"] = [o.answer if k else "" for o, k in zip(outputs, ok, strict=True)]
    # citation numbers outside 1..len(context) are kept as invalid, not dropped silently
    ctx["citations"] = [list(o.citations) if k else [] for o, k in zip(outputs, ok, strict=True)]
    return ctx


def run(split: str) -> pd.DataFrame:
    qs = pd.read_parquet(data.OUT_DIR / "questions.parquet")
    qs = qs[qs["split"] == split]
    chunks = pd.read_parquet(data.OUT_DIR / "chunks.parquet")
    rankings = pd.read_parquet(data.OUT_DIR / "rankings.parquet")
    retriever = json.loads((config.REPORTS / "rag_retrieval.json").read_text())[
        "selected_for_generation"]
    ctx = contexts(qs, chunks, rankings, retriever).merge(qs[["qid", "question", "note_id"]],
                                                          on="qid")
    note = chunks["note_id"].to_numpy()
    ctx["gold_in_context"] = [bool((note[rows] == gold).any())
                              for rows, gold in zip(ctx["chunk_rows"], ctx["note_id"],
                                                    strict=True)]
    texts = chunks["text"].tolist()
    selection_path = config.REPORTS / "rag_prompt_selection.json"
    if split == "dev":
        out = pd.concat([_generate(ctx, texts, name) for name in PROMPTS], ignore_index=True)
        scores = {name: decision_accuracy(out[out["prompt"] == name]) for name in PROMPTS}
        chosen = max(PROMPTS, key=lambda n: (scores[n]["balanced"], n == "base"))
        selection_path.write_text(json.dumps(
            {"rule": "higher dev balanced answer/abstain accuracy (retrieval condition)",
             "dev": scores, "selected": chosen}, indent=2))
    else:
        chosen = json.loads(selection_path.read_text())["selected"]
        out = _generate(ctx, texts, chosen)
    out["retriever"] = retriever
    out.drop(columns=["question", "note_id"]).to_parquet(
        data.OUT_DIR / f"answers_{split}.parquet", index=False)
    return out
