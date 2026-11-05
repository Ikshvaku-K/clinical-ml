"""On-policy DPO pairs.

1. sample: the SFT model (served by vLLM as the "sft" LoRA) writes N_SAMPLES summaries per
   DPO prompt (temperature 0.8, seeded per prompt).
2. score: phi-4 splits each summary into claims and checks each against the whole note.
3. build: per prompt, among samples that finished normally and made at least one claim,
   a *chosen* summary with every claim supported and a *rejected* one with at least one
   unsupported claim, lengths within LENGTH_RATIO of each other (so DPO cannot learn that
   shorter is safer). The pair with the largest faithfulness gap wins; ties go to the
   pair closest in length.

Deviation from the pre-set rule, decided before any evaluation (Kiran, 2026-11-05): the
plan required a gap >= 0.25. The SFT model's samples were 99.1% faithful, so that rule
left 30 pairs; "fully supported vs not" keeps the same intent and gives several hundred.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import pandas as pd

from clinical_ml import config
from clinical_ml.rag.generate import format_excerpts
from clinical_ml.rag.judge import Claims, _llm
from clinical_ml.rag.models import MODELS
from clinical_ml.summarise.data import OUT, messages

N_SAMPLES = 4
TEMPERATURE, TOP_P, MAX_TOKENS = 0.8, 0.95, 512
ORIGINAL_MIN_GAP = 0.25          # the pre-set rule; kept to report how many pairs it gave
LENGTH_RATIO = 1.25


def complete(model: str, items: pd.DataFrame, n: int, temperature: float,
             seed_offset: int = 0) -> pd.DataFrame:
    """Chat completions from the vLLM "summarise" server; one row per returned choice."""
    from openai import OpenAI

    client = OpenAI(base_url=MODELS["summarise"].base_url, api_key="EMPTY", timeout=600)

    def one(i_row):
        i, row = i_row
        r = client.chat.completions.create(
            model=model, messages=messages(row.note, row.question), n=n,
            temperature=temperature, top_p=TOP_P if temperature else 1.0,
            max_tokens=MAX_TOKENS, seed=config.SEED + seed_offset + i,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return [{"note_id": row.note_id, "sample": c.index, "text": c.message.content or "",
                 "finish_reason": c.finish_reason} for c in r.choices]

    with ThreadPoolExecutor(64) as pool:
        rows = [x for out in pool.map(one, enumerate(items.itertuples())) for x in out]
    return pd.DataFrame(rows)


def faithfulness(texts: list[str], notes: list[str], role: str) -> pd.DataFrame:
    """Claims in each text checked against its note by the judge served for `role`."""
    out = _llm(Claims, role).batch(
        [{"excerpts": format_excerpts([n]), "answer": t} for t, n in zip(texts, notes,
                                                                          strict=True)],
        config={"max_concurrency": 64}, return_exceptions=True)
    ok = [isinstance(c, Claims) for c in out]
    n_claims = [len(c.claims) if k else 0 for c, k in zip(out, ok, strict=True)]
    n_sup = [sum(x.supported for x in c.claims) if k else 0 for c, k in zip(out, ok,
                                                                            strict=True)]
    return pd.DataFrame({"n_claims": n_claims, "n_supported": n_sup, "judge_error": [not k for
                                                                                     k in ok]})


def _items(split: str) -> pd.DataFrame:
    items = pd.read_parquet(OUT / "items.parquet")
    return items[items["split"] == split].reset_index(drop=True)


def sample() -> dict:
    s = complete("sft", _items("dpo"), N_SAMPLES, TEMPERATURE)
    s.to_parquet(OUT / "dpo_samples.parquet", index=False)
    return {"samples": len(s), "finish_reasons": s["finish_reason"].value_counts().to_dict()}


def score() -> dict:
    s = pd.read_parquet(OUT / "dpo_samples.parquet")
    notes = _items("dpo").set_index("note_id")["note"]
    f = faithfulness(s["text"].tolist(), notes.loc[s["note_id"]].tolist(), "judge")
    s = pd.concat([s, f], axis=1)
    s.to_parquet(OUT / "dpo_samples_scored.parquet", index=False)
    return {"scored": len(s), "judge_errors": int(f["judge_error"].sum())}


def choose(group: pd.DataFrame, min_gap: float | None = None) -> tuple[int, int] | None:
    """(chosen, rejected) row labels for one prompt, or None if no pair qualifies. With
    `min_gap`, the original gap rule is applied instead of fully-supported-vs-not."""
    best, key = None, None
    for a, b in combinations(group.index, 2):
        hi, lo = (a, b) if group.at[a, "faith"] >= group.at[b, "faith"] else (b, a)
        gap = group.at[hi, "faith"] - group.at[lo, "faith"]
        wa, wb = group.at[a, "words"], group.at[b, "words"]
        if max(wa, wb) > LENGTH_RATIO * min(wa, wb):
            continue
        if min_gap is None and not (group.at[hi, "faith"] == 1.0 and group.at[lo, "faith"] < 1.0):
            continue
        if min_gap is not None and gap < min_gap:
            continue
        k = (gap, -abs(wa - wb))
        if key is None or k > key:
            best, key = (hi, lo), k
    return best


def build() -> dict:
    s = pd.read_parquet(OUT / "dpo_samples_scored.parquet")
    s["words"] = s["text"].str.split().str.len()
    usable = s[(s["finish_reason"] == "stop") & ~s["judge_error"] & (s["n_claims"] > 0)
               & (s["words"] > 0)].copy()
    usable["faith"] = usable["n_supported"] / usable["n_claims"]
    items = _items("dpo").set_index("note_id")
    rows = []
    for nid, g in usable.groupby("note_id"):
        pick = choose(g)
        if pick:
            c, r = g.loc[pick[0]], g.loc[pick[1]]
            rows.append({"note_id": nid, "note": items.at[nid, "note"],
                         "question": items.at[nid, "question"], "chosen": c["text"],
                         "rejected": r["text"], "faith_chosen": c["faith"],
                         "faith_rejected": r["faith"], "words_chosen": c["words"],
                         "words_rejected": r["words"]})
    pairs = pd.DataFrame(rows)
    pairs.to_parquet(OUT / "dpo_pairs.parquet", index=False)
    per_prompt = usable.groupby("note_id")["faith"]
    summary = {
        "prompts": int(items.shape[0]), "samples": len(s), "usable_samples": len(usable),
        "sft_sample_faithfulness": {"mean": float(usable["faith"].mean()),
                                    "share_fully_supported": float((usable["faith"] == 1)
                                                                   .mean())},
        "prompts_with_any_unsupported_sample": float((per_prompt.min() < 1).mean()),
        "pairs": len(pairs),
        "pair_gap_mean": float((pairs["faith_chosen"] - pairs["faith_rejected"]).mean())
        if len(pairs) else None,
        "words_chosen_vs_rejected": [float(pairs["words_chosen"].mean()),
                                     float(pairs["words_rejected"].mean())] if len(pairs)
        else None,
        "rules": {"pair": "chosen fully supported, rejected with >= 1 unsupported claim",
                  "length_ratio": LENGTH_RATIO, "samples": N_SAMPLES,
                  "temperature": TEMPERATURE},
        "pairs_under_original_rule": int(sum(
            choose(g, ORIGINAL_MIN_GAP) is not None for _, g in usable.groupby("note_id"))),
        "original_rule": f"faithfulness gap >= {ORIGINAL_MIN_GAP}",
    }
    (config.REPORTS / "summ_pairs.json").write_text(json.dumps(summary, indent=2))
    return summary
