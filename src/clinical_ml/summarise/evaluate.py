"""2b evaluation: base vs SFT vs SFT+DPO on the held-out summarisation notes.

* generate: greedy summaries from the three models (the base model, and the "sft" and
  "dpo" LoRA adapters on the same vLLM server).
* judge: the judge served for a role checks every claim against the note and grades
  completeness against the reference (correct / partial / incorrect). The primary judge
  is "eval_judge" (Mistral-Small FP8, a third model family); phi-4, which ranked the DPO
  pairs, is run as well to show how far the two disagree.
* metrics_report: per model and judge, with paired note-level bootstrap CIs for the
  differences DPO - SFT, SFT - base and DPO - base. ROUGE-L and length need no judge.
"""

import json

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.eval.metrics import bootstrap_fn
from clinical_ml.rag.judge import Grade, _llm
from clinical_ml.summarise.data import OUT
from clinical_ml.summarise.pairs import _items, complete, faithfulness

MODELS_EVAL = ("summarise", "sft", "dpo")         # vLLM model names: base, then adapters
LABELS = {"summarise": "base", "sft": "sft", "dpo": "dpo"}
JUDGES = ("eval_judge", "judge")
N_BOOT = 1000


def generate() -> dict:
    items = pd.concat([_items("dev"), _items("test")], ignore_index=True)
    outs = []
    for m in MODELS_EVAL:
        o = complete(m, items, n=1, temperature=0.0, seed_offset=10_000)
        o["model"] = LABELS[m]
        outs.append(o.drop(columns="sample"))
    out = pd.concat(outs, ignore_index=True).merge(items[["note_id", "split"]], on="note_id")
    out.to_parquet(OUT / "eval_outputs.parquet", index=False)
    return {"rows": len(out), "finish_reasons": out.groupby("model")["finish_reason"]
            .value_counts().unstack(fill_value=0).to_dict("index")}


def judge(role: str = "eval_judge") -> dict:
    out = pd.read_parquet(OUT / "eval_outputs.parquet")
    items = pd.concat([_items("dev"), _items("test")]).set_index("note_id")
    f = faithfulness(out["text"].tolist(), items.loc[out["note_id"], "note"].tolist(), role)
    grades = _llm(Grade, role).batch(
        [{"question": items.at[n, "question"], "reference": items.at[n, "answer"],
          "answer": t} for n, t in zip(out["note_id"], out["text"], strict=True)],
        config={"max_concurrency": 64}, return_exceptions=True)
    res = pd.concat([out[["note_id", "split", "model"]], f], axis=1)
    res["grade"] = [g.label if isinstance(g, Grade) else "judge_error" for g in grades]
    res["judge"] = role
    res.to_parquet(OUT / f"eval_judged_{role}.parquet", index=False)
    return {"judge": role, "rows": len(res), "claim_errors": int(f["judge_error"].sum()),
            "grade_errors": int((res["grade"] == "judge_error").sum())}


def _rouge_l(texts: list[str], refs: list[str]) -> np.ndarray:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return np.array([sc.score(r, t)["rougeL"].fmeasure for t, r in zip(texts, refs,
                                                                         strict=True)])


def _per_note(split: str) -> pd.DataFrame:
    """One row per (note, model): judge-free and per-judge measures."""
    out = pd.read_parquet(OUT / "eval_outputs.parquet")
    out = out[out["split"] == split].copy()
    items = _items(split).set_index("note_id")
    out["rouge_l"] = _rouge_l(out["text"].tolist(), items.loc[out["note_id"], "answer"].tolist())
    out["words"] = out["text"].str.split().str.len()
    out["finished"] = out["finish_reason"] == "stop"
    for role in JUDGES:
        j = pd.read_parquet(OUT / f"eval_judged_{role}.parquet")
        j = j[j["split"] == split].set_index(["note_id", "model"])
        key = pd.MultiIndex.from_frame(out[["note_id", "model"]])
        ok = ~j.loc[key, "judge_error"].to_numpy() & (j.loc[key, "n_claims"].to_numpy() > 0)
        n_claims, n_sup = j.loc[key, "n_claims"].to_numpy(), j.loc[key, "n_supported"].to_numpy()
        out[f"{role}_faith"] = np.where(ok, n_sup / np.maximum(n_claims, 1), np.nan)
        out[f"{role}_unsupported"] = np.where(ok, n_claims - n_sup, np.nan)
        grade = j.loc[key, "grade"].to_numpy()
        valid = grade != "judge_error"
        out[f"{role}_complete"] = np.where(valid, np.isin(grade, ["correct", "partial"]), np.nan)
    return out


MEASURES = ["rouge_l", "words", "finished"] + [f"{r}_{m}" for r in JUDGES
                                                for m in ("faith", "unsupported", "complete")]


def metrics_report() -> dict:
    report = {"protocol": {"n_boot": N_BOOT, "bootstrap_unit": "note",
                           "primary_judge": JUDGES[0], "models": list(LABELS.values())},
              "splits": {}}
    for split in ("dev", "test"):
        df = _per_note(split)
        wide = df.pivot(index="note_id", columns="model", values=MEASURES)
        arrays = {f"{m}|{k}": wide[(m, k)].to_numpy(dtype=float)
                  for m in MEASURES for k in LABELS.values()}

        def fn(**a):
            r = {}
            for m in MEASURES:
                for k in LABELS.values():
                    r[f"{m}|{k}"] = float(np.nanmean(a[f"{m}|{k}"]))
                for hi, lo in (("dpo", "sft"), ("sft", "base"), ("dpo", "base")):
                    d = a[f"{m}|{hi}"] - a[f"{m}|{lo}"]
                    r[f"{m}|{hi}-{lo}"] = float(np.nanmean(d))
            return r

        ci = bootstrap_fn(fn, arrays, wide.index.to_numpy(), N_BOOT, config.SEED)
        report["splits"][split] = {"notes": int(len(wide)), "results": ci}
    (config.REPORTS / "summ_eval.json").write_text(json.dumps(report, indent=2))
    test = report["splits"]["test"]["results"]
    return {k: test[k] for k in test if k.startswith(("eval_judge_faith", "rouge_l|", "words|"))}
