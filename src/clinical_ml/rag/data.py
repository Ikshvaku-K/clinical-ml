"""Phase 2a data: Asclepius synthetic discharge summaries -> chunks + a fixed question set.

Asclepius (Kweon et al., 2024; CC BY-NC-SA 4.0) has one synthetic note and one generated
question-answer pair per patient, and the questions refer to "the patient". So retrieval
is evaluated as known-item search: a question must find its own note among all notes.

Pre-specified (fixed before any retrieval or generation run):
* Notes are deduplicated by text; the ground truth for a question is its note's id.
* Chunks follow the note's own paragraphs: a header line ("Hospital Course:") is kept
  with the paragraph after it, paragraphs are packed up to MAX_WORDS, and an over-long
  paragraph is split at sentence ends (or, failing that, into word windows).
* Questions come from tasks whose questions carry case-specific content (TASKS). The
  templated tasks and paraphrasing (which quotes the note verbatim) are excluded, as are
  rows whose reference answer calls the question unanswerable.
* A seeded sample gives N_DEV dev and N_TEST test questions from distinct notes. For
  HELD_OUT of each, the source note is removed from the index: the right behaviour for
  those is to abstain.
"""

import hashlib
import json
import re

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.sources import _fetch, sha256

REPO = "starmpcc/Asclepius-Synthetic-Clinical-Notes"
REVISION = "52af6897eb1bf275a8937ae54bf42da381fe9421"
FILE = "synthetic.csv"
FILE_SHA256 = "083b1b3531ad5865bcac23250ec04a532b70df8ff534b8aa9bee87b04cf97c84"  # HF LFS oid
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"

RAW_DIR = config.RAW / "asclepius"
OUT_DIR = config.DATA / "rag"
TASKS = ("Question Answering", "Relation Extraction", "Temporal Information Extraction",
         "Summarization")
MAX_WORDS = 150
N_DEV, N_TEST = 300, 1000
HELD_OUT = 0.10

_HEADER = re.compile(r"^[A-Z][^\n.]{0,60}:$")
_SENT = re.compile(r"(?<=[.!?])\s+")
_UNANSWERABLE = re.compile(r"\bunanswerable\b", re.I)


def download() -> dict:
    path = RAW_DIR / FILE
    if not path.exists():
        _fetch(URL, path)
    digest = sha256(path)
    if digest != FILE_SHA256:
        raise RuntimeError(f"Asclepius checksum mismatch: {digest}")
    return {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": digest}


def note_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _words(text: str) -> int:
    return len(text.split())


def _split_long(par: str) -> list[str]:
    """Split at sentence ends; a sentence longer than MAX_WORDS (e.g. a long list) is cut
    into word windows, so no chunk exceeds the embedding model's input length."""
    sents = []
    for sent in _SENT.split(par):
        w = sent.split()
        sents += [" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS)] or [sent]
    out, cur = [], []
    for sent in sents:
        if cur and _words(" ".join(cur + [sent])) > MAX_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    return out + ([" ".join(cur)] if cur else [])


def chunk_note(note: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", note.strip()) if p.strip()]
    units, header = [], None
    for p in paras:
        if _HEADER.match(p):                      # a bare header joins the next paragraph
            header = f"{header}\n{p}" if header else p
            continue
        text = f"{header}\n{p}" if header else p
        header = None
        units += _split_long(text) if _words(text) > MAX_WORDS else [text]
    if header:
        units.append(header)
    chunks, cur = [], []
    for u in units:
        if cur and _words("\n\n".join(cur + [u])) > MAX_WORDS:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(u)
    return chunks + (["\n\n".join(cur)] if cur else [])


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / FILE, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["patient_id", "note", "question", "answer", "task"], df.columns
    df["note_id"] = df["note"].map(note_id)
    return df


def sample_questions(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    eligible = df[df["task"].isin(TASKS) & ~df["answer"].str.contains(_UNANSWERABLE)]
    eligible = eligible.drop_duplicates("note_id")          # one question per note
    order = np.random.default_rng(seed).permutation(len(eligible))
    picked = eligible.iloc[order[:N_DEV + N_TEST]].reset_index(drop=True)
    picked["split"] = ["dev"] * N_DEV + ["test"] * N_TEST
    rng = np.random.default_rng(seed + 1)
    picked["in_index"] = True
    for split, n in (("dev", N_DEV), ("test", N_TEST)):
        idx = picked.index[picked["split"] == split]
        held = rng.choice(idx, int(round(HELD_OUT * n)), replace=False)
        picked.loc[held, "in_index"] = False
    picked.insert(0, "qid", [f"{s}-{i:04d}" for s, i in
                             zip(picked["split"], picked.groupby("split").cumcount(), strict=True)])
    return picked[["qid", "split", "task", "patient_id", "note_id", "in_index", "question",
                   "answer"]]


def build_chunks(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    notes = df.drop_duplicates("note_id")
    notes = notes[~notes["note_id"].isin(exclude)]
    rows = [(f"{nid}-{i}", nid, i, text)
            for nid, note in zip(notes["note_id"], notes["note"], strict=True)
            for i, text in enumerate(chunk_note(note))]
    return pd.DataFrame(rows, columns=["chunk_id", "note_id", "chunk_index", "text"])


def run() -> dict:
    meta = download()
    df = load()
    questions = sample_questions(df)
    chunks = build_chunks(df, set(questions.loc[~questions["in_index"], "note_id"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions.to_parquet(OUT_DIR / "questions.parquet", index=False)
    chunks.to_parquet(OUT_DIR / "chunks.parquet", index=False)
    words = chunks["text"].map(_words)
    eligible = df[df["task"].isin(TASKS)]
    summary = {
        "source": meta,
        "rows": len(df), "unique_notes": int(df["note_id"].nunique()),
        "rows_by_task": df["task"].value_counts().to_dict(),
        "eligible_rows": int(len(eligible)),
        "excluded_unanswerable_reference": int(eligible["answer"].str.contains(
            _UNANSWERABLE).sum()),
        "indexed_notes": int(chunks["note_id"].nunique()), "chunks": len(chunks),
        "chunk_words": {"median": float(words.median()), "p95": float(words.quantile(0.95)),
                        "max": int(words.max())},
        "questions": {s: {"n": int((questions.split == s).sum()),
                          "held_out": int(((questions.split == s) & ~questions.in_index).sum()),
                          "by_task": questions[questions.split == s]["task"].value_counts()
                          .to_dict()}
                      for s in ("dev", "test")},
    }
    (config.REPORTS / "rag_data.json").write_text(json.dumps(summary, indent=2))
    return summary
