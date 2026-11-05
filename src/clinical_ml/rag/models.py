"""Pinned open-weight models for Phase 2 and how vLLM serves them on the GPU pod.

Every model is pinned to a Hugging Face commit, and every server speaks the OpenAI API
on localhost. `clinical-ml rag serve <role>` starts one (vLLM lives in its own env,
.venv-vllm, because it pins its own torch).
"""

import os
import sys
from dataclasses import dataclass, field

from clinical_ml import config


@dataclass(frozen=True)
class Served:
    name: str
    revision: str
    port: int
    gpu_memory: float
    args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


MODELS = {
    # Dense retriever (MIT). Queries get bge's retrieval instruction; passages do not.
    "embed": Served("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                    8101, 0.05, ("--runner", "pooling", "--convert", "embed")),
    # Cross-encoder reranker (Apache-2.0).
    "rerank": Served("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                     8102, 0.08, ("--runner", "pooling", "--max-model-len", "1024")),
    # Generator, also the 2b fine-tuning base (Apache-2.0). Thinking mode is turned off
    # per request, so answers are direct.
    "generate": Served("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
                       8103, 0.40, ("--max-model-len", "8192")),
    # Judge from a different model family, so the generator is not graded by its own
    # kind (MIT).
    "judge": Served("microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
                    8104, 0.40, ("--max-model-len", "8192")),
    # 2b: the generator again, serving the SFT and DPO LoRA adapters that exist so far.
    "summarise": Served("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
                        8103, 0.85, ("--max-model-len", "8192", "--enable-lora",
                                     "--max-lora-rank", "16")),
    # 2b evaluation judge, a third model family (Apache-2.0). FP8 build of
    # Mistral-Small-3.2-24B-Instruct-2506, because the bf16 weights (48 GB) do not fit the
    # pod's disk next to everything else.
    "eval_judge": Served("RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
                         "9bc507cf77b5045acd65c8c76b49cf081fe8e985", 8105, 0.85,
                         ("--max-model-len", "8192", "--tokenizer-mode", "mistral",
                          "--config-format", "mistral", "--load-format", "mistral")),
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HF_HOME = config.ROOT / ".tools" / "hf"


def use_project_hf_cache() -> None:
    """Keep every Hugging Face download inside the project (never ~/.cache)."""
    os.environ["HF_HOME"] = str(HF_HOME)


ADAPTERS = config.DATA / "summarise" / "adapters"


def purge(keep: tuple[str, ...] = ()) -> list[str]:
    """Delete cached models (all but the roles in `keep`) the way huggingface_hub expects,
    so shared blobs go too. The pod's disk holds only one large model at a time."""
    from huggingface_hub import scan_cache_dir

    use_project_hf_cache()
    keep_repos = {MODELS[r].name for r in keep}
    info = scan_cache_dir(HF_HOME / "hub")
    gone = [r for r in info.repos if r.repo_id not in keep_repos]
    info.delete_revisions(*[rev.commit_hash for r in gone for rev in r.revisions]).execute()
    # huggingface_hub 2.4 keeps file contents in a shared hub/blobs store that the call
    # above leaves behind; remove any blob no remaining snapshot links to.
    hub = HF_HOME / "hub"
    linked = {p.resolve() for p in hub.glob("models--*/snapshots/*/**/*") if p.is_symlink()}
    for blob in (hub / "blobs").rglob("*"):
        if blob.is_file() and blob.resolve() not in linked:
            blob.unlink()
    return [r.repo_id for r in gone]


def serve(role: str) -> None:
    """Replace this process with a vLLM server for one role."""
    m = MODELS[role]
    vllm = config.ROOT / ".venv-vllm" / "bin" / "vllm"
    use_project_hf_cache()
    extra = []
    if role == "summarise":
        found = [f"{a}={ADAPTERS / a}" for a in ("sft", "dpo") if (ADAPTERS / a).exists()]
        extra = ["--lora-modules", *found] if found else []
    argv = [str(vllm), "serve", m.name, "--revision", m.revision, "--host", "127.0.0.1",
            "--port", str(m.port), "--gpu-memory-utilization", str(m.gpu_memory),
            "--served-model-name", role, *m.args, *extra]
    print(" ".join(argv), file=sys.stderr, flush=True)
    os.execv(argv[0], argv)
