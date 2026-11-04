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
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HF_HOME = config.ROOT / ".tools" / "hf"


def use_project_hf_cache() -> None:
    """Keep every Hugging Face download inside the project (never ~/.cache)."""
    os.environ["HF_HOME"] = str(HF_HOME)


def serve(role: str) -> None:
    """Replace this process with a vLLM server for one role."""
    m = MODELS[role]
    vllm = config.ROOT / ".venv-vllm" / "bin" / "vllm"
    use_project_hf_cache()
    argv = [str(vllm), "serve", m.name, "--revision", m.revision, "--host", "127.0.0.1",
            "--port", str(m.port), "--gpu-memory-utilization", str(m.gpu_memory),
            "--served-model-name", role, *m.args]
    print(" ".join(argv), file=sys.stderr, flush=True)
    os.execv(argv[0], argv)
