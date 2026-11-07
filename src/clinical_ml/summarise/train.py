"""LoRA SFT, then DPO, on Qwen3-8B with TRL, on one GPU.

Run in the GPU env:  .venv-vllm/bin/python -m clinical_ml.summarise.train sft|dpo

Settings are fixed in advance (README, Phase 2b protocol). Prompts are rendered with the model's own
chat template with thinking off, exactly as vLLM renders them at inference, and TRL adds
the end-of-turn token to each completion.
* SFT: prompt -> reference summary, loss on the completion only.
* DPO: starts from the SFT adapter. TRL copies that adapter as a frozen "ref" adapter, so
  the reference model is the SFT model, and the result is still one servable LoRA.
"""

import json
import sys
import time

import pandas as pd

from clinical_ml import config
from clinical_ml.rag.models import ADAPTERS, MODELS, use_project_hf_cache
from clinical_ml.summarise.data import OUT, messages

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                           "down_proj"]}
COMMON = {"bf16": True, "gradient_checkpointing": True, "max_length": 2048,
          "lr_scheduler_type": "cosine", "warmup_steps": 0.03, "logging_steps": 10,
          "save_strategy": "no", "report_to": "none", "seed": config.SEED}
SFT = {"num_train_epochs": 2, "learning_rate": 1e-4, "per_device_train_batch_size": 8,
       "gradient_accumulation_steps": 2}
DPO = {"num_train_epochs": 1, "learning_rate": 2e-5, "beta": 0.1,     # sigmoid loss
       "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4}


def _tokenizer():
    from transformers import AutoTokenizer

    m = MODELS["summarise"]
    return AutoTokenizer.from_pretrained(m.name, revision=m.revision)


def render(tok, note: str, question: str) -> str:
    return tok.apply_chat_template(messages(note, question), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def _base():
    import torch
    from transformers import AutoModelForCausalLM

    m = MODELS["summarise"]
    return AutoModelForCausalLM.from_pretrained(m.name, revision=m.revision,
                                                dtype=torch.bfloat16, device_map="cuda")


def _finish(trainer, stage: str, n_examples: int, t0: float) -> dict:
    import torch

    out_dir = ADAPTERS / stage
    trainer.model.save_pretrained(out_dir, selected_adapters=["default"])
    log = {"stage": stage, "examples": n_examples, "seconds": round(time.time() - t0, 1),
           "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
           "settings": {"lora": LORA, "common": COMMON, stage: SFT if stage == "sft" else DPO},
           "log_history": trainer.state.log_history}
    (config.REPORTS / f"summ_train_{stage}.json").write_text(json.dumps(log, indent=2))
    return log


def sft() -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    tok = _tokenizer()
    items = pd.read_parquet(OUT / "items.parquet")
    items = items[items["split"] == "sft"]
    ds = Dataset.from_dict({
        "prompt": [render(tok, n, q) for n, q in zip(items["note"], items["question"],
                                                     strict=True)],
        "completion": items["answer"].tolist()})
    trainer = SFTTrainer(model=_base(), processing_class=tok, train_dataset=ds,
                         peft_config=LoraConfig(**LORA),
                         args=SFTConfig(output_dir=str(OUT / "runs" / "sft"),
                                        completion_only_loss=True, **COMMON, **SFT))
    trainer.train()
    return _finish(trainer, "sft", len(ds), t0)


def dpo() -> dict:
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    t0 = time.time()
    tok = _tokenizer()
    pairs = pd.read_parquet(OUT / "dpo_pairs.parquet")
    ds = Dataset.from_dict({
        "prompt": [render(tok, n, q) for n, q in zip(pairs["note"], pairs["question"],
                                                     strict=True)],
        "chosen": pairs["chosen"].tolist(), "rejected": pairs["rejected"].tolist()})
    model = PeftModel.from_pretrained(_base(), ADAPTERS / "sft", is_trainable=True)
    trainer = DPOTrainer(model=model, processing_class=tok, train_dataset=ds,
                         args=DPOConfig(output_dir=str(OUT / "runs" / "dpo"), **COMMON, **DPO))
    trainer.train()
    return _finish(trainer, "dpo", len(ds), t0)


if __name__ == "__main__":
    use_project_hf_cache()
    stage = sys.argv[1]
    result = {"sft": sft, "dpo": dpo}[stage]()
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))
