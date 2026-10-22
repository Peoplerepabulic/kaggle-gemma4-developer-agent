#!/usr/bin/env python3
"""LoRA SFT on filtered passing trajectories (PEFT -> .safetensors).

Train on the *unquantized* Gemma 4 variant matching your chosen QAT model
(Kaggle Staff: unquantized variants are on the same Kaggle Models page).
Exported adapter goes to agent/adapters/sft_v1/ with adapter_config.json +
adapter_model.safetensors, exactly as the submission format expects.

Run inside a Kaggle L4x4 notebook (internet OFF, use the offline wheelhouse)
or on RunPod. Defaults are conservative for 12B on 96 GB total VRAM.
"""
import argparse, json
from pathlib import Path

def load_jsonl(path: str):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_gold.jsonl")
    ap.add_argument("--base", default="google/gemma-4-12b-it")  # unquantized twin of the QAT model
    ap.add_argument("--out", default="agent/adapters/sft_v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    # Imports here so --help works without GPU deps installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    rows = load_jsonl(args.data)
    assert rows, "empty SFT data — run collect_trajectories.py first"
    print(f"{len(rows)} gold trajectories from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    # TODO (Week 4): tokenize rows -> (issue + trajectory messages) as input,
    # patch-producing turns as labels (mask the rest), train with HF Trainer,
    # then: model.save_pretrained(args.out)  # writes PEFT adapter files
    print("Skeleton ready — wire the Trainer loop in Week 4 (see PLAN.md).")

if __name__ == "__main__":
    main()
