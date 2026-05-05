#!/usr/bin/env python3
"""DPO fine-tuning of Llama-3.2-3B-Instruct on CausalFlow MBPP repair pairs.

Usage:
    python DPO/train_dpo.py
    python DPO/train_dpo.py --load_in_4bit          # for T4 (15 GB VRAM)
    python DPO/train_dpo.py --model_id meta-llama/Llama-3.2-1B-Instruct  # fallback
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_and_split(pairs_path: str, test_fraction: float = 0.1, seed: int = 42):
    pairs = json.load(open(pairs_path))

    # Stratified by step_type to preserve llm_response / reasoning ratio
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p["step_type"]].append(p)

    rng = random.Random(seed)
    train, test = [], []
    for group in by_type.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction))
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    return train, test


def format_pair(p: dict) -> dict:
    parts = [f"Problem: {p['problem_statement']}"]
    if p["prior_context"].strip():
        parts.append(f"Context: {p['prior_context'].strip()}")
    parts.append("Produce a corrected version of the following step:")
    return {
        "prompt": "\n".join(parts),
        "chosen": p["chosen"],
        "rejected": p["rejected"],
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, test_pairs = load_and_split(
        args.pairs, test_fraction=args.test_fraction
    )
    type_counts = defaultdict(int)
    for p in train_pairs:
        type_counts[p["step_type"]] += 1
    print(f"Train: {len(train_pairs)} pairs {dict(type_counts)}")
    print(f"Test:  {len(test_pairs)} pairs")

    # Save test split so eval_dpo.py can pick it up without re-splitting
    (output_dir / "test_pairs.json").write_text(json.dumps(test_pairs, indent=2))

    train_dataset = Dataset.from_list([format_pair(p) for p in train_pairs])

    bnb_config = None
    torch_dtype = torch.bfloat16
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None  # set via bnb_config

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=args.hf_token,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=not args.load_in_4bit,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT uses base as implicit reference
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Training ...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="DPO/mbpp_dpo_pairs.json")
    parser.add_argument("--output_dir", default="DPO/output")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--hf_token", default=None, help="HuggingFace access token for gated weights")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--load_in_4bit", action="store_true", help="QLoRA mode for T4 GPUs")
    args = parser.parse_args()
    main(args)
