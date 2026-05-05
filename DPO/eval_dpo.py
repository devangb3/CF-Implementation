#!/usr/bin/env python3
"""Evaluate the DPO fine-tuned model on held-out MBPP pairs.

No gold answer is given to the model at inference — only problem statement +
prior context. Correctness is determined by exec()-based execution of generated
code against the MBPP test assertions (same logic as Docker reexecutor, no Docker
dependency needed for evaluation).

Usage:
    python DPO/eval_dpo.py --model_dir DPO/output/final --test_pairs DPO/output/test_pairs.json
"""

import argparse
import json
import textwrap
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── exec-based code evaluation ────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Pull code out of markdown fences if present, otherwise return as-is."""
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # inside a fence
                # strip language tag on first line
                lines = part.strip().splitlines()
                if lines and not lines[0].strip().startswith(("def ", "import ", "class ", "#")):
                    lines = lines[1:]
                return "\n".join(lines)
    return text.strip()


def exec_test(code: str, test_list: list[str], timeout_sec: int = 5) -> bool:
    """Run code + test assertions in a fresh namespace. Returns True if all pass."""
    clean = extract_code(code)
    full_source = clean + "\n" + "\n".join(test_list)
    try:
        exec(compile(full_source, "<string>", "exec"), {})
        return True
    except Exception:
        return False


# ── prompt formatting (must match train_dpo.py exactly) ──────────────────────

def build_prompt(pair: dict) -> str:
    parts = [f"Problem: {pair['problem_statement']}"]
    if pair["prior_context"].strip():
        parts.append(f"Context: {pair['prior_context'].strip()}")
    parts.append("Produce a corrected version of the following step:")
    return "\n".join(parts)


# ── generation ────────────────────────────────────────────────────────────────

def generate_repair(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    test_pairs = json.load(open(args.test_pairs))
    print(f"Evaluating on {len(test_pairs)} held-out pairs")

    # Load MBPP test assertions from HuggingFace (task_id → test_list)
    print("Loading MBPP dataset for test assertions ...")
    mbpp = load_dataset("Muennighoff/mbpp", split="test+validation+train")
    mbpp_tests = {row["task_id"]: row["test_list"] for row in mbpp}

    # Load fine-tuned model
    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Support both merged models and PEFT adapters
    model_path = Path(args.model_dir)
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        base_id = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        print(f"  Loading base {base_id} + LoRA adapter")
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto", token=args.hf_token
        )
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()

    results = []
    code_pass = code_total = 0

    for pair in test_pairs:
        task_id = int(pair["problem_id"].replace("mbpp-", ""))
        tests = mbpp_tests.get(task_id, [])
        prompt = build_prompt(pair)
        generated = generate_repair(model, tokenizer, prompt, args.max_new_tokens)

        entry = {
            "problem_id": pair["problem_id"],
            "step_type": pair["step_type"],
            "generated": generated,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "passed": None,
        }

        if pair["step_type"] == "llm_response" and tests:
            passed = exec_test(generated, tests)
            entry["passed"] = passed
            code_total += 1
            if passed:
                code_pass += 1
            status = "PASS" if passed else "FAIL"
            print(f"  {pair['problem_id']} [{pair['step_type']}] → {status}")
        else:
            # Reasoning steps: report similarity to chosen as proxy
            from difflib import SequenceMatcher
            sim = SequenceMatcher(None, generated, pair["chosen"]).ratio()
            entry["similarity_to_chosen"] = round(sim, 3)
            print(f"  {pair['problem_id']} [{pair['step_type']}] → similarity={sim:.3f}")

        results.append(entry)

    # Summary
    repair_rate = code_pass / code_total if code_total else 0.0
    reasoning_pairs = [r for r in results if r["step_type"] == "reasoning"]
    avg_sim = (
        sum(r.get("similarity_to_chosen", 0) for r in reasoning_pairs) / len(reasoning_pairs)
        if reasoning_pairs else 0.0
    )

    summary = {
        "code_repair_rate": round(repair_rate, 4),
        "code_passed": code_pass,
        "code_total": code_total,
        "reasoning_avg_similarity": round(avg_sim, 4),
        "reasoning_total": len(reasoning_pairs),
        "comparison_table": {
            "CausalFlow (main, gold in prompt)": "44.9%",
            "No-gold ablation (zero-shot)": "~0%",
            "DPO fine-tuned LLaMA 3B (this run)": f"{repair_rate:.1%}",
        },
    }

    print("\n── Results ───────────────────────────────────────────")
    print(f"Code repair rate : {code_pass}/{code_total} = {repair_rate:.1%}")
    print(f"Reasoning sim   : {avg_sim:.3f} avg over {len(reasoning_pairs)} pairs")
    print("\nComparison:")
    for condition, rate in summary["comparison_table"].items():
        print(f"  {condition:<45} {rate}")

    out_path = Path(args.output_dir) / "eval_results.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="DPO/output/final")
    parser.add_argument("--test_pairs", default="DPO/output/test_pairs.json")
    parser.add_argument("--output_dir", default="DPO/output")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)
