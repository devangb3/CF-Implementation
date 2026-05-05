"""
Ablation: Robustness of CRS attributions and repair outcomes to stochastic LLM outputs.

Motivation (mo7N):
  Are CRS attributions and repair outcomes stable across independent intervention
  proposal generations, or are they artifacts of random LLM sampling?

Protocol:
  - 50-problem random subset of GSM8K (fixed seed for reproducibility)
  - Each problem: agent runs ONCE → fixed trace (pass or fail)
  - For each failed trace: intervention proposal generation runs NUM_RUNS=3
    times independently with natural LLM stochasticity (temperature=0.7)
  - K=3 repair proposals per causal step per run (default in CounterfactualRepair)
  - Multi-agent critique is skipped — this ablation isolates intervention/repair variance
  - Metrics (per run, then mean ± std across runs):
      * Repair rate   = fraction of failed problems with ≥1 successful repair
      * Post-repair accuracy = (initially correct + repaired) / 50

Results → Ablation_Stochasticity/results/
"""

import os
import sys
import json
import random
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from causal_flow import CausalFlow
from llm_client import LLMClient
from math_reexecutor import MathReexecutor
from trace_logger import TraceLogger, StepType
from experiments.gsm8k.gsm8k_agent import GSM8KAgent

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_PROBLEMS = 50
NUM_RUNS = 3       # independent intervention-generation passes per failed trace
K_PROPOSALS = 3   # repair proposals per causal step (default in CounterfactualRepair)
RANDOM_SEED = 42
MODEL = "google/gemini-2.0-flash-lite-001"  # same model as main GSM8K experiment
RESULTS_DIR = Path(__file__).parent / "results"
# ───────────────────────────────────────────────────────────────────────────────


def sample_gsm8k(num_problems: int, seed: int) -> List[Dict[str, str]]:
    """Randomly sample problems from GSM8K test split with a fixed seed."""
    dataset = load_dataset("gsm8k", "main", split="test")
    reexecutor = MathReexecutor()

    all_problems = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "gold_answer": str(
                reexecutor.extract_number(item["answer"]) or item["answer"]
            ),
            "dataset_index": i,
        }
        for i, item in enumerate(dataset)
    ]

    rng = random.Random(seed)
    return rng.sample(all_problems, num_problems)


def build_execution_context(
    trace: TraceLogger,
    question: str,
    gold_answer: str,
    problem_id: str,
) -> Dict[str, Any]:
    logs = "\n".join(
        str(step.tool_output)
        for step in trace.steps
        if step.step_type == StepType.TOOL_RESPONSE and step.tool_output
    )
    return {
        "question": question,
        "gold_answer": gold_answer,
        "agent_final_answer": trace.final_answer or "",
        "logs": logs,
        "problem_id": problem_id,
    }


def run_one_causalflow(
    trace: TraceLogger,
    execution_context: Dict[str, Any],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    """
    One independent CausalFlow analysis on a failed trace.

    Creates a fresh CausalFlow instance (and therefore fresh LLMClient) so each
    call makes entirely new API requests — no result reuse across runs.
    """
    cf = CausalFlow(api_key=api_key, model=model, mongo_storage=None)
    try:
        analysis = cf.analyze_trace(
            trace,
            reexecutor=None,                          # LLM-based outcome prediction
            execution_context=execution_context,
            skip_critique=True,                       # isolate intervention/repair variance
            intervene_step_types={
                StepType.TOOL_CALL,
                StepType.LLM_RESPONSE,
                StepType.REASONING,
                StepType.TOOL_RESPONSE,
            },
        )
        rm = analysis.get("metrics", {}).get("repair_metrics", {})
        return {
            "successful_repairs": rm.get("successful_repairs", 0),
            "total_repairs_attempted": rm.get("total_repairs_attempted", 0),
            "repair_success_rate": rm.get("success_rate", 0.0),
            "causal_steps_identified": len(
                analysis.get("causal_attribution", {}).get("causal_steps", [])
            ),
            "repaired": rm.get("successful_repairs", 0) > 0,
            "error": None,
        }
    except Exception as exc:
        print(f"    [CausalFlow error] {exc}")
        return {
            "successful_repairs": 0,
            "total_repairs_attempted": 0,
            "repair_success_rate": 0.0,
            "causal_steps_identified": 0,
            "repaired": False,
            "error": str(exc),
        }


def compute_summary(
    num_total: int,
    num_initially_correct: int,
    per_run_repaired: List[List[bool]],  # per_run_repaired[run_idx] = [bool, ...] over failed problems
    elapsed_seconds: float,
) -> Dict[str, Any]:
    num_failed = len(per_run_repaired[0]) if per_run_repaired else 0

    per_run_metrics = []
    for run_idx, repaired_flags in enumerate(per_run_repaired):
        num_repaired = sum(repaired_flags)
        repair_rate = num_repaired / num_failed if num_failed > 0 else 0.0
        post_repair_acc = (num_initially_correct + num_repaired) / num_total
        per_run_metrics.append(
            {
                "run_idx": run_idx,
                "num_repaired": num_repaired,
                "repair_rate": round(repair_rate, 4),
                "post_repair_accuracy": round(post_repair_acc, 4),
            }
        )

    repair_rates = [m["repair_rate"] for m in per_run_metrics]
    accuracies = [m["post_repair_accuracy"] for m in per_run_metrics]

    return {
        "num_total": num_total,
        "num_initially_correct": num_initially_correct,
        "num_failed": num_failed,
        "initial_accuracy": round(num_initially_correct / num_total, 4),
        "per_run_metrics": per_run_metrics,
        "repair_rate": {
            "mean": round(float(np.mean(repair_rates)), 4),
            "std": round(float(np.std(repair_rates)), 4),
            "per_run": repair_rates,
        },
        "post_repair_accuracy": {
            "mean": round(float(np.mean(accuracies)), 4),
            "std": round(float(np.std(accuracies)), 4),
            "per_run": accuracies,
        },
        "elapsed_minutes": round(elapsed_seconds / 60.0, 2),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE — Stochasticity Robustness")
    print("=" * 60)
    print(f"Total problems       : {summary['num_total']}")
    print(
        f"Initially correct    : {summary['num_initially_correct']}"
        f" ({summary['initial_accuracy']:.1%})"
    )
    print(f"Failed (analyzed)    : {summary['num_failed']}")
    print()
    rr = summary["repair_rate"]
    pa = summary["post_repair_accuracy"]
    print(f"Repair Rate          : {rr['mean']:.4f} ± {rr['std']:.4f}")
    print(f"Post-Repair Accuracy : {pa['mean']:.4f} ± {pa['std']:.4f}")
    print()
    print("Per-run breakdown:")
    for m in summary["per_run_metrics"]:
        print(
            f"  Run {m['run_idx']}: "
            f"repair_rate={m['repair_rate']:.4f}, "
            f"post_repair_accuracy={m['post_repair_accuracy']:.4f}, "
            f"repaired={m['num_repaired']}"
        )
    print(f"\nElapsed: {summary['elapsed_minutes']:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in .env")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {NUM_PROBLEMS} GSM8K problems (seed={RANDOM_SEED})...")
    problems = sample_gsm8k(NUM_PROBLEMS, RANDOM_SEED)
    print(f"Loaded {len(problems)} problems\n")

    agent = GSM8KAgent(
        llm_client=LLMClient(api_key=api_key, model=MODEL),
        model=MODEL,
    )

    experiment_start = time.time()

    results: Dict[str, Any] = {
        "config": {
            "num_problems": NUM_PROBLEMS,
            "num_runs": NUM_RUNS,
            "k_proposals": K_PROPOSALS,
            "random_seed": RANDOM_SEED,
            "model": MODEL,
            "skip_critique": True,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": [],
        "summary": {},
    }

    # per_run_repaired[run_idx] accumulates one bool per *failed* problem
    per_run_repaired: List[List[bool]] = [[] for _ in range(NUM_RUNS)]
    num_initially_correct = 0

    for prob_idx, problem in enumerate(tqdm(problems, desc="Problems")):
        question = problem["question"]
        gold_answer = problem["gold_answer"]
        problem_id = f"ablation_{prob_idx}"

        prob_result: Dict[str, Any] = {
            "problem_idx": prob_idx,
            "dataset_index": problem["dataset_index"],
            "question": (question[:120] + "...") if len(question) > 120 else question,
            "gold_answer": gold_answer,
            "initially_correct": None,
            "agent_answer": None,
            "runs": [],
        }

        # ── Step 1: solve once ───────────────────────────────────────────────
        try:
            trace: TraceLogger = agent.solve(question, gold_answer)
            prob_result["initially_correct"] = trace.success
            prob_result["agent_answer"] = trace.final_answer
        except Exception as exc:
            print(f"\nAgent error on problem {prob_idx}: {exc}")
            prob_result["initially_correct"] = False
            prob_result["agent_error"] = str(exc)
            results["problems"].append(prob_result)
            # Treat as failed but un-analyzable
            for run_idx in range(NUM_RUNS):
                per_run_repaired[run_idx].append(False)
            continue

        if trace.success:
            num_initially_correct += 1
            results["problems"].append(prob_result)
            continue  # no CausalFlow analysis needed

        # ── Step 2: build context (shared across runs) ───────────────────────
        ctx = build_execution_context(trace, question, gold_answer, problem_id)

        # ── Step 3: run intervention pipeline NUM_RUNS times in parallel ───────
        print(f"\n[{prob_idx+1}/{NUM_PROBLEMS}] Failed → running {NUM_RUNS} analyses in parallel")
        run_results: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=NUM_RUNS) as pool:
            futures = {
                pool.submit(run_one_causalflow, trace, ctx, api_key, MODEL): run_idx
                for run_idx in range(NUM_RUNS)
            }
            for future in as_completed(futures):
                run_idx = futures[future]
                run_result = future.result()
                run_results[run_idx] = run_result
                print(
                    f"  Run {run_idx + 1} done — "
                    f"repaired={run_result['repaired']}, "
                    f"successful_repairs={run_result['successful_repairs']}, "
                    f"causal_steps={run_result['causal_steps_identified']}"
                )

        for run_idx in range(NUM_RUNS):
            run_result = run_results[run_idx]
            prob_result["runs"].append({"run_idx": run_idx, **run_result})
            per_run_repaired[run_idx].append(run_result["repaired"])

        results["problems"].append(prob_result)

        # Checkpoint after each problem so a crash doesn't lose everything
        checkpoint_path = RESULTS_DIR / "ablation_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Step 4: compute and save summary ────────────────────────────────────
    summary = compute_summary(
        num_total=NUM_PROBLEMS,
        num_initially_correct=num_initially_correct,
        per_run_repaired=per_run_repaired,
        elapsed_seconds=time.time() - experiment_start,
    )
    results["summary"] = summary

    results_path = RESULTS_DIR / "ablation_results.json"
    summary_path = RESULTS_DIR / "ablation_summary.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)


if __name__ == "__main__":
    main()
