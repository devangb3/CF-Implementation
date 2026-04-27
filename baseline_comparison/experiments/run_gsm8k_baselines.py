"""
GSM8K Baseline Comparison Experiment
=====================================
Compares three iterative-reasoning baselines against a Direct (single-pass CoT) baseline
on the GSM8K math benchmark.

Baselines implemented:
    Direct          – one-shot Chain-of-Thought (lower bound)
    Self-Refine     – Madaan et al. 2023  (https://arxiv.org/abs/2303.17651)
    Self-Reflection – arXiv 2405.06682    (https://arxiv.org/pdf/2405.06682)
    Tree of Thoughts – Yao et al. 2023   (https://arxiv.org/abs/2305.10601)

Results are written to baseline_comparison/results/gsm8k_<timestamp>.json
after every problem so the run can be interrupted and inspected mid-way.

Usage:
    python baseline_comparison/experiments/run_gsm8k_baselines.py
    python baseline_comparison/experiments/run_gsm8k_baselines.py --num_rows 50
    python baseline_comparison/experiments/run_gsm8k_baselines.py --baselines direct self_refine
    python baseline_comparison/experiments/run_gsm8k_baselines.py --model google/gemini-2.0-flash-lite-001
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from math_reexecutor import MathReexecutor

baseline_root = Path(__file__).parent.parent
sys.path.insert(0, str(baseline_root))

from baselines.self_refine import SelfRefine
from baselines.self_reflection import SelfReflection
from baselines.tree_of_thoughts import TreeOfThoughts

RESULTS_DIR = baseline_root / "results"
ALL_BASELINES = ["direct", "self_refine", "self_reflection", "tree_of_thoughts"]


def _find_latest_checkpoint(prefix: str) -> Optional[Path]:
    """Return the most recently modified results file for this prefix, or None."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_checkpoint(path: Path) -> tuple:
    """Load checkpoint; return (path, meta, all_results, stats_accumulators)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    all_results = data.get("problems", {})
    raw = data.get("running_stats", data.get("summary", {}))
    # Strip derived fields so we get raw accumulators only
    stats = {
        b: {k: v for k, v in s.items()
            if k not in ("accuracy", "avg_llm_calls", "avg_elapsed_seconds")}
        for b, s in raw.items()
    }
    return path, meta, all_results, stats


def _minimality(a: str, b: str) -> float:
    x = a.split()
    y = b.split()
    L = max(len(x), len(y))
    if L == 0:
        return 1.0
    m = sum(1 for k in range(min(len(x), len(y))) if x[k] == y[k])
    return (m / L) * (1 - 0.5 * abs(len(x) - len(y)) / L)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class GSM8KLoader:
    def __init__(self) -> None:
        self._reexecutor = MathReexecutor()

    def load(self, num_rows: Optional[int] = None) -> List[Dict[str, str]]:
        dataset = load_dataset("gsm8k", "main", split="test")
        data = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
        if num_rows is not None:
            data = data[:num_rows]
        print(f"Loaded {len(data)} GSM8K problems")
        return data

    def gold_answer(self, raw_answer: str) -> str:
        num = self._reexecutor.extract_number(raw_answer)
        return str(num) if num is not None else raw_answer


# ---------------------------------------------------------------------------
# Direct baseline (plain CoT, single pass)
# ---------------------------------------------------------------------------

class DirectBaseline:
    """Single-pass Chain-of-Thought – the simplest possible approach."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.reexecutor = MathReexecutor()

    def solve(self, problem: str, gold_answer: str) -> Dict[str, Any]:
        solution = self.llm.generate(
            (
                f"Solve the following math problem step by step.\n\n"
                f"Problem: {problem}\n\n"
                f"Solution:"
            ),
            system_message="You are a careful math solver. Show all steps clearly.",
        )
        num = self.reexecutor.extract_number(solution)
        final_answer = str(num) if num is not None else solution.strip()
        is_correct = self.reexecutor.compare_answers(final_answer, gold_answer)
        return {
            "final_answer": final_answer,
            "is_correct": is_correct,
            "initial_correct": is_correct,
            "minimality": 1.0,
            "llm_calls": 1,
            "solution": solution,
        }


# ---------------------------------------------------------------------------
# Per-problem runner
# ---------------------------------------------------------------------------

def run_one(
    baseline_name: str,
    solver: Any,
    problem: str,
    gold_answer: str,
) -> Dict[str, Any]:
    start = time.time()
    try:
        result = solver.solve(problem, gold_answer)
    except Exception as exc:
        result = {
            "final_answer": "",
            "is_correct": False,
            "error": str(exc),
            "llm_calls": 0,
        }
    result["elapsed_seconds"] = round(time.time() - start, 2)
    result["baseline"] = baseline_name
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _blank_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "errors": 0,
        "total_llm_calls": 0,
        "total_elapsed_seconds": 0.0,
        "initially_correct": 0,
        "repaired": 0,
        "total_minimality": 0.0,
        "minimality_count": 0,
    }


def _update_stats(stats: Dict[str, Any], result: Dict[str, Any]) -> None:
    stats["total"] += 1
    is_correct = result.get("is_correct", False)
    stats["correct"] += int(is_correct)
    stats["errors"] += int("error" in result)
    stats["total_llm_calls"] += result.get("llm_calls", 0)
    stats["total_elapsed_seconds"] += result.get("elapsed_seconds", 0.0)
    initial_correct = result.get("initial_correct")
    if initial_correct is not None:
        stats["initially_correct"] += int(initial_correct)
        if not initial_correct and is_correct:
            stats["repaired"] += 1
    minimality = result.get("minimality")
    if minimality is not None:
        stats["total_minimality"] += minimality
        stats["minimality_count"] += 1


def _finalize_stats(stats: Dict[str, Any]) -> None:
    n = stats["total"]
    stats["accuracy"] = stats["correct"] / n if n > 0 else 0.0
    stats["avg_llm_calls"] = stats["total_llm_calls"] / n if n > 0 else 0.0
    stats["avg_elapsed_seconds"] = stats["total_elapsed_seconds"] / n if n > 0 else 0.0
    initially_wrong = n - stats["initially_correct"]
    stats["repair_rate"] = stats["repaired"] / initially_wrong if initially_wrong > 0 else None
    mc = stats["minimality_count"]
    stats["avg_minimality"] = stats["total_minimality"] / mc if mc > 0 else None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

class GSM8KBaselineExperiment:
    def __init__(
        self,
        api_key: str,
        model: str,
        baselines_to_run: List[str],
        self_refine_max_iter: int = 4,
        tot_candidates: int = 3,
        tot_beam: int = 2,
        tot_depth: int = 3,
    ) -> None:
        self.model = model
        self.baselines_to_run = baselines_to_run
        self.loader = GSM8KLoader()
        self._api_key = api_key
        self._self_refine_max_iter = self_refine_max_iter
        self._tot_candidates = tot_candidates
        self._tot_beam = tot_beam
        self._tot_depth = tot_depth

    def _make_solvers(self) -> Dict[str, Any]:
        """Create a fresh, independent set of solvers (one per thread)."""
        def client() -> LLMClient:
            return LLMClient(api_key=self._api_key, model=self.model, temperature=0.7)

        solvers: Dict[str, Any] = {}
        if "direct" in self.baselines_to_run:
            solvers["direct"] = DirectBaseline(client())
        if "self_refine" in self.baselines_to_run:
            solvers["self_refine"] = SelfRefine(client(), max_iter=self._self_refine_max_iter)
        if "self_reflection" in self.baselines_to_run:
            solvers["self_reflection"] = SelfReflection(client())
        if "tree_of_thoughts" in self.baselines_to_run:
            solvers["tree_of_thoughts"] = TreeOfThoughts(
                client(),
                num_candidates=self._tot_candidates,
                beam_width=self._tot_beam,
                max_depth=self._tot_depth,
            )
        return solvers

    def run(
        self,
        num_rows: Optional[int] = None,
        resume_path: Optional[Path] = None,
        workers: int = 1,
        sample_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        data = self.loader.load(None if sample_n is not None else num_rows)
        if sample_n is not None:
            data = random.Random(42).sample(data, min(sample_n, len(data)))
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # --- Resume or fresh start ---
        if resume_path is not None:
            output_path, experiment_meta, all_results, stats = _load_checkpoint(resume_path)
            done = len(all_results)
            print(f"\nResuming from: {output_path}")
            print(f"  Already completed: {done}/{len(data)} problems")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"gsm8k_{timestamp}.json"
            experiment_meta = {
                "experiment": "GSM8K Baseline Comparison",
                "model": self.model,
                "num_problems": len(data),
                "baselines": self.baselines_to_run,
                "started_at": timestamp,
            }
            all_results: Dict[str, Dict[str, Any]] = {}
            stats: Dict[str, Dict[str, Any]] = {b: _blank_stats() for b in self.baselines_to_run}

        print(f"\nRunning {len(self.baselines_to_run)} baseline(s) on {len(data)} GSM8K problems")
        print(f"Baselines: {', '.join(self.baselines_to_run)}")
        print(f"Workers:   {workers}")
        print(f"Output:    {output_path}\n")

        lock = threading.Lock()
        _local = threading.local()

        def get_solvers() -> Dict[str, Any]:
            if not hasattr(_local, "solvers"):
                _local.solvers = self._make_solvers()
            return _local.solvers

        def process_one(idx: int, item: Dict[str, Any]) -> None:
            question = item["question"]
            gold = self.loader.gold_answer(item["answer"])
            problem_key = f"gsm8k_{idx}"

            problem_results: Dict[str, Any] = {
                "problem_id": problem_key,
                "question": question,
                "gold_answer": gold,
                "baselines": {},
            }

            solvers = get_solvers()
            for name, solver in solvers.items():
                result = run_one(name, solver, question, gold)
                problem_results["baselines"][name] = result

            with lock:
                for name in solvers:
                    _update_stats(stats[name], problem_results["baselines"][name])
                all_results[problem_key] = problem_results
                checkpoint = {
                    "meta": experiment_meta,
                    "running_stats": {
                        b: {
                            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0,
                            **s,
                        }
                        for b, s in stats.items()
                    },
                    "problems": all_results,
                }
                output_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        pending = [
            (idx, item) for idx, item in enumerate(data)
            if f"gsm8k_{idx}" not in all_results
        ]

        with tqdm(total=len(data), initial=len(data) - len(pending), desc="GSM8K problems") as pbar:
            if workers == 1:
                for idx, item in pending:
                    process_one(idx, item)
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_one, idx, item): idx
                        for idx, item in pending
                    }
                    for future in as_completed(futures):
                        future.result()  # re-raise any solver exception
                        pbar.update(1)

        # Finalize stats
        for s in stats.values():
            _finalize_stats(s)

        final_output = {
            "meta": experiment_meta,
            "summary": stats,
            "problems": all_results,
        }
        output_path.write_text(json.dumps(final_output, indent=2), encoding="utf-8")

        # Print summary table
        self._print_summary(stats)
        print(f"\nFull results saved to: {output_path}")

        return final_output

    def _print_summary(self, stats: Dict[str, Dict[str, Any]]) -> None:
        W = 100
        print("\n" + "=" * W)
        print("GSM8K BASELINE COMPARISON — RESULTS SUMMARY")
        print("=" * W)
        print(f"{'Baseline':<22} {'Accuracy':>10} {'Correct':>9} {'Total':>7} {'Repair Rate':>12} {'Minimality':>11} {'Avg Calls':>11} {'Avg Time(s)':>12}")
        print("-" * W)
        for name, s in stats.items():
            rr = s.get("repair_rate")
            rr_str = f"{rr:>12.1%}" if rr is not None else f"{'N/A':>12}"
            am = s.get("avg_minimality")
            am_str = f"{am:>11.3f}" if am is not None else f"{'N/A':>11}"
            print(
                f"{name:<22} "
                f"{s['accuracy']:>10.1%} "
                f"{s['correct']:>9} "
                f"{s['total']:>7} "
                f"{rr_str} "
                f"{am_str} "
                f"{s['avg_llm_calls']:>11.1f} "
                f"{s['avg_elapsed_seconds']:>12.1f}"
            )
        print("=" * W)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run GSM8K baseline comparison (Self-Refine, Self-Reflection, Tree of Thoughts)"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Number of GSM8K problems to run (default: all 1319)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=ALL_BASELINES,
        default=ALL_BASELINES,
        help="Which baselines to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        # Same model as existing GSM8K experiment to avoid data-contamination from newer models
        default="google/gemini-2.0-flash-lite-001",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--self_refine_max_iter",
        type=int,
        default=4,
        help="Maximum refinement iterations for Self-Refine (default: 4)",
    )
    parser.add_argument(
        "--tot_candidates",
        type=int,
        default=3,
        help="Tree of Thoughts: candidate thoughts per state (k, default: 3)",
    )
    parser.add_argument(
        "--tot_beam",
        type=int,
        default=2,
        help="Tree of Thoughts: beam width between levels (b, default: 2)",
    )
    parser.add_argument(
        "--tot_depth",
        type=int,
        default=3,
        help="Tree of Thoughts: BFS depth levels (T, default: 3)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="FILE",
        help="Resume from a checkpoint. Omit FILE to auto-pick the latest gsm8k_*.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1). Try 8 for ~6-8x speedup.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N problems (seed=42). Overrides --num_rows.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in environment / .env file")

    resume_path: Optional[Path] = None
    if args.resume is not None:
        if args.resume == "latest":
            resume_path = _find_latest_checkpoint("gsm8k")
            if resume_path is None:
                raise RuntimeError("No gsm8k_*.json checkpoint found in results/")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise RuntimeError(f"Checkpoint not found: {resume_path}")

    experiment = GSM8KBaselineExperiment(
        api_key=api_key,
        model=args.model,
        baselines_to_run=args.baselines,
        self_refine_max_iter=args.self_refine_max_iter,
        tot_candidates=args.tot_candidates,
        tot_beam=args.tot_beam,
        tot_depth=args.tot_depth,
    )
    experiment.run(num_rows=args.num_rows, resume_path=resume_path, workers=args.workers, sample_n=args.sample)


if __name__ == "__main__":
    main()
