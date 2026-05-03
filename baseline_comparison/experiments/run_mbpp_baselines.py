"""
MBPP Baseline Comparison Experiment
=====================================
Compares three iterative-reasoning baselines against a Direct (single-pass) baseline
on the MBPP Python code-generation benchmark.

Each baseline generates Python code that is tested against unit tests via Docker.
The deterministic Docker executor provides ground-truth pass/fail signals which
are fed back into the refinement loop (unlike GSM8K where we rely on an AST evaluator).

Baselines:
    Direct          – single-pass code generation (lower bound)
    Self-Refine     – Madaan et al. 2023  (https://arxiv.org/abs/2303.17651)
    Self-Reflection – arXiv 2405.06682    (https://arxiv.org/pdf/2405.06682)
    Tree of Thoughts – Yao et al. 2023   (https://arxiv.org/abs/2305.10601)

Model:       openai/gpt-5-chat
Temperature: 0.2 for code generation (deterministic precision)
             0.7 for reasoning/feedback/evaluation steps (exploratory)

Requires Docker to be running for test execution.

Usage:
    python baseline_comparison/experiments/run_mbpp_baselines.py
    python baseline_comparison/experiments/run_mbpp_baselines.py --num_rows 50
    python baseline_comparison/experiments/run_mbpp_baselines.py --baselines direct self_refine
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from experiments.mbpp.mbpp_loader import MBPPDataLoader
from experiments.humaneval.docker_code_executor import DockerCodeExecutor
from experiments.humaneval.humaneval_reexecutor import HumanevalReexecutor

RESULTS_DIR = Path(__file__).parent.parent / "results"
ALL_BASELINES = ["direct", "self_refine", "self_reflection", "tree_of_thoughts"]


def _find_latest_checkpoint(prefix: str) -> Optional[Path]:
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_checkpoint(path: Path) -> tuple:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    all_results = data.get("tasks", {})
    raw = data.get("running_stats", data.get("summary", {}))
    stats = {
        b: {k: v for k, v in s.items()
            if k not in ("accuracy", "avg_llm_calls", "avg_elapsed_seconds")}
        for b, s in raw.items()
    }
    return path, meta, all_results, stats


MODEL = "openai/gpt-5-chat"
CODE_TEMPERATURE = 0.2    # low temperature for code generation (precision)
REASON_TEMPERATURE = 0.7  # higher temperature for feedback / reflection / evaluation
STOP_SIGNAL = "[STOP]"


# ---------------------------------------------------------------------------
# Shared code utilities
# ---------------------------------------------------------------------------

def _minimality(a: str, b: str) -> float:
    x = a.split()
    y = b.split()
    L = max(len(x), len(y))
    if L == 0:
        return 1.0
    m = sum(1 for k in range(min(len(x), len(y))) if x[k] == y[k])
    return (m / L) * (1 - 0.5 * abs(len(x) - len(y)) / L)


def extract_code(response: str) -> str:
    blocks = re.findall(r"```python(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    fallback = re.findall(r"```(.*?)```", response, flags=re.DOTALL)
    if fallback:
        return fallback[-1].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Baseline solver classes
# ---------------------------------------------------------------------------

class DirectMBPP:
    """Single-pass code generation (CoT baseline)."""

    def __init__(self, llm: LLMClient, reexecutor: HumanevalReexecutor) -> None:
        self.llm = llm
        self.reexecutor = reexecutor

    def _generate(self, prompt: str, entry_point: str = "", extra: str = "") -> str:
        full = (
            f"Complete the following Python function. Return ONLY executable Python code.\n\n"
            f"{prompt}"
        )
        if entry_point:
            full += f"\n\nIMPORTANT: The function MUST be named exactly `{entry_point}`."
        if extra:
            full += f"\n\nContext:\n{extra}"
        return self.llm.generate(
            full,
            system_message="You are a precise Python expert. Return only code, no explanations.",
            temperature=CODE_TEMPERATURE,
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._generate(task["prompt"], entry_point=task.get("entry_point", ""))
        code = extract_code(raw)
        success, _, logs = self.reexecutor.run_solution(task["prompt"], code, task["tests"])
        return {
            "is_correct": success,
            "initial_correct": success,
            "minimality": 1.0,
            "llm_calls": 1,
            "final_code": code,
            "test_logs": logs,
        }


class SelfRefineMBPP(DirectMBPP):
    """
    Self-Refine for code generation.
    Provides the actual test failure output as feedback so the LLM can pinpoint errors.
    """

    def __init__(self, llm: LLMClient, reexecutor: HumanevalReexecutor, max_iter: int = 3) -> None:
        super().__init__(llm, reexecutor)
        self.max_iter = max_iter

    def _feedback(self, prompt: str, code: str, test_logs: str) -> str:
        return self.llm.generate(
            (
                f"Review this Python solution for the task below and identify exactly "
                f"what causes the test failures.\n\n"
                f"Task:\n{prompt}\n\n"
                f"Solution:\n```python\n{code}\n```\n\n"
                f"Test failure output:\n{test_logs}\n\n"
                f"Provide specific, actionable feedback on what to fix. "
                f"If all tests pass (no errors shown), write '{STOP_SIGNAL}' at the end.\n\n"
                f"Feedback:"
            ),
            system_message="You are a rigorous Python code reviewer. Be specific about bugs.",
            temperature=REASON_TEMPERATURE,
        )

    def _refine(self, prompt: str, code: str, feedback: str) -> str:
        raw = self.llm.generate(
            (
                f"Fix the Python function based on the feedback below. "
                f"Return ONLY corrected Python code.\n\n"
                f"Task:\n{prompt}\n\n"
                f"Previous code:\n```python\n{code}\n```\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Corrected code:"
            ),
            system_message="You are a precise Python expert. Return only code.",
            temperature=CODE_TEMPERATURE,
        )
        return extract_code(raw)

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._generate(task["prompt"], entry_point=task.get("entry_point", ""))
        code = extract_code(raw)
        llm_calls = 1
        history: List[Dict[str, Any]] = []

        for i in range(self.max_iter):
            success, _, logs = self.reexecutor.run_solution(task["prompt"], code, task["tests"])
            history.append({"iteration": i, "code": code, "success": success, "logs": logs})
            if success:
                break
            feedback = self._feedback(task["prompt"], code, logs)
            llm_calls += 1
            if STOP_SIGNAL in feedback:
                break
            code = self._refine(task["prompt"], code, feedback)
            llm_calls += 1

        success, _, logs = self.reexecutor.run_solution(task["prompt"], code, task["tests"])
        initial_code = history[0]["code"] if history else code
        return {
            "is_correct": success,
            "initial_correct": history[0]["success"] if history else None,
            "minimality": _minimality(initial_code, code) if initial_code != code else 1.0,
            "llm_calls": llm_calls,
            "num_iterations": len(history),
            "final_code": code,
            "test_logs": logs,
            "history": history,
        }


class SelfReflectionMBPP(DirectMBPP):
    """
    Self-Reflection for code generation.
    Generates a structured 5-part reflection on failing tests, then re-generates.
    """

    def _reflect(self, prompt: str, bad_code: str, test_logs: str) -> str:
        return self.llm.generate(
            (
                f"Your Python solution failed its tests. Provide a structured self-reflection.\n\n"
                f"Task:\n{prompt}\n\n"
                f"Failing code:\n```python\n{bad_code}\n```\n\n"
                f"Test failures:\n{test_logs}\n\n"
                f"Self-reflection (all five parts required):\n"
                f"1. EXPLANATION: Why is the code wrong?\n"
                f"2. ERROR KEYWORDS: 3-5 concise keywords (e.g. 'off-by-one', 'wrong formula').\n"
                f"3. CORRECT APPROACH: Outline the correct algorithm step by step.\n"
                f"4. INSTRUCTIONS: Precise steps to implement it correctly.\n"
                f"5. GENERAL ADVICE: What principle to remember to avoid this mistake?\n\n"
                f"Self-Reflection:"
            ),
            system_message="You are a self-critical Python expert. Diagnose errors honestly.",
            temperature=REASON_TEMPERATURE,
        )

    def _re_generate(self, prompt: str, reflection: str, entry_point: str = "") -> str:
        ep_hint = f"\n\nIMPORTANT: The function MUST be named exactly `{entry_point}`." if entry_point else ""
        raw = self.llm.generate(
            (
                f"Using your self-reflection as a guide, write a correct Python implementation.\n\n"
                f"Task:\n{prompt}{ep_hint}\n\n"
                f"Self-Reflection:\n{reflection}\n\n"
                f"Correct implementation (Python code only):"
            ),
            system_message="You are a precise Python expert. Return only code.",
            temperature=CODE_TEMPERATURE,
        )
        return extract_code(raw)

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        entry_point = task.get("entry_point", "")
        raw = self._generate(task["prompt"], entry_point=entry_point)
        initial_code = extract_code(raw)
        llm_calls = 1

        success, _, logs = self.reexecutor.run_solution(task["prompt"], initial_code, task["tests"])
        if success:
            return {
                "is_correct": True,
                "initial_correct": True,
                "minimality": 1.0,
                "llm_calls": llm_calls,
                "reflection_used": False,
                "final_code": initial_code,
                "test_logs": logs,
            }

        reflection = self._reflect(task["prompt"], initial_code, logs)
        llm_calls += 1
        final_code = self._re_generate(task["prompt"], reflection, entry_point=entry_point)
        llm_calls += 1

        success, _, final_logs = self.reexecutor.run_solution(task["prompt"], final_code, task["tests"])
        return {
            "is_correct": success,
            "initial_correct": False,
            "minimality": _minimality(initial_code, final_code),
            "llm_calls": llm_calls,
            "reflection_used": True,
            "final_code": final_code,
            "test_logs": final_logs,
            "initial_code": initial_code,
            "initial_logs": logs,
            "reflection": reflection,
        }


class TreeOfThoughtsMBPP(DirectMBPP):
    """
    Tree of Thoughts for code generation.

    BFS over code strategies:
        Level 1 – propose k high-level algorithmic approaches (evaluate → keep top b)
        Level 2 – for each surviving approach, generate full implementation
        Test all implementations; return first passing or the one with fewest failures.
    """

    def __init__(
        self,
        llm: LLMClient,
        reexecutor: HumanevalReexecutor,
        num_candidates: int = 3,
        beam_width: int = 2,
    ) -> None:
        super().__init__(llm, reexecutor)
        self.k = num_candidates
        self.b = beam_width

    def _propose_approaches(self, prompt: str) -> List[str]:
        response = self.llm.generate(
            (
                f"Propose {self.k} distinct algorithmic strategies to implement this Python function.\n\n"
                f"Task:\n{prompt}\n\n"
                f"For each strategy describe the algorithm in 1-2 sentences (no code). "
                f"Number them 1, 2, {self.k}."
            ),
            system_message="You are a creative algorithmist. Propose diverse approaches.",
            temperature=REASON_TEMPERATURE,
        )
        approaches: List[str] = []
        for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)', response, re.DOTALL):
            a = m.group(1).strip()
            if a:
                approaches.append(a)
        if len(approaches) < self.k:
            lines = [l.strip() for l in response.splitlines() if l.strip()]
            seen = set(approaches)
            for l in lines:
                if l not in seen:
                    approaches.append(l)
                    seen.add(l)
        return approaches[:self.k]

    def _evaluate_approaches(self, prompt: str, approaches: List[str]) -> List[float]:
        text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(approaches))
        response = self.llm.generate(
            (
                f"Rate each strategy for implementing this Python function (1-10).\n\n"
                f"Task:\n{prompt}\n\n"
                f"Strategies:\n{text}\n\n"
                f"Score each 1-10 (10 = most correct and efficient). "
                f"Reply with only {len(approaches)} comma-separated numbers, e.g. '8, 6, 7'."
            ),
            system_message="You are a precise Python algorithm evaluator. Rate strategies 1-10.",
            temperature=REASON_TEMPERATURE,
        )
        raw = re.findall(r'\b(10|[1-9])\b', response)
        scores = [float(s) for s in raw[:len(approaches)]]
        while len(scores) < len(approaches):
            scores.append(5.0)
        return scores

    def _implement(self, prompt: str, approach: str, entry_point: str = "") -> str:
        ep_hint = f"\n\nIMPORTANT: The function MUST be named exactly `{entry_point}`." if entry_point else ""
        raw = self.llm.generate(
            (
                f"Implement this Python function using the algorithmic approach described below.\n\n"
                f"Task:\n{prompt}{ep_hint}\n\n"
                f"Approach: {approach}\n\n"
                f"Return ONLY Python code."
            ),
            system_message="You are a precise Python expert. Return only code.",
            temperature=CODE_TEMPERATURE,
        )
        return extract_code(raw)

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        approaches = self._propose_approaches(task["prompt"])
        llm_calls = 1

        scores = self._evaluate_approaches(task["prompt"], approaches)
        llm_calls += 1

        # Keep top-b approaches
        ranked = sorted(zip(approaches, scores), key=lambda x: x[1], reverse=True)
        top_approaches = ranked[:self.b]

        beam_results: List[Dict[str, Any]] = []
        best_code = ""
        best_success = False
        best_logs = ""

        for approach, score in top_approaches:
            code = self._implement(task["prompt"], approach, entry_point=task.get("entry_point", ""))
            llm_calls += 1
            success, _, logs = self.reexecutor.run_solution(task["prompt"], code, task["tests"])
            beam_results.append({"approach": approach, "score": score, "success": success, "logs": logs})
            # Prefer first passing solution, or keep last if none pass
            if success and not best_success:
                best_success = True
                best_code = code
                best_logs = logs
            elif not best_success:
                best_code = code
                best_logs = logs

        return {
            "is_correct": best_success,
            "minimality": None,
            "llm_calls": llm_calls,
            "final_code": best_code,
            "test_logs": best_logs,
            "beam_results": beam_results,
        }


# ---------------------------------------------------------------------------
# Statistics helpers (same pattern as GSM8K baseline)
# ---------------------------------------------------------------------------

def _blank_stats() -> Dict[str, Any]:
    return {
        "total": 0, "correct": 0, "errors": 0,
        "total_llm_calls": 0, "total_elapsed_seconds": 0.0,
        "initially_correct": 0, "repaired": 0,
        "total_minimality": 0.0, "minimality_count": 0,
    }


def _update(stats: Dict[str, Any], result: Dict[str, Any]) -> None:
    stats["total"] += 1
    is_correct = result.get("is_correct", False)
    stats["correct"] += int(is_correct)
    stats["errors"] += int("error" in result)
    stats["total_llm_calls"] += result.get("llm_calls", 0)
    stats["total_elapsed_seconds"] += result.get("elapsed_seconds", 0.0)
    initial_correct = result.get("initial_correct")
    if initial_correct is not None:
        stats["initially_correct"] = stats.get("initially_correct", 0) + int(initial_correct)
        if not initial_correct and is_correct:
            stats["repaired"] = stats.get("repaired", 0) + 1
    minimality = result.get("minimality")
    if minimality is not None:
        stats["total_minimality"] = stats.get("total_minimality", 0.0) + minimality
        stats["minimality_count"] = stats.get("minimality_count", 0) + 1


def _finalize(stats: Dict[str, Any]) -> None:
    n = stats["total"]
    stats["accuracy"] = stats["correct"] / n if n > 0 else 0.0
    stats["avg_llm_calls"] = stats["total_llm_calls"] / n if n > 0 else 0.0
    stats["avg_elapsed_seconds"] = stats["total_elapsed_seconds"] / n if n > 0 else 0.0
    initially_wrong = n - stats.get("initially_correct", 0)
    stats["repair_rate"] = stats.get("repaired", 0) / initially_wrong if initially_wrong > 0 else None
    mc = stats.get("minimality_count", 0)
    stats["avg_minimality"] = stats.get("total_minimality", 0.0) / mc if mc > 0 else None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

class MBPPBaselineExperiment:
    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        baselines_to_run: Optional[List[str]] = None,
        self_refine_max_iter: int = 3,
        tot_candidates: int = 3,
        tot_beam: int = 2,
    ) -> None:
        self.model = model
        self.baselines_to_run = baselines_to_run or ALL_BASELINES
        self.loader = MBPPDataLoader(dataset_name="mbpp", split="all")
        self._api_key = api_key
        self._self_refine_max_iter = self_refine_max_iter
        self._tot_candidates = tot_candidates
        self._tot_beam = tot_beam

    def _make_solvers(self) -> Dict[str, Any]:
        """Create a fresh, independent set of solvers (one per thread)."""
        executor = DockerCodeExecutor()
        reexecutor = HumanevalReexecutor(executor)
        llm = LLMClient(api_key=self._api_key, model=self.model, temperature=CODE_TEMPERATURE)
        solvers: Dict[str, Any] = {}
        if "direct" in self.baselines_to_run:
            solvers["direct"] = DirectMBPP(llm, reexecutor)
        if "self_refine" in self.baselines_to_run:
            solvers["self_refine"] = SelfRefineMBPP(llm, reexecutor, max_iter=self._self_refine_max_iter)
        if "self_reflection" in self.baselines_to_run:
            solvers["self_reflection"] = SelfReflectionMBPP(llm, reexecutor)
        if "tree_of_thoughts" in self.baselines_to_run:
            solvers["tree_of_thoughts"] = TreeOfThoughtsMBPP(
                llm, reexecutor, num_candidates=self._tot_candidates, beam_width=self._tot_beam
            )
        return solvers

    def run(
        self,
        num_rows: Optional[int] = None,
        resume_path: Optional[Path] = None,
        workers: int = 1,
        sample_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        tasks = self.loader.load_data(None if sample_n is not None else num_rows)
        if sample_n is not None:
            tasks = random.Random(42).sample(tasks, min(sample_n, len(tasks)))
        print(f"Loaded {len(tasks)} MBPP tasks")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if resume_path is not None:
            output_path, meta, all_results, stats = _load_checkpoint(resume_path)
            print(f"\nResuming from: {output_path}")
            print(f"  Already completed: {len(all_results)}/{len(tasks)} tasks")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"mbpp_{timestamp}.json"
            meta = {
                "experiment": "MBPP Baseline Comparison",
                "model": self.model,
                "code_temperature": CODE_TEMPERATURE,
                "reasoning_temperature": REASON_TEMPERATURE,
                "num_tasks": len(tasks),
                "baselines": self.baselines_to_run,
                "started_at": timestamp,
            }
            all_results: Dict[str, Dict[str, Any]] = {}
            stats: Dict[str, Dict[str, Any]] = {b: _blank_stats() for b in self.baselines_to_run}

        print(f"\nRunning {len(self.baselines_to_run)} baseline(s) on {len(tasks)} MBPP tasks")
        print(f"Baselines: {', '.join(self.baselines_to_run)}")
        print(f"Workers:   {workers}")
        print(f"Output:    {output_path}\n")

        def cleanup(test: str) -> str:
            test = test.strip()
            m = re.search(r"def check\(candidate\):", test)
            return test[m.start():] if m else test

        lock = threading.Lock()
        _local = threading.local()

        def get_solvers() -> Dict[str, Any]:
            if not hasattr(_local, "solvers"):
                _local.solvers = self._make_solvers()
            return _local.solvers

        def process_one(task: Dict[str, Any]) -> None:
            task_id = task["task_id"]
            clean_tests = cleanup(task["tests"])
            task_for_solvers = {**task, "tests": clean_tests}

            problem_results: Dict[str, Any] = {
                "task_id": task_id,
                "prompt": task["prompt"],
                "entry_point": task["entry_point"],
                "baselines": {},
            }

            solvers = get_solvers()
            for name, solver in solvers.items():
                start = time.time()
                try:
                    result = solver.solve(task_for_solvers)
                except Exception as exc:
                    result = {"is_correct": False, "error": str(exc), "llm_calls": 0}
                result["elapsed_seconds"] = round(time.time() - start, 2)
                problem_results["baselines"][name] = result

            with lock:
                for name in solvers:
                    _update(stats[name], problem_results["baselines"][name])
                all_results[task_id] = problem_results
                output_path.write_text(
                    json.dumps({"meta": meta, "running_stats": stats, "tasks": all_results}, indent=2),
                    encoding="utf-8",
                )

        pending = [t for t in tasks if t["task_id"] not in all_results]

        with tqdm(total=len(tasks), initial=len(tasks) - len(pending), desc="MBPP tasks") as pbar:
            if workers == 1:
                for task in pending:
                    process_one(task)
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_one, task): task["task_id"] for task in pending}
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)

        for s in stats.values():
            _finalize(s)

        final = {"meta": meta, "summary": stats, "tasks": all_results}
        output_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

        self._print_summary(stats)
        print(f"\nFull results saved to: {output_path}")
        return final

    def _print_summary(self, stats: Dict[str, Dict[str, Any]]) -> None:
        W = 100
        print("\n" + "=" * W)
        print("MBPP BASELINE COMPARISON — RESULTS SUMMARY")
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
        description="MBPP baseline comparison (Self-Refine, Self-Reflection, Tree of Thoughts)"
    )
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Number of MBPP tasks (default: all ~947)")
    parser.add_argument("--baselines", nargs="+", choices=ALL_BASELINES, default=ALL_BASELINES,
                        help="Which baselines to run (default: all)")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="OpenRouter model identifier")
    parser.add_argument("--self_refine_max_iter", type=int, default=3,
                        help="Self-Refine: max refinement iterations (default: 3)")
    parser.add_argument("--tot_candidates", type=int, default=3,
                        help="Tree of Thoughts: candidate approaches (k, default: 3)")
    parser.add_argument("--tot_beam", type=int, default=2,
                        help="Tree of Thoughts: beam width (b, default: 2)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="FILE",
                        help="Resume from checkpoint. Omit FILE to auto-pick latest mbpp_*.json.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1). Try 8 for ~6-8x speedup.")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Randomly sample N tasks (seed=42). Overrides --num_rows.")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in environment / .env file")

    resume_path: Optional[Path] = None
    if args.resume is not None:
        resume_path = _find_latest_checkpoint("mbpp") if args.resume == "latest" else Path(args.resume)
        if resume_path is None or not resume_path.exists():
            raise RuntimeError(f"Checkpoint not found: {resume_path}")

    experiment = MBPPBaselineExperiment(
        api_key=api_key,
        model=args.model,
        baselines_to_run=args.baselines,
        self_refine_max_iter=args.self_refine_max_iter,
        tot_candidates=args.tot_candidates,
        tot_beam=args.tot_beam,
    )
    experiment.run(num_rows=args.num_rows, resume_path=resume_path, workers=args.workers, sample_n=args.sample)


if __name__ == "__main__":
    main()
