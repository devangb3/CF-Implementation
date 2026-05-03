"""
MedBrowseComp Baseline Comparison Experiment
=============================================
Compares three iterative-reasoning baselines against a Direct baseline on the
AIM-Harvard MedBrowseComp_CUA medical web-search QA benchmark.

Design: all baselines share the same initial BrowseCompAgent web-search run.
After the agent collects web evidence and produces an initial answer, each
refinement strategy re-processes the gathered context using LLM-only calls
(no additional web searches) to improve the final answer.

This isolates the refinement strategy as the independent variable, making
comparisons against CausalFlow's repair approach directly meaningful.

Baselines:
    Direct          – BrowseCompAgent answer with no refinement
    Self-Refine     – Madaan et al. 2023  (https://arxiv.org/abs/2303.17651)
    Self-Reflection – arXiv 2405.06682    (https://arxiv.org/pdf/2405.06682)
    Tree of Thoughts – Yao et al. 2023   (https://arxiv.org/abs/2305.10601)

Model:       google/gemini-3-flash-preview
Temperature: 0.3 for all agent steps (web browsing + refinement)

Usage:
    python baseline_comparison/experiments/run_medbrowsecomp_baselines.py
    python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --num_examples 50
    python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --baselines direct self_refine
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import random
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from trace_logger import TraceLogger, StepType
from experiments.browsecomp.browsecomp_agent import BrowseCompAgent
from experiments.browsecomp.browsecomp_eval import grade_response
from experiments.browsecomp.web_env import WebEnvironment

RESULTS_DIR = Path(__file__).parent.parent / "results"
ALL_BASELINES = ["direct", "self_refine", "self_reflection", "tree_of_thoughts"]


def _find_latest_checkpoint(prefix: str) -> Optional[Path]:
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_checkpoint(path: Path) -> tuple:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    all_results = data.get("examples", {})
    raw = data.get("running_stats", data.get("summary", {}))
    stats = {
        b: {k: v for k, v in s.items()
            if k not in ("accuracy", "avg_llm_calls", "avg_elapsed_seconds")}
        for b, s in raw.items()
    }
    return path, meta, all_results, stats


MODEL = "google/gemini-3-flash-preview"
AGENT_TEMPERATURE = 0.3   # for all agent + refinement steps
STOP_SIGNAL = "[STOP]"


def _minimality(a: str, b: str) -> float:
    x = a.split()
    y = b.split()
    L = max(len(x), len(y))
    if L == 0:
        return 1.0
    m = sum(1 for k in range(min(len(x), len(y))) if x[k] == y[k])
    return (m / L) * (1 - 0.5 * abs(len(x) - len(y)) / L)


# ---------------------------------------------------------------------------
# Context extraction from trace
# ---------------------------------------------------------------------------

def extract_browsing_context(trace: TraceLogger, max_chars: int = 8000) -> str:
    """
    Distil the key facts the agent gathered from its web-search trace.
    Returns a text block suitable for injecting into refinement prompts.
    """
    parts: List[str] = []

    for step in trace.steps:
        if step.step_type == StepType.TOOL_RESPONSE and step.tool_output:
            snippet = step.tool_output[:1500]
            if len(step.tool_output) > 1500:
                snippet += "... [truncated]"
            parts.append(f"[Evidence]: {snippet}")
        elif step.step_type == StepType.REASONING and step.text:
            parts.append(f"[Agent Reasoning]: {step.text[:400]}")

    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[context truncated]"
    return context


# ---------------------------------------------------------------------------
# Web refinement strategies
# ---------------------------------------------------------------------------

class WebRefinementMixin:
    """Shared LLM refinement helpers for web QA tasks."""

    llm: LLMClient  # set by subclass __init__

    # ---- Self-Refine -------------------------------------------------------

    def _sr_feedback(self, question: str, context: str, answer: str) -> str:
        return self.llm.generate(
            (
                f"You are reviewing an answer to a medical research question.\n\n"
                f"Question: {question}\n\n"
                f"Web evidence gathered:\n{context}\n\n"
                f"Current answer: {answer}\n\n"
                f"Identify any weaknesses, gaps, or factual errors in the answer "
                f"based on the evidence above. Be specific. "
                f"If the answer is fully supported and accurate, write '{STOP_SIGNAL}'.\n\n"
                f"Feedback:"
            ),
            system_message="You are a critical medical fact-checker.",
            temperature=AGENT_TEMPERATURE,
        )

    def _sr_refine(self, question: str, context: str, answer: str, feedback: str) -> str:
        return self.llm.generate(
            (
                f"Revise the answer using the feedback and evidence provided.\n\n"
                f"Question: {question}\n\n"
                f"Web evidence:\n{context}\n\n"
                f"Previous answer: {answer}\n\n"
                f"Feedback: {feedback}\n\n"
                f"Revised answer (concise, exact):"
            ),
            system_message="You are a precise medical researcher.",
            temperature=AGENT_TEMPERATURE,
        )

    # ---- Self-Reflection ---------------------------------------------------

    def _reflect(self, question: str, context: str, wrong_answer: str) -> str:
        return self.llm.generate(
            (
                f"Your initial answer to the medical question below may be wrong.\n\n"
                f"Question: {question}\n\n"
                f"Web evidence:\n{context}\n\n"
                f"Your answer: {wrong_answer}\n\n"
                f"Provide a structured self-reflection:\n"
                f"1. EXPLANATION: Why might the answer be wrong or incomplete?\n"
                f"2. ERROR KEYWORDS: 3-5 concise error type labels.\n"
                f"3. CORRECT REASONING: Step through the evidence to reach the right answer.\n"
                f"4. INSTRUCTIONS: Steps to answer this type of question correctly.\n"
                f"5. GENERAL ADVICE: What to watch for with medical factoid questions.\n\n"
                f"Self-Reflection:"
            ),
            system_message="You are a self-critical medical researcher.",
            temperature=AGENT_TEMPERATURE,
        )

    def _re_answer(self, question: str, context: str, reflection: str) -> str:
        return self.llm.generate(
            (
                f"Using your self-reflection, answer the question as precisely as possible.\n\n"
                f"Question: {question}\n\n"
                f"Web evidence:\n{context}\n\n"
                f"Your self-reflection:\n{reflection}\n\n"
                f"Final answer (exact, concise):"
            ),
            system_message="You are a precise medical researcher.",
            temperature=AGENT_TEMPERATURE,
        )

    # ---- Tree of Thoughts -------------------------------------------------

    def _tot_candidates(self, question: str, context: str, k: int) -> List[str]:
        response = self.llm.generate(
            (
                f"Generate {k} distinct candidate answers to this medical question "
                f"using different interpretations of the evidence.\n\n"
                f"Question: {question}\n\n"
                f"Web evidence:\n{context}\n\n"
                f"List {k} candidate answers, numbered 1-{k}. "
                f"Each should be a concise, exact answer."
            ),
            system_message="You are a thorough medical researcher. Explore different interpretations.",
            temperature=AGENT_TEMPERATURE,
        )
        answers: List[str] = []
        for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)', response, re.DOTALL):
            a = m.group(1).strip()
            if a:
                answers.append(a)
        if len(answers) < k:
            lines = [l.strip() for l in response.splitlines() if l.strip() and not l.strip().startswith('#')]
            seen = set(answers)
            for l in lines:
                if l not in seen:
                    answers.append(l)
                    seen.add(l)
        return answers[:k]

    def _tot_evaluate(self, question: str, context: str, candidates: List[str]) -> List[float]:
        text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(candidates))
        response = self.llm.generate(
            (
                f"Rate each candidate answer for this medical question (1-10).\n\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{context}\n\n"
                f"Candidates:\n{text}\n\n"
                f"Score each 1-10 (10 = best supported by evidence). "
                f"Reply with {len(candidates)} comma-separated numbers only."
            ),
            system_message="You are a precise medical fact evaluator.",
            temperature=AGENT_TEMPERATURE,
        )
        raw = re.findall(r'\b(10|[1-9])\b', response)
        scores = [float(s) for s in raw[:len(candidates)]]
        while len(scores) < len(candidates):
            scores.append(5.0)
        return scores


class DirectWebRefinement(WebRefinementMixin):
    """No refinement — returns the BrowseCompAgent's initial answer."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def refine(
        self, question: str, initial_answer: str, context: str, llm_calls_so_far: int
    ) -> Dict[str, Any]:
        return {
            "final_answer": initial_answer,
            "refinement_llm_calls": 0,
            "total_llm_calls": llm_calls_so_far,
        }


class SelfRefineWeb(WebRefinementMixin):
    def __init__(self, llm: LLMClient, max_iter: int = 3) -> None:
        self.llm = llm
        self.max_iter = max_iter

    def refine(
        self, question: str, initial_answer: str, context: str, llm_calls_so_far: int
    ) -> Dict[str, Any]:
        answer = initial_answer
        refinement_calls = 0

        for _ in range(self.max_iter):
            feedback = self._sr_feedback(question, context, answer)
            refinement_calls += 1
            if STOP_SIGNAL in feedback:
                break
            answer = self._sr_refine(question, context, answer, feedback)
            refinement_calls += 1

        return {
            "final_answer": answer,
            "refinement_llm_calls": refinement_calls,
            "total_llm_calls": llm_calls_so_far + refinement_calls,
        }


class SelfReflectionWeb(WebRefinementMixin):
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def refine(
        self, question: str, initial_answer: str, context: str, llm_calls_so_far: int
    ) -> Dict[str, Any]:
        reflection = self._reflect(question, context, initial_answer)
        final_answer = self._re_answer(question, context, reflection)
        return {
            "final_answer": final_answer,
            "refinement_llm_calls": 2,
            "total_llm_calls": llm_calls_so_far + 2,
            "reflection": reflection,
        }


class TreeOfThoughtsWeb(WebRefinementMixin):
    def __init__(self, llm: LLMClient, num_candidates: int = 3) -> None:
        self.llm = llm
        self.k = num_candidates

    def refine(
        self, question: str, initial_answer: str, context: str, llm_calls_so_far: int
    ) -> Dict[str, Any]:
        candidates = self._tot_candidates(question, context, self.k)
        refinement_calls = 1

        scores = self._tot_evaluate(question, context, candidates)
        refinement_calls += 1

        best_idx = scores.index(max(scores))
        best_answer = candidates[best_idx] if candidates else initial_answer

        return {
            "final_answer": best_answer,
            "refinement_llm_calls": refinement_calls,
            "total_llm_calls": llm_calls_so_far + refinement_calls,
            "candidates": list(zip(candidates, scores)),
        }


# ---------------------------------------------------------------------------
# Statistics helpers
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
    stats["total_llm_calls"] += result.get("total_llm_calls", 0)
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

class MedBrowseCompBaselineExperiment:
    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        search_api_key: Optional[str] = None,
        baselines_to_run: Optional[List[str]] = None,
        max_steps: int = 10,
        self_refine_max_iter: int = 3,
        tot_candidates: int = 3,
    ) -> None:
        self.model = model
        self.baselines_to_run = baselines_to_run or ALL_BASELINES
        self._api_key = api_key
        self._search_api_key = search_api_key
        self._max_steps = max_steps
        self._self_refine_max_iter = self_refine_max_iter
        self._tot_candidates = tot_candidates

    def _make_components(self) -> tuple:
        """Create a fresh, independent agent + grader + refiners (one set per thread)."""
        agent_llm = LLMClient(api_key=self._api_key, model=self.model, temperature=AGENT_TEMPERATURE)
        web_env = WebEnvironment(search_api_key=self._search_api_key)
        agent = BrowseCompAgent(llm_client=agent_llm, web_env=web_env, max_steps=self._max_steps)
        grader_llm = LLMClient(api_key=self._api_key, model=self.model, temperature=AGENT_TEMPERATURE)
        ref_llm = LLMClient(api_key=self._api_key, model=self.model, temperature=AGENT_TEMPERATURE)
        refiners: Dict[str, Any] = {}
        if "direct" in self.baselines_to_run:
            refiners["direct"] = DirectWebRefinement(ref_llm)
        if "self_refine" in self.baselines_to_run:
            refiners["self_refine"] = SelfRefineWeb(ref_llm, max_iter=self._self_refine_max_iter)
        if "self_reflection" in self.baselines_to_run:
            refiners["self_reflection"] = SelfReflectionWeb(ref_llm)
        if "tree_of_thoughts" in self.baselines_to_run:
            refiners["tree_of_thoughts"] = TreeOfThoughtsWeb(ref_llm, num_candidates=self._tot_candidates)
        return agent, grader_llm, refiners

    def _load_data(self, num_examples: Optional[int]) -> List[Dict[str, Any]]:
        dataset = load_dataset("AIM-Harvard/MedBrowseComp_CUA", split="MedBrowseComp_CUA")
        data = [{"question": item["prompt"], "answer": item["gold"]} for item in dataset]
        print(f"Loaded {len(data)} MedBrowseComp examples")
        if num_examples:
            data = random.Random(42).sample(data, min(num_examples, len(data)))
        return data

    def run(
        self,
        num_examples: Optional[int] = None,
        resume_path: Optional[Path] = None,
        workers: int = 1,
        sample_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        data = self._load_data(sample_n if sample_n is not None else num_examples)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if resume_path is not None:
            output_path, meta, all_results, stats = _load_checkpoint(resume_path)
            print(f"\nResuming from: {output_path}")
            print(f"  Already completed: {len(all_results)}/{len(data)} examples")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"medbrowsecomp_{timestamp}.json"
            meta = {
                "experiment": "MedBrowseComp Baseline Comparison",
                "model": self.model,
                "agent_temperature": AGENT_TEMPERATURE,
                "num_examples": len(data),
                "baselines": self.baselines_to_run,
                "started_at": timestamp,
            }
            all_results: Dict[str, Dict[str, Any]] = {}
            stats: Dict[str, Dict[str, Any]] = {b: _blank_stats() for b in self.baselines_to_run}

        print(f"\nRunning {len(self.baselines_to_run)} baseline(s) on {len(data)} MedBrowseComp examples")
        print(f"Baselines: {', '.join(self.baselines_to_run)}")
        print(f"Workers:   {workers}")
        print(f"Output:    {output_path}\n")

        lock = threading.Lock()
        _local = threading.local()

        def get_components() -> tuple:
            if not hasattr(_local, "components"):
                _local.components = self._make_components()
            return _local.components

        def process_one(idx: int, row: Dict[str, Any]) -> None:
            problem_id = f"medbrowsecomp_{idx}"
            question = row["question"]
            gold = row["answer"]

            agent, grader_llm, refiners = get_components()

            # --- Phase 1: web agent ---
            try:
                trace: TraceLogger = agent.solve(
                    problem_id=problem_id, question=question, gold_answer=gold
                )
            except Exception as exc:
                print(f"  Agent error on {problem_id}: {exc}")
                with lock:
                    for b in self.baselines_to_run:
                        _update(stats[b], {"is_correct": False, "error": str(exc), "total_llm_calls": 0, "elapsed_seconds": 0})
                return

            initial_answer = trace.final_answer or ""
            agent_llm_calls = sum(1 for s in trace.steps if s.step_type == StepType.LLM_RESPONSE)
            context = extract_browsing_context(trace)

            initial_correct = grade_response(gold, initial_answer, grader_llm) if initial_answer else False

            problem_results: Dict[str, Any] = {
                "problem_id": problem_id,
                "question": question,
                "gold_answer": gold,
                "initial_answer": initial_answer,
                "initial_correct": initial_correct,
                "agent_llm_calls": agent_llm_calls,
                "baselines": {},
            }

            # --- Phase 2: refinement strategies ---
            for name, refiner in refiners.items():
                start = time.time()
                try:
                    ref_result = refiner.refine(question, initial_answer, context, agent_llm_calls)
                    final_answer = ref_result.get("final_answer", initial_answer)
                    is_correct = grade_response(gold, final_answer, grader_llm) if final_answer else False
                    ref_result["is_correct"] = is_correct
                    ref_result["initial_correct"] = initial_correct
                    ref_result["minimality"] = (
                        _minimality(initial_answer, final_answer)
                        if final_answer != initial_answer else 1.0
                    )
                except Exception as exc:
                    ref_result = {
                        "is_correct": False, "error": str(exc),
                        "total_llm_calls": agent_llm_calls,
                        "initial_correct": initial_correct,
                    }
                ref_result["elapsed_seconds"] = round(time.time() - start, 2)
                problem_results["baselines"][name] = ref_result

            with lock:
                for name in refiners:
                    _update(stats[name], problem_results["baselines"][name])
                all_results[problem_id] = problem_results
                output_path.write_text(
                    json.dumps({"meta": meta, "running_stats": stats, "examples": all_results}, indent=2),
                    encoding="utf-8",
                )

        pending = [
            (idx, row) for idx, row in enumerate(data)
            if f"medbrowsecomp_{idx}" not in all_results
        ]

        with tqdm(total=len(data), initial=len(data) - len(pending), desc="MedBrowseComp") as pbar:
            if workers == 1:
                for idx, row in pending:
                    process_one(idx, row)
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_one, idx, row): idx for idx, row in pending}
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)

        for s in stats.values():
            _finalize(s)

        final = {"meta": meta, "summary": stats, "examples": all_results}
        output_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

        self._print_summary(stats)
        print(f"\nFull results saved to: {output_path}")
        return final

    def _print_summary(self, stats: Dict[str, Dict[str, Any]]) -> None:
        W = 100
        print("\n" + "=" * W)
        print("MedBrowseComp BASELINE COMPARISON — RESULTS SUMMARY")
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
        description="MedBrowseComp baseline comparison"
    )
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples (default: all 484)")
    parser.add_argument("--baselines", nargs="+", choices=ALL_BASELINES, default=ALL_BASELINES)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Max web-browsing steps per problem (default: 10)")
    parser.add_argument("--self_refine_max_iter", type=int, default=3)
    parser.add_argument("--tot_candidates", type=int, default=3)
    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="FILE",
                        help="Resume from checkpoint. Omit FILE to auto-pick latest medbrowsecomp_*.json.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1). Try 4 for web-bound tasks.")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Randomly sample N examples (seed=42). Overrides --num_examples.")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found")
    search_api_key = os.getenv("SERPER_API_KEY")
    if not search_api_key:
        raise RuntimeError("SERPER_API_KEY not found. Web search will fail unless cache is warm.")

    resume_path: Optional[Path] = None
    if args.resume is not None:
        resume_path = _find_latest_checkpoint("medbrowsecomp") if args.resume == "latest" else Path(args.resume)
        if resume_path is None or not resume_path.exists():
            raise RuntimeError(f"Checkpoint not found: {resume_path}")

    experiment = MedBrowseCompBaselineExperiment(
        api_key=api_key,
        model=args.model,
        search_api_key=search_api_key,
        baselines_to_run=args.baselines,
        max_steps=args.max_steps,
        self_refine_max_iter=args.self_refine_max_iter,
        tot_candidates=args.tot_candidates,
    )
    results = experiment.run(num_examples=args.num_examples, resume_path=resume_path, workers=args.workers, sample_n=args.sample)
    print(f"\nBest baseline accuracy: "
          f"{max(s['accuracy'] for s in results['summary'].values()):.1%}")


if __name__ == "__main__":
    main()
