"""
Self-Refine baseline (Madaan et al., 2023)
https://arxiv.org/abs/2303.17651

A single LLM iteratively generates output, produces feedback on its own output,
and refines based on that feedback.  No extra training or supervised data needed.

Algorithm:
    y0 = GENERATE(x)
    for i in 1..max_iter:
        fb = FEEDBACK(x, y_{i-1})
        if STOP in fb: break
        y_i = REFINE(x, y_{i-1}, fb)
    return y_last
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from math_reexecutor import MathReexecutor

STOP_SIGNAL = "[STOP]"


def _minimality(a: str, b: str) -> float:
    x = a.split()
    y = b.split()
    L = max(len(x), len(y))
    if L == 0:
        return 1.0
    m = sum(1 for k in range(min(len(x), len(y))) if x[k] == y[k])
    return (m / L) * (1 - 0.5 * abs(len(x) - len(y)) / L)


class SelfRefine:
    """
    Self-Refine for GSM8K math reasoning.

    Each call to solve() counts:
        - 1 initial generation
        - up to max_iter * 2  (feedback + refine) calls
    """

    def __init__(self, llm_client: LLMClient, max_iter: int = 4):
        self.llm = llm_client
        self.max_iter = max_iter
        self.reexecutor = MathReexecutor()

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def _prompt_generate(self, problem: str) -> str:
        return (
            f"Solve the following math problem step by step, showing every calculation.\n\n"
            f"Problem: {problem}\n\n"
            f"Solution:"
        )

    def _prompt_feedback(self, problem: str, solution: str) -> str:
        return (
            f"You are a rigorous math reviewer.\n\n"
            f"Problem: {problem}\n\n"
            f"Solution:\n{solution}\n\n"
            f"Identify any errors in logic, arithmetic, or problem interpretation above. "
            f"Be specific: quote the wrong part, explain the mistake, and suggest the fix. "
            f"If the solution is completely correct, write '{STOP_SIGNAL}' as the final line.\n\n"
            f"Feedback:"
        )

    def _prompt_refine(self, problem: str, solution: str, feedback: str) -> str:
        return (
            f"Revise the math solution below using the feedback provided. "
            f"Fix every identified error while keeping correct parts.\n\n"
            f"Problem: {problem}\n\n"
            f"Previous Solution:\n{solution}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Revised Solution:"
        )

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _generate(self, problem: str) -> str:
        return self.llm.generate(
            self._prompt_generate(problem),
            system_message="You are a careful math solver. Show all steps clearly.",
        )

    def _feedback(self, problem: str, solution: str) -> str:
        return self.llm.generate(
            self._prompt_feedback(problem, solution),
            system_message="You are a strict math reviewer. Find all errors precisely.",
        )

    def _refine(self, problem: str, solution: str, feedback: str) -> str:
        return self.llm.generate(
            self._prompt_refine(problem, solution, feedback),
            system_message="You are a careful math solver. Fix all identified errors.",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem: str, gold_answer: str) -> Dict[str, Any]:
        """
        Run Self-Refine on one GSM8K problem.

        Returns a dict with:
            final_answer      – extracted numeric answer string
            is_correct        – bool
            initial_answer    – answer after pass 0 (no feedback yet)
            initial_correct   – bool
            num_iterations    – number of refine cycles completed (0 = stopped at feedback)
            llm_calls         – total LLM calls consumed
            history           – list of {iteration, solution, feedback} dicts
        """
        solution = self._generate(problem)
        llm_calls = 1

        initial_answer = self._extract(solution)
        initial_correct = self.reexecutor.compare_answers(initial_answer, gold_answer)

        history: List[Dict[str, Any]] = [
            {"iteration": 0, "solution": solution, "feedback": None}
        ]

        for i in range(1, self.max_iter + 1):
            feedback = self._feedback(problem, solution)
            llm_calls += 1
            history[-1]["feedback"] = feedback

            if STOP_SIGNAL in feedback:
                break

            solution = self._refine(problem, solution, feedback)
            llm_calls += 1
            history.append({"iteration": i, "solution": solution, "feedback": None})

        final_answer = self._extract(solution)
        is_correct = self.reexecutor.compare_answers(final_answer, gold_answer)

        initial_sol = history[0]["solution"] if history else solution
        final_sol = history[-1]["solution"] if history else solution
        return {
            "final_answer": final_answer,
            "is_correct": is_correct,
            "initial_answer": initial_answer,
            "initial_correct": initial_correct,
            "num_iterations": len(history) - 1,
            "llm_calls": llm_calls,
            "history": history,
            "minimality": _minimality(initial_sol, final_sol) if initial_sol != final_sol else 1.0,
        }

    def _extract(self, text: str) -> str:
        num = self.reexecutor.extract_number(text)
        return str(num) if num is not None else text.strip()
