"""
Self-Reflection baseline (arXiv 2405.06682)
https://arxiv.org/pdf/2405.06682

Models generate a structured self-reflection on why their initial answer was wrong,
then re-answer the question with that reflection injected into the prompt.
The reflection is a single-pass operation (not iterative).

Reflection components (Figure 7 in the paper):
    1. Why the previous answer was wrong
    2. Specific error keywords
    3. Correct step-by-step reasoning
    4. Detailed instructions for this problem type
    5. General advice to avoid the mistake in future

Re-answer uses the reflection WITHOUT revealing the ground-truth label.

Algorithm:
    y0 = ANSWER(x)
    if y0 is incorrect:
        r  = REFLECT(x, y0)          # structured 5-part reflection
        y1 = RE_ANSWER(x, r)         # re-answer guided by reflection
        return y1
    return y0
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from math_reexecutor import MathReexecutor


def _minimality(a: str, b: str) -> float:
    x = a.split()
    y = b.split()
    L = max(len(x), len(y))
    if L == 0:
        return 1.0
    m = sum(1 for k in range(min(len(x), len(y))) if x[k] == y[k])
    return (m / L) * (1 - 0.5 * abs(len(x) - len(y)) / L)


class SelfReflection:
    """
    Self-Reflection for GSM8K math reasoning.

    Each call to solve() counts:
        - 1 initial answer call
        - (if wrong) 1 reflection call + 1 re-answer call  →  3 calls total
        - (if right) 1 call total
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.reexecutor = MathReexecutor()

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def _prompt_initial(self, problem: str) -> str:
        return (
            f"Solve the following math problem step by step.\n\n"
            f"Problem: {problem}\n\n"
            f"Solution:"
        )

    def _prompt_reflect(self, problem: str, wrong_solution: str) -> str:
        return (
            f"Your previous answer to the math problem below was incorrect.\n\n"
            f"Problem: {problem}\n\n"
            f"Incorrect Solution:\n{wrong_solution}\n\n"
            f"Provide a structured self-reflection with all five parts:\n\n"
            f"1. EXPLANATION: Why was the previous solution wrong? "
            f"Identify the exact reasoning or calculation error.\n"
            f"2. ERROR KEYWORDS: List 3-5 concise keywords describing the error type "
            f"(e.g., 'arithmetic error', 'wrong formula', 'misread problem').\n"
            f"3. CORRECT SOLUTION: Solve the problem correctly step by step.\n"
            f"4. INSTRUCTIONS: Write step-by-step instructions for solving this type "
            f"of problem correctly in future.\n"
            f"5. GENERAL ADVICE: What general principle should be remembered to avoid "
            f"this kind of mistake?\n\n"
            f"Self-Reflection:"
        )

    def _prompt_re_answer(self, problem: str, reflection: str) -> str:
        return (
            f"Using your self-reflection as a guide, solve the math problem correctly.\n\n"
            f"Problem: {problem}\n\n"
            f"Your Self-Reflection:\n{reflection}\n\n"
            f"Now provide the correct solution, following your own instructions above:\n\n"
            f"Final Solution:"
        )

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _answer(self, problem: str) -> str:
        return self.llm.generate(
            self._prompt_initial(problem),
            system_message="You are a careful math solver. Show all steps clearly.",
        )

    def _reflect(self, problem: str, wrong_solution: str) -> str:
        return self.llm.generate(
            self._prompt_reflect(problem, wrong_solution),
            system_message=(
                "You are a self-critical math solver. "
                "Diagnose your errors honestly and thoroughly."
            ),
        )

    def _re_answer(self, problem: str, reflection: str) -> str:
        return self.llm.generate(
            self._prompt_re_answer(problem, reflection),
            system_message="You are a careful math solver. Apply your reflection to get it right.",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem: str, gold_answer: str) -> Dict[str, Any]:
        """
        Run Self-Reflection on one GSM8K problem.

        Returns a dict with:
            final_answer      – extracted numeric answer string
            is_correct        – bool
            initial_answer    – answer before reflection
            initial_correct   – bool
            reflection_used   – bool (False if initial was correct)
            reflection        – the reflection text (None if not used)
            llm_calls         – total LLM calls consumed
        """
        initial_solution = self._answer(problem)
        llm_calls = 1

        initial_answer = self._extract(initial_solution)
        initial_correct = self.reexecutor.compare_answers(initial_answer, gold_answer)

        if initial_correct:
            return {
                "final_answer": initial_answer,
                "is_correct": True,
                "initial_answer": initial_answer,
                "initial_correct": True,
                "reflection_used": False,
                "reflection": None,
                "initial_solution": initial_solution,
                "final_solution": initial_solution,
                "minimality": 1.0,
                "llm_calls": llm_calls,
            }

        reflection = self._reflect(problem, initial_solution)
        llm_calls += 1

        final_solution = self._re_answer(problem, reflection)
        llm_calls += 1

        final_answer = self._extract(final_solution)
        is_correct = self.reexecutor.compare_answers(final_answer, gold_answer)

        return {
            "final_answer": final_answer,
            "is_correct": is_correct,
            "initial_answer": initial_answer,
            "initial_correct": False,
            "reflection_used": True,
            "reflection": reflection,
            "initial_solution": initial_solution,
            "final_solution": final_solution,
            "minimality": _minimality(initial_solution, final_solution),
            "llm_calls": llm_calls,
        }

    def _extract(self, text: str) -> str:
        num = self.reexecutor.extract_number(text)
        return str(num) if num is not None else text.strip()
