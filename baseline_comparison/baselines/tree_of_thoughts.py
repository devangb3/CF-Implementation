"""
Tree of Thoughts baseline (Yao et al., 2023)
https://arxiv.org/abs/2305.10601

Generalizes Chain-of-Thought by maintaining a tree of intermediate reasoning states.
Uses BFS to explore multiple partial solutions simultaneously, pruning unlikely branches.

BFS algorithm (adapted for GSM8K math reasoning):
    states = [(problem_context, score=0)]
    for depth in 1..T:
        candidates = []
        for each state:
            thoughts = PROPOSE(state, k)      # k candidate next steps
            scores   = EVALUATE(state, thoughts)  # batch scoring
            candidates.extend(zip(thoughts, scores))
        states = top_b(candidates)            # keep b best
    for each surviving state:
        answer = GENERATE_ANSWER(state)
    return answer from highest-scored state

Paper results (for reference):
    Game of 24:  IO 7.3%, CoT 4.0%, CoT-SC 9.0%, ToT 74%
    Mini Crosswords: IO 14%, CoT 15.6%, ToT 60%

Adapted for GSM8K:
    - "Thoughts" = intermediate calculation/reasoning steps
    - T=3 depth levels (most GSM8K problems need 3-5 steps)
    - k=3 candidate thoughts per state per level
    - b=2 beam width (states kept between levels)
    - Evaluation: LLM scores each thought batch 1-10
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from math_reexecutor import MathReexecutor

# Type alias for a BFS state: (reasoning_chain, cumulative_score)
State = Tuple[str, float]


class TreeOfThoughts:
    """
    Tree of Thoughts (BFS variant) for GSM8K math reasoning.

    LLM call budget per problem (with k=3, b=2, T=3):
        Depth 1: 1 propose + 1 batch-eval  = 2
        Depth 2: b*1 propose + b*1 eval    = 4
        Depth 3: b*1 propose + b*1 eval    = 4
        Final:   b answer calls             = 2
        Total:   ~12 calls per problem
    """

    def __init__(
        self,
        llm_client: LLMClient,
        num_candidates: int = 3,  # k: thoughts proposed per state
        beam_width: int = 2,      # b: states kept between levels
        max_depth: int = 3,       # T: BFS levels
    ):
        self.llm = llm_client
        self.k = num_candidates
        self.b = beam_width
        self.T = max_depth
        self.reexecutor = MathReexecutor()

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def _prompt_propose(self, problem: str, chain: str) -> str:
        chain_text = chain if chain else "No steps taken yet."
        return (
            f"You are solving a math problem step by step.\n\n"
            f"Problem: {problem}\n\n"
            f"Work done so far:\n{chain_text}\n\n"
            f"Propose exactly {self.k} different candidate next reasoning steps. "
            f"Each should make distinct progress toward the answer. "
            f"Format: number each step on its own line as '1. ...', '2. ...', etc."
        )

    def _prompt_evaluate(self, problem: str, chain: str, thoughts: List[str]) -> str:
        chain_text = chain if chain else "No steps taken yet."
        thoughts_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(thoughts))
        return (
            f"Rate each candidate next step for solving this math problem.\n\n"
            f"Problem: {problem}\n\n"
            f"Work so far:\n{chain_text}\n\n"
            f"Candidate next steps:\n{thoughts_text}\n\n"
            f"Score each step 1-10 (10 = definitely correct and useful, "
            f"1 = wrong or unhelpful). "
            f"Reply with ONLY the {len(thoughts)} scores separated by commas, "
            f"e.g.: '8, 5, 3'"
        )

    def _prompt_final_answer(self, problem: str, chain: str) -> str:
        return (
            f"Complete the solution to this math problem.\n\n"
            f"Problem: {problem}\n\n"
            f"Reasoning chain:\n{chain}\n\n"
            f"Based on this reasoning, state the final numerical answer clearly."
        )

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _propose(self, problem: str, chain: str) -> List[str]:
        response = self.llm.generate(
            self._prompt_propose(problem, chain),
            system_message="You are a creative math solver. Propose distinct next steps.",
        )
        # Parse numbered list; fall back to non-empty lines
        thoughts: List[str] = []
        for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)', response, re.DOTALL):
            t = m.group(1).strip()
            if t:
                thoughts.append(t)
        if len(thoughts) < self.k:
            lines = [l.strip() for l in response.splitlines() if l.strip()]
            # avoid duplicates
            seen = set(thoughts)
            for l in lines:
                if l not in seen:
                    thoughts.append(l)
                    seen.add(l)
        return thoughts[:self.k]

    def _evaluate(self, problem: str, chain: str, thoughts: List[str]) -> List[float]:
        if not thoughts:
            return []
        response = self.llm.generate(
            self._prompt_evaluate(problem, chain, thoughts),
            system_message="You are a precise math evaluator. Rate steps 1-10.",
        )
        raw_scores = re.findall(r'\b(10|[1-9])\b', response)
        scores = [float(s) for s in raw_scores[:len(thoughts)]]
        # Pad missing scores with 5 (neutral)
        while len(scores) < len(thoughts):
            scores.append(5.0)
        return scores

    def _generate_answer(self, problem: str, chain: str) -> str:
        return self.llm.generate(
            self._prompt_final_answer(problem, chain),
            system_message="You are a math solver. State the final answer from the reasoning.",
        )

    # ------------------------------------------------------------------
    # BFS search
    # ------------------------------------------------------------------

    def _bfs(self, problem: str) -> Tuple[List[State], int]:
        """
        Run BFS over thought space.
        Returns (surviving_states, total_llm_calls).
        """
        states: List[State] = [("", 0.0)]
        llm_calls = 0

        for depth in range(self.T):
            candidates: List[State] = []

            for chain, chain_score in states:
                thoughts = self._propose(problem, chain)
                llm_calls += 1
                scores = self._evaluate(problem, chain, thoughts)
                llm_calls += 1

                for thought, score in zip(thoughts, scores):
                    step_label = f"Step {depth + 1}: {thought}"
                    new_chain = f"{chain}\n{step_label}".strip()
                    candidates.append((new_chain, chain_score + score))

            # Keep top-b states by cumulative score
            candidates.sort(key=lambda s: s[1], reverse=True)
            states = candidates[:self.b]

        return states, llm_calls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem: str, gold_answer: str) -> Dict[str, Any]:
        """
        Run Tree of Thoughts (BFS) on one GSM8K problem.

        Returns a dict with:
            final_answer           – extracted numeric answer string
            is_correct             – bool
            best_chain             – reasoning chain of the selected answer
            beam_results           – list of {chain, score, answer, correct} for all beams
            num_thoughts_evaluated – total thoughts scored during BFS
            llm_calls              – total LLM calls consumed
        """
        surviving_states, llm_calls = self._bfs(problem)

        beam_results: List[Dict[str, Any]] = []
        best_answer: Optional[str] = None
        best_chain: str = ""
        best_score: float = -1.0

        for chain, score in surviving_states:
            answer_text = self._generate_answer(problem, chain)
            llm_calls += 1

            answer = self._extract(answer_text)
            correct = self.reexecutor.compare_answers(answer, gold_answer)

            beam_results.append({
                "chain": chain,
                "score": score,
                "answer": answer,
                "correct": correct,
            })

            if score > best_score:
                best_score = score
                best_answer = answer
                best_chain = chain

        final_answer = best_answer or ""
        is_correct = self.reexecutor.compare_answers(final_answer, gold_answer)

        num_thoughts = self.k + self.b * self.k * (self.T - 1)

        return {
            "final_answer": final_answer,
            "is_correct": is_correct,
            "best_chain": best_chain,
            "best_score": best_score,
            "beam_results": beam_results,
            "num_thoughts_evaluated": num_thoughts,
            "minimality": None,
            "llm_calls": llm_calls,
        }

    def _extract(self, text: str) -> str:
        num = self.reexecutor.extract_number(text)
        return str(num) if num is not None else text.strip()
