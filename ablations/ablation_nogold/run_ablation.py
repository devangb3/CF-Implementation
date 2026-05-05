"""No-gold repair-prompt ablation runner (Half A).

Replays CausalFlow's repair phase over the same repairable-trace sample as
the minimality ablation, but with ``use_gold_in_prompts=False`` so gold is
never interpolated into the counterfactual-repair prompt. Attribution and
critique paths are unchanged. Results are persisted under
``experiment_name = ablation_nogold_{benchmark}`` so analysis can compute a
paired repair-rate delta against the existing ``ablation_minimality_*``
collections without regenerating the with-gold arm.

Usage:
    python -m ablations.ablation_nogold.run_ablation \\
        --benchmark gsm8k \\
        --source-run-id run_GSM8K_2026-01-27T... \\
        [--limit N] [--dry-run]

Uses the LLM outcome predictor for ``success_predicted`` across all four
benchmarks to keep the judge consistent with the minimality ablation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from mongodb_storage import MongoDBStorage
from trace_logger import StepType, TraceLogger


# Sample sizes must match ablations/ablation_minimality so the two arms
# are paired per problem_id. SealQA/MedBrowse take all repairable (actual
# pools in causal_flow_dups are smaller than the original plan: 26 and 67).
SAMPLE_SIZES: Dict[str, Optional[int]] = {
    "gsm8k": 100,
    "mbpp": 100,
    "sealqa": None,      # all repairable (26 in source run)
    "medbrowse": None,   # all repairable (67 in source run)
}
K_PROPOSALS = 5

INTERVENE_STEP_TYPES = {
    StepType.TOOL_CALL,
    StepType.LLM_RESPONSE,
    StepType.REASONING,
    StepType.TOOL_RESPONSE,
}


def _is_repairable(failing_doc: Dict[str, Any]) -> bool:
    metrics = failing_doc.get("metrics") or {}
    repairs = metrics.get("repairs") or {}
    return (repairs.get("successful_repairs") or 0) > 0


def load_repairable_traces(
    storage: MongoDBStorage, source_run_id: str
) -> List[Dict[str, Any]]:
    run = storage.get_run(source_run_id)
    if not run:
        raise RuntimeError(f"Source run not found: {source_run_id}")
    failing = run.get("failing_traces", []) or []
    repairable = [t for t in failing if _is_repairable(t)]
    print(
        f"Source run {source_run_id}: {len(failing)} failing, "
        f"{len(repairable)} repairable"
    )
    return repairable


def select_sample(
    repairable: List[Dict[str, Any]], sample_size: Optional[int]
) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size >= len(repairable):
        return list(repairable)
    return repairable[:sample_size]


def _build_execution_context(
    trace: TraceLogger, failing_doc: Dict[str, Any]
) -> Dict[str, Any]:
    logs = "\n".join(
        str(step.tool_output)
        for step in trace.steps
        if step.step_type == StepType.TOOL_RESPONSE and step.tool_output
    )
    return {
        "question": failing_doc.get("problem_statement") or trace.problem_statement or "",
        "gold_answer": failing_doc.get("gold_answer") or trace.gold_answer or "",
        "agent_final_answer": failing_doc.get("final_answer") or trace.final_answer or "",
        "logs": logs,
        "problem_id": failing_doc.get("problem_id"),
    }


def _reconstruct_trace(failing_doc: Dict[str, Any]) -> TraceLogger:
    trace_data = failing_doc.get("trace") or {}
    if "problem_statement" not in trace_data:
        trace_data["problem_statement"] = failing_doc.get("problem_statement", "")
    if "gold_answer" not in trace_data:
        trace_data["gold_answer"] = failing_doc.get("gold_answer", "")
    if "final_answer" not in trace_data:
        trace_data["final_answer"] = failing_doc.get("final_answer", "")
    return TraceLogger.from_dict(trace_data)


def run_one(
    causal_flow: CausalFlow,
    failing_doc: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    trace = _reconstruct_trace(failing_doc)
    execution_context = _build_execution_context(trace, failing_doc)
    start = time.time()
    analysis = causal_flow.analyze_trace(
        trace,
        reexecutor=None,
        execution_context=execution_context,
        skip_critique=True,
        intervene_step_types=INTERVENE_STEP_TYPES,
        num_proposals=K_PROPOSALS,
        compute_semantic_minimality=False,
        use_gold_in_prompts=False,
    )
    elapsed_minutes = (time.time() - start) / 60.0
    return analysis, elapsed_minutes


def resolve_model_for_benchmark(benchmark: str) -> str:
    return {
        "gsm8k": "google/gemini-2.0-flash-lite-001",
        "mbpp": "openai/gpt-5-chat",
        "sealqa": "google/gemini-3-flash-preview",
        "medbrowse": "google/gemini-3-flash-preview",
    }[benchmark]


def _run_pass(
    benchmark: str,
    source_docs: List[Dict[str, Any]],
    storage: MongoDBStorage,
    api_key: str,
    model: str,
    run_id: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    stats: Dict[str, int] = {"ok": 0, "errors": 0}
    if dry_run:
        print(f"[dry-run] would process {len(source_docs)} traces")
        return stats

    causal_flow = CausalFlow(api_key=api_key, model=model, mongo_storage=storage)

    for doc in tqdm(source_docs, desc=benchmark):
        try:
            analysis, elapsed = run_one(causal_flow, doc)
            storage.add_failing_trace(
                run_id=run_id,
                trace_data=doc.get("trace") or {},
                problem_id=doc.get("problem_id"),
                problem_statement=doc.get("problem_statement"),
                gold_answer=doc.get("gold_answer"),
                final_answer=doc.get("final_answer"),
                analysis_results=analysis,
                metrics=analysis.get("metrics", {}),
                causal_flow_analysis_time_minutes=elapsed,
                extra_metadata={
                    "ablation_benchmark": benchmark,
                    "ablation_arm": "no_gold",
                    "source_problem_id": doc.get("problem_id"),
                },
            )
            stats["ok"] += 1
        except Exception as exc:
            stats["errors"] += 1
            print(f"  error on problem {doc.get('problem_id')}: {exc}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="No-gold repair-prompt ablation runner (Half A)")
    parser.add_argument(
        "--benchmark", required=True, choices=list(SAMPLE_SIZES.keys())
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help="MongoDB run_id of the original experiment for this benchmark "
             "(same value used for the minimality ablation)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap sample size (smoke tests)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + sample only; skip LLM work and MongoDB writes",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (defaults to the one used in the original run)",
    )
    parser.add_argument(
        "--db-name",
        default="causal_flow_dups",
        help="MongoDB database name (default: causal_flow_dups — the DB that holds "
             "the paired minimality-ablation runs)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in .env file")

    # Override MONGODB_NAME before MongoDBStorage reads it.
    os.environ["MONGODB_NAME"] = args.db_name
    storage = MongoDBStorage()
    repairable = load_repairable_traces(storage, args.source_run_id)

    sample_size = SAMPLE_SIZES[args.benchmark]
    if args.limit is not None:
        sample_size = args.limit
    sample = select_sample(repairable, sample_size)
    print(f"Sample: {len(sample)} traces")

    model = args.model or resolve_model_for_benchmark(args.benchmark)
    experiment_name = f"ablation_nogold_{args.benchmark}"

    if not args.dry_run:
        run_id = storage.create_run(
            experiment_name=experiment_name,
            num_problems=len(sample),
            model_used=model,
        )
    else:
        run_id = "dry-run"

    stats = _run_pass(
        benchmark=args.benchmark,
        source_docs=sample,
        storage=storage,
        api_key=api_key or "",
        model=model,
        run_id=run_id,
        dry_run=args.dry_run,
    )
    print(f"Totals: {stats}")


if __name__ == "__main__":
    main()
