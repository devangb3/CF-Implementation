"""Build the reviewer CSV from a sample manifest.

Reads the manifest produced by ``sample_for_review.py``, re-fetches the run
doc from MongoDB, extracts the problem/gold/original + repaired artifacts
for each sampled proposal, and writes a CSV the human reviewer fills in.

One row per (trace, step, proposal). Columns match the labeling conventions
described in ``README.md``.

Usage:
    python -m ablations.ablation_judge.build_review_csv --dataset sealqa
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient


REVIEW_DIR = Path(__file__).parent / "review"
MAX_TEXT_CHARS = 2000  # truncate per-field to keep the CSV navigable


FIELDNAMES = [
    "dataset",
    "run_id",
    "problem_id",
    "step_id",
    "proposal_idx",
    "judge_verdict",          # "success" or "failure" (from success_predicted)
    "problem_statement",
    "gold_answer",
    "original_final_answer",
    "repaired_final_answer",  # from repaired_trace if stored; else blank
    "original_step_text",
    "repaired_step_text",
    "downstream_context",     # short summary of descendant steps, if available
    "human_label",            # filled in: correct / incorrect / unclear
    "human_notes",
]


def get_db(db_name: str) -> Any:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in .env")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name]


def _truncate(s: Any, n: int = MAX_TEXT_CHARS) -> str:
    if s is None:
        return ""
    text = s if isinstance(s, str) else json.dumps(s, default=str)
    return text if len(text) <= n else text[:n] + " …[truncated]"


def _step_text(step: Dict[str, Any]) -> str:
    """Flatten the salient fields of a step into one readable string."""
    if not isinstance(step, dict):
        return ""
    parts: List[str] = []
    if step.get("step_type"):
        parts.append(f"[{step['step_type']}]")
    if step.get("tool_name"):
        parts.append(f"tool={step['tool_name']}")
    if step.get("tool_args"):
        parts.append(f"args={json.dumps(step['tool_args'], default=str)}")
    for key in ("text", "action", "observation", "tool_output", "memory_value"):
        val = step.get(key)
        if val:
            parts.append(f"{key}: {val}")
    return "\n".join(parts)


def _downstream_summary(trace: Dict[str, Any], step_id: str) -> str:
    """All steps AFTER the intervened step (from the stored failing trace)."""
    steps = trace.get("steps") or []
    try:
        sid_int = int(step_id)
    except (TypeError, ValueError):
        return ""
    lines: List[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("step_id") is None:
            continue
        if int(s["step_id"]) <= sid_int:
            continue
        desc = s.get("text") or s.get("action") or s.get("tool_output") or ""
        lines.append(f"  step {s['step_id']} [{s.get('step_type','?')}]: {desc}")
    return "\n".join(lines)


def _find_trace(run_doc: Dict[str, Any], problem_id: Any) -> Optional[Dict[str, Any]]:
    for t in run_doc.get("failing_traces", []) or []:
        if str(t.get("problem_id")) == str(problem_id):
            return t
    return None


def _find_proposal(
    trace: Dict[str, Any], step_id: str, proposal_idx: Optional[int], want_success: bool
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (proposal_dict, successful_repair_dict_or_None).

    For the success stratum we prefer ``successful_repairs`` (which also has
    ``repaired_trace``). For the failure stratum we scan ``all_proposals_by_step``.
    """
    analysis = trace.get("analysis") or {}
    cf = analysis.get("counterfactual_repair") or {}
    successful = cf.get("successful_repairs") or cf.get("best_repairs") or {}
    all_by_step = cf.get("all_proposals_by_step") or {}

    if want_success:
        repair = successful.get(str(step_id))
        if isinstance(repair, dict) and (
            proposal_idx is None or repair.get("proposal_idx") == proposal_idx
        ):
            return repair, repair
        return None, None

    proposals = all_by_step.get(str(step_id)) or []
    for p in proposals:
        if not isinstance(p, dict):
            continue
        if p.get("success_predicted") is False and (
            proposal_idx is None or p.get("proposal_idx") == proposal_idx
        ):
            return p, None
    return None, None


def build_rows(run_doc: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    dataset = manifest["dataset"]
    run_id = manifest["run_id"]

    for entry in manifest["sample"]:
        problem_id = entry["problem_id"]
        step_id = str(entry["step_id"])
        proposal_idx = entry.get("proposal_idx")
        want_success = entry["stratum"] == "success"
        trace = _find_trace(run_doc, problem_id)
        if trace is None:
            print(f"WARN: trace not found for problem_id={problem_id!r}")
            continue

        proposal, repair = _find_proposal(trace, step_id, proposal_idx, want_success)
        if proposal is None:
            print(
                f"WARN: proposal not found for problem_id={problem_id!r} "
                f"step={step_id} idx={proposal_idx} stratum={entry['stratum']}"
            )
            continue

        # Repaired final answer: only present when the reexecutor path stored a trace.
        repaired_final = ""
        if repair is not None:
            rt = repair.get("repaired_trace")
            if isinstance(rt, dict):
                repaired_final = str(rt.get("final_answer", "") or "")

        original_step = proposal.get("original_step") or {}
        repaired_step = proposal.get("repaired_step") or {}
        # Fallback to the flat text fields if the nested step dicts were stripped.
        orig_text = _step_text(original_step) or str(proposal.get("original_text") or "")
        repd_text = _step_text(repaired_step) or str(proposal.get("repaired_text") or "")

        rows.append({
            "dataset": dataset,
            "run_id": run_id,
            "problem_id": str(problem_id),
            "step_id": step_id,
            "proposal_idx": "" if proposal_idx is None else str(proposal_idx),
            "judge_verdict": "success" if want_success else "failure",
            "problem_statement": _truncate(trace.get("problem_statement")),
            "gold_answer": _truncate(trace.get("gold_answer"), 500),
            "original_final_answer": _truncate(trace.get("final_answer"), 500),
            "repaired_final_answer": _truncate(repaired_final, 500),
            "original_step_text": _truncate(orig_text),
            "repaired_step_text": _truncate(repd_text),
            "downstream_context": _truncate(_downstream_summary(trace.get("trace") or {}, step_id), 20000),
            "human_label": "",
            "human_notes": "",
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["sealqa", "medbrowse"])
    p.add_argument(
        "--db-name",
        default=os.getenv("MONGODB_NAME", "causal_flow_dups"),
    )
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    manifest_path = args.manifest or (REVIEW_DIR / f"{args.dataset}_sample.json")
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    db = get_db(args.db_name)
    run_doc = db.runs.find_one({"run_id": manifest["run_id"]})
    if not run_doc:
        print(f"ERROR: run not found: {manifest['run_id']}", file=sys.stderr)
        return 2

    rows = build_rows(run_doc, manifest)

    out_path = args.out or (REVIEW_DIR / f"{args.dataset}_review.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
