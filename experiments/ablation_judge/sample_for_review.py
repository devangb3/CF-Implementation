"""Stratified sampler for the judge-accuracy ablation.

Pulls repair proposals from the local MongoDB dump
(default db ``causal_flow_dups``, collection ``runs``), filters to a single
source run per dataset, stratifies by ``success_predicted``, and writes a
deterministic JSON sample manifest that ``build_review_csv.py`` consumes.

Usage:
    # discover run IDs in local Mongo
    python -m experiments.ablation_judge.sample_for_review --list-runs

    # sample for one dataset
    python -m experiments.ablation_judge.sample_for_review \
        --dataset sealqa --run-id run_SealQA_...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient


DATASETS: Dict[str, Dict[str, Any]] = {
    "sealqa":      {"experiment_name": "SealQA",        "n_pos": 30, "n_neg": 0},
    "medbrowse":   {"experiment_name": "MedBrowseComp", "n_pos": 30, "n_neg": 0},
}
# Note: original runs only persist judge-success repairs (via `successful_repairs`
# or `best_repairs`). `all_proposals_by_step` — which would contain
# success_predicted=False rows — is only populated in the ablation_minimality
# reruns. We audit the inflation-relevant judge-success stratum directly from the
# headline runs;
SEED = 20260427

REVIEW_DIR = Path(__file__).parent / "review"


def get_db(db_name: str) -> Any:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in .env")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name]


def list_runs(db: Any) -> None:
    for exp in sorted({d["experiment_name"] for d in DATASETS.values()}):
        print(f"\n=== {exp} ===")
        cursor = db.runs.find(
            {"experiment_name": exp},
            {"run_id": 1, "timestamp": 1, "model_used": 1, "stats": 1},
        ).sort("timestamp", -1)
        for doc in cursor:
            stats = doc.get("stats", {})
            print(
                f"  {doc['run_id']}  "
                f"model={doc.get('model_used','?')}  "
                f"total={stats.get('total','?')}  "
                f"failing={stats.get('failing','?')}"
            )


def _iter_proposals(run_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten all per-trace repair proposals into a list of candidate rows.

    Each row captures the minimum needed to (a) identify the proposal in Mongo
    and (b) reconstruct the reviewer context later.
    """
    out: List[Dict[str, Any]] = []
    for trace in run_doc.get("failing_traces", []) or []:
        analysis = trace.get("analysis") or {}
        cf = analysis.get("counterfactual_repair") or {}
        # Headline runs use either `successful_repairs` (MedBrowseComp) or
        # `best_repairs` (SealQA) — both are keyed by step_id.
        successful = cf.get("successful_repairs") or cf.get("best_repairs") or {}
        all_by_step = cf.get("all_proposals_by_step") or {}

        problem_id = trace.get("problem_id")

        # Judge-success stratum: the repairs selected as successful (one per step).
        for step_id, repair in successful.items():
            if not isinstance(repair, dict):
                continue
            if repair.get("success_predicted") is False:
                continue
            out.append({
                "stratum": "success",
                "problem_id": problem_id,
                "step_id": str(step_id),
                "proposal_idx": repair.get("proposal_idx"),
                "success_predicted": True,
            })

        # Judge-failure stratum: all proposals with success_predicted=False.
        for step_id, proposals in all_by_step.items():
            if not isinstance(proposals, list):
                continue
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                if p.get("success_predicted") is False:
                    out.append({
                        "stratum": "failure",
                        "problem_id": problem_id,
                        "step_id": str(step_id),
                        "proposal_idx": p.get("proposal_idx"),
                        "success_predicted": False,
                    })

    return out


def _sample(
    pool: List[Dict[str, Any]], n: Optional[int], rng: random.Random
) -> List[Dict[str, Any]]:
    if n is None or n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


def sample_run(
    db: Any, dataset: str, run_id: str, out_path: Path
) -> Tuple[int, int]:
    cfg = DATASETS[dataset]
    run_doc = db.runs.find_one({"run_id": run_id})
    if not run_doc:
        raise RuntimeError(f"run_id not found: {run_id}")
    if run_doc.get("experiment_name") != cfg["experiment_name"]:
        raise RuntimeError(
            f"run {run_id} has experiment_name="
            f"{run_doc.get('experiment_name')!r}, expected "
            f"{cfg['experiment_name']!r}"
        )

    candidates = _iter_proposals(run_doc)
    pos_pool = [c for c in candidates if c["stratum"] == "success"]
    neg_pool = [c for c in candidates if c["stratum"] == "failure"]

    rng = random.Random(SEED)
    # Sort for determinism before sampling.
    pos_pool.sort(key=lambda c: (str(c["problem_id"]), c["step_id"], c["proposal_idx"] or 0))
    neg_pool.sort(key=lambda c: (str(c["problem_id"]), c["step_id"], c["proposal_idx"] or 0))

    pos_sample = _sample(pos_pool, cfg["n_pos"], rng)
    neg_sample = _sample(neg_pool, cfg["n_neg"], rng)

    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "experiment_name": cfg["experiment_name"],
        "run_id": run_id,
        "seed": SEED,
        "requested": {"n_pos": cfg["n_pos"], "n_neg": cfg["n_neg"]},
        "pool_sizes": {"pos": len(pos_pool), "neg": len(neg_pool)},
        "sample": pos_sample + neg_sample,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return len(pos_sample), len(neg_sample)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list-runs", action="store_true")
    p.add_argument("--dataset", choices=sorted(DATASETS.keys()))
    p.add_argument("--run-id", help="run_id in Mongo to sample from")
    p.add_argument(
        "--db-name",
        default=os.getenv("MONGODB_NAME", "causal_flow_dups"),
        help="Mongo db name (default: causal_flow_dups)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    db = get_db(args.db_name)

    if args.list_runs:
        list_runs(db)
        return 0

    if not args.dataset or not args.run_id:
        p.error("--dataset and --run-id are required unless --list-runs is set")

    out_path = args.out or (REVIEW_DIR / f"{args.dataset}_sample.json")
    n_pos, n_neg = sample_run(db, args.dataset, args.run_id, out_path)
    print(f"Wrote {out_path}  (pos={n_pos}, neg={n_neg})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
