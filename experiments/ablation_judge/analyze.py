"""Compute judge-accuracy metrics from labeled reviewer CSVs.

Reads ``review/<dataset>_labeled.csv`` files, treats ``human_label`` as ground
truth, and reports per-dataset precision on judge-success, recall on
judge-failure, overall agreement, Cohen's κ, a Wilson 95% CI on precision,
and an inflation-adjusted repair rate:

    adjusted_repair_rate = reported_repair_rate × precision_on_success

Rows with ``human_label == "unclear"`` are excluded from precision/recall
and counted separately.

Usage:
    python -m experiments.ablation_judge.analyze \
        [--reported-rates sealqa=0.219 medbrowse=0.445]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REVIEW_DIR = Path(__file__).parent / "review"
RESULTS_JSON = Path(__file__).parent / "results.json"
RESULTS_MD = Path(__file__).parent / "RESULTS.md"

# Reported repair rates from causalflow.tex Table "per_model". These can be
# overridden via --reported-rates on the CLI.
DEFAULT_REPORTED_RATES: Dict[str, float] = {
    "sealqa":    0.219,   #  32/146
    "medbrowse": 0.445,   # 149/335
}

DATASETS = ["sealqa", "medbrowse"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohens_kappa(rows: List[Dict[str, str]]) -> Optional[float]:
    """κ between judge_verdict (success/failure) and human_label
    (correct/incorrect). ``unclear`` rows are excluded.
    """
    a: List[int] = []  # judge says success? 1/0
    b: List[int] = []  # human says correct? 1/0
    for r in rows:
        hl = r["human_label"].strip().lower()
        if hl not in ("correct", "incorrect"):
            continue
        a.append(1 if r["judge_verdict"] == "success" else 0)
        b.append(1 if hl == "correct" else 0)
    n = len(a)
    if n == 0:
        return None
    # κ is undefined when one rater has no variance (e.g. all judge-success).
    if sum(a) in (0, n) or sum(b) in (0, n):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def analyze_dataset(
    rows: List[Dict[str, str]], reported_rate: Optional[float]
) -> Dict[str, Any]:
    n_total = len(rows)
    unclear = [r for r in rows if r["human_label"].strip().lower() == "unclear"]
    scored = [r for r in rows if r["human_label"].strip().lower() in ("correct", "incorrect")]

    # Precision on judge-success = P(human=correct | judge=success)
    js_rows = [r for r in scored if r["judge_verdict"] == "success"]
    js_correct = sum(1 for r in js_rows if r["human_label"].strip().lower() == "correct")
    precision = (js_correct / len(js_rows)) if js_rows else None
    prec_ci = wilson_ci(js_correct, len(js_rows)) if js_rows else None

    # Recall on judge-failure = P(human=incorrect | judge=failure)
    # (i.e. the judge correctly rejected truly bad repairs)
    jf_rows = [r for r in scored if r["judge_verdict"] == "failure"]
    jf_incorrect = sum(1 for r in jf_rows if r["human_label"].strip().lower() == "incorrect")
    specificity = (jf_incorrect / len(jf_rows)) if jf_rows else None

    # Overall agreement: fraction of scored rows where judge and human align.
    agreement_n = sum(
        1 for r in scored
        if (r["judge_verdict"] == "success") == (r["human_label"].strip().lower() == "correct")
    )
    agreement = (agreement_n / len(scored)) if scored else None

    kappa = cohens_kappa(scored)

    adjusted = None
    if precision is not None and reported_rate is not None:
        adjusted = reported_rate * precision

    return {
        "n_reviewed_total": n_total,
        "n_unclear": len(unclear),
        "n_scored": len(scored),
        "judge_success": {
            "n": len(js_rows),
            "human_correct": js_correct,
            "precision": precision,
            "precision_wilson_95ci": prec_ci,
        },
        "judge_failure": {
            "n": len(jf_rows),
            "human_incorrect": jf_incorrect,
            "specificity": specificity,
        },
        "agreement": agreement,
        "cohens_kappa": kappa,
        "reported_repair_rate": reported_rate,
        "adjusted_repair_rate": adjusted,
    }


def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _fmt_ci(ci: Optional[Tuple[float, float]]) -> str:
    if ci is None:
        return "—"
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def render_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Judge-Accuracy Ablation Results\n")
    lines.append("Addresses reviewer GNVB: how accurate is the LLM validator used")
    lines.append("at repair time for predictive re-execution? If judge precision")
    lines.append("on judge-'success' rows is below 1.0, the reported repair rate")
    lines.append("is inflated proportionally.\n")
    lines.append("## Per-Dataset Summary\n")
    lines.append(
        "| Dataset | Reviewed | Unclear | Judge-success (n / precision / 95% CI) "
        "| Judge-failure (n / specificity) | Agreement | κ | Reported | Adjusted |"
    )
    lines.append(
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |"
    )
    for ds, res in results.items():
        js = res["judge_success"]
        jf = res["judge_failure"]
        kappa = res["cohens_kappa"]
        kappa_str = "—" if kappa is None else f"{kappa:.2f}"
        lines.append(
            f"| {ds} "
            f"| {res['n_reviewed_total']} "
            f"| {res['n_unclear']} "
            f"| {js['n']} / {_fmt_pct(js['precision'])} / {_fmt_ci(js['precision_wilson_95ci'])} "
            f"| {jf['n']} / {_fmt_pct(jf['specificity'])} "
            f"| {_fmt_pct(res['agreement'])} "
            f"| {kappa_str} "
            f"| {_fmt_pct(res['reported_repair_rate'])} "
            f"| {_fmt_pct(res['adjusted_repair_rate'])} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "Precision on judge-success is the headline number: it bounds how much "
        "the reported repair rate can be inflated by a permissive judge. "
        "`Adjusted = Reported × Precision` gives a conservative lower bound "
        "on the true repair rate under the assumption that the sampled "
        "judge-success stratum is representative."
    )
    return "\n".join(lines) + "\n"


def load_labeled(dataset: str) -> List[Dict[str, str]]:
    path = REVIEW_DIR / f"{dataset}_labeled.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reported-rates",
        nargs="*",
        default=None,
        help="Override reported rates, e.g. sealqa=0.219 medbrowse=0.445",
    )
    args = p.parse_args(argv)

    reported = dict(DEFAULT_REPORTED_RATES)
    if args.reported_rates:
        for item in args.reported_rates:
            k, _, v = item.partition("=")
            reported[k.strip()] = float(v)

    results: Dict[str, Dict[str, Any]] = {}
    for ds in DATASETS:
        rows = load_labeled(ds)
        if not rows:
            print(f"NOTE: no labeled file for {ds}; skipping.")
            continue
        results[ds] = analyze_dataset(rows, reported.get(ds))

    if not results:
        print("No labeled datasets found. Label review/*_review.csv first.", file=sys.stderr)
        return 1

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    RESULTS_MD.write_text(render_markdown(results))
    print(f"Wrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
