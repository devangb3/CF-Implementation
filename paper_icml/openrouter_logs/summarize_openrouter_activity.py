#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional


def _parse_decimal(value: str) -> Decimal:
    value = (value or "").strip()
    if value == "":
        return Decimal("0")
    try:
        return Decimal(value)
    except InvalidOperation:
        return Decimal("0")


def _parse_created_at(value: str) -> datetime:
    value = (value or "").strip()
    if value == "":
        raise ValueError("missing created_at")

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognized created_at format: {value!r}")


@dataclass
class Totals:
    rows: int = 0
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    sum_cost_total: Decimal = Decimal("0")
    sum_cost_web_search: Decimal = Decimal("0")
    sum_cost_cache: Decimal = Decimal("0")
    sum_cost_file_processing: Decimal = Decimal("0")
    sum_byok_usage_inference: Decimal = Decimal("0")

    @property
    def sum_total_cost(self) -> Decimal:
        return (
            self.sum_cost_total
            + self.sum_cost_web_search
            + self.sum_cost_cache
            + self.sum_cost_file_processing
            + self.sum_byok_usage_inference
        )

    def update(
        self,
        created_at: datetime,
        cost_total: Decimal,
        cost_web_search: Decimal,
        cost_cache: Decimal,
        cost_file_processing: Decimal,
        byok_usage_inference: Decimal,
    ) -> None:
        self.rows += 1
        if self.start is None or created_at < self.start:
            self.start = created_at
        if self.end is None or created_at > self.end:
            self.end = created_at
        self.sum_cost_total += cost_total
        self.sum_cost_web_search += cost_web_search
        self.sum_cost_cache += cost_cache
        self.sum_cost_file_processing += cost_file_processing
        self.sum_byok_usage_inference += byok_usage_inference


def _iter_rows(path: Path) -> Iterable[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize OpenRouter activity CSV: first/last timestamp and total costs for a model."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=Path("paper_icml/openrouter_logs/openrouter_activity_2026-01-29.csv"),
        help="Path to openrouter_activity_*.csv (default: %(default)s)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        type=str,
        help="Exact match on model_permaslug (e.g. openai/gpt-5-chat-2025-08-07)",
    )
    group.add_argument(
        "--model-contains",
        type=str,
        default="gpt-5-chat",
        help="Substring match on model_permaslug (default: %(default)s)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Derive the window from the model filter, then sum costs for all models within that window."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Optional: window start datetime (YYYY-MM-DD HH:MM:SS[.mmm]). Requires --end.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Optional: window end datetime (YYYY-MM-DD HH:MM:SS[.mmm]). Requires --start.",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: CSV not found: {args.csv_path}", file=sys.stderr)
        return 2

    model_counts: Counter[str] = Counter()

    def _matches_filter(model: str) -> bool:
        if args.model is not None:
            return model == args.model
        return bool(args.model_contains and args.model_contains in model)

    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    if (args.start is None) != (args.end is None):
        print("ERROR: --start and --end must be provided together.", file=sys.stderr)
        return 2
    if args.start is not None and args.end is not None:
        window_start = _parse_created_at(args.start)
        window_end = _parse_created_at(args.end)
        if window_end < window_start:
            print("ERROR: --end must be >= --start.", file=sys.stderr)
            return 2
    elif args.all_models:
        for row in _iter_rows(args.csv_path):
            model = (row.get("model_permaslug") or "").strip()
            if model == "":
                continue
            model_counts[model] += 1
            if not _matches_filter(model):
                continue
            try:
                created_at = _parse_created_at(row.get("created_at") or "")
            except ValueError:
                continue
            if window_start is None or created_at < window_start:
                window_start = created_at
            if window_end is None or created_at > window_end:
                window_end = created_at

        if window_start is None or window_end is None:
            print("No rows matched your model filter to derive a time window.", file=sys.stderr)
            print("Available models in CSV:", file=sys.stderr)
            for model, count in model_counts.most_common():
                print(f"  {model}: {count}", file=sys.stderr)
            return 1

    totals_by_model: dict[str, Totals] = defaultdict(Totals)

    for row in _iter_rows(args.csv_path):
        model = (row.get("model_permaslug") or "").strip()
        if model == "":
            continue
        if model not in model_counts:
            model_counts[model] += 1

        try:
            created_at = _parse_created_at(row.get("created_at") or "")
        except ValueError:
            continue

        if window_start is not None and window_end is not None:
            if created_at < window_start or created_at > window_end:
                continue

        if not args.all_models and not _matches_filter(model):
            continue

        cost_total = _parse_decimal(row.get("cost_total") or "")
        cost_web_search = _parse_decimal(row.get("cost_web_search") or "")
        cost_cache = _parse_decimal(row.get("cost_cache") or "")
        cost_file_processing = _parse_decimal(row.get("cost_file_processing") or "")
        byok_usage_inference = _parse_decimal(row.get("byok_usage_inference") or "")
        totals_by_model[model].update(
            created_at,
            cost_total,
            cost_web_search,
            cost_cache,
            cost_file_processing,
            byok_usage_inference,
        )

    if not totals_by_model:
        print("No rows matched your model filter.", file=sys.stderr)
        print("Available models in CSV:", file=sys.stderr)
        for model, count in model_counts.most_common():
            print(f"  {model}: {count}", file=sys.stderr)
        return 1

    def _fmt_dt(dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt else "N/A"

    print(f"CSV: {args.csv_path}")
    if window_start is not None and window_end is not None:
        print(f"Window start: {_fmt_dt(window_start)}")
        print(f"Window end:   {_fmt_dt(window_end)}")

    if not args.all_models:
        if args.model is not None:
            print(f"Filter: model == {args.model}")
        else:
            print(f"Filter: model contains {args.model_contains!r}")
        items = sorted(totals_by_model.items())
    else:
        items = sorted(totals_by_model.items(), key=lambda kv: kv[1].sum_total_cost, reverse=True)

    grand_total = Decimal("0")
    for model, totals in items:
        grand_total += totals.sum_total_cost
        print(f"\nModel: {model}")
        print(f"Rows: {totals.rows}")
        print(f"Start: {_fmt_dt(totals.start)}")
        print(f"End:   {_fmt_dt(totals.end)}")
        print(f"Sum total_cost:            {totals.sum_total_cost}")
        print(f"  cost_total:              {totals.sum_cost_total}")
        print(f"  cost_web_search:         {totals.sum_cost_web_search}")
        print(f"  cost_cache:              {totals.sum_cost_cache}")
        print(f"  cost_file_processing:    {totals.sum_cost_file_processing}")
        print(f"  byok_usage_inference:    {totals.sum_byok_usage_inference}")

    if args.all_models:
        print(f"\nGrand total_cost: {grand_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
