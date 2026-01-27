#!/usr/bin/env python3
"""
Compute CRS Precision and Consensus Rate metrics from experiment JSON files.

This script analyzes the experiment results to compute:
- CRS Precision: Average precision of CRS-flagged steps (fraction that admit validated repairs)
- Consensus Rate: Fraction of CRS-flagged steps with consensus_score >= tau_c (0.5)

Output is formatted for Table 5 in the paper.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_json_file(filepath: Path) -> list:
    """Load JSON file and return list of experiment runs."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics_from_run(run: dict, experiment_filter: str = None) -> dict:
    """
    Extract precision and consensus metrics from a single experiment run.

    Returns dict with:
    - precision_values: list of precision scores from failed traces with analysis
    - consensus_scores: list of all consensus scores from CRS-flagged steps
    - consensus_above_threshold: count of scores >= 0.5
    - total_crs_steps: total count of CRS-flagged steps
    """
    exp_name = run.get('experiment_name', '')

    # Filter by experiment name if specified
    if experiment_filter and experiment_filter not in exp_name:
        return None

    precision_values = []
    consensus_scores = []
    consensus_above_threshold = 0
    total_crs_steps = 0

    # Process failed traces with analysis
    # Note: key is 'failing_traces' in JSON files
    failed_traces = run.get('failing_traces', []) or run.get('failed_traces', [])

    for trace in failed_traces:
        # Metrics are stored at trace level, not inside analysis
        metrics = trace.get('metrics', {})
        if not metrics:
            continue

        # Extract precision from attribution metrics
        attribution = metrics.get('attribution', {})
        precision = attribution.get('precision')
        if precision is not None:
            precision_values.append(precision)

        # Extract consensus scores from multi_agent metrics
        multi_agent = metrics.get('multi_agent', {})
        steps_with_agreement = multi_agent.get('steps_with_agreement', [])

        for step in steps_with_agreement:
            cs = step.get('consensus_score')
            if cs is not None:
                consensus_scores.append(cs)
                total_crs_steps += 1
                if cs >= 0.5:
                    consensus_above_threshold += 1

    return {
        'experiment_name': exp_name,
        'precision_values': precision_values,
        'consensus_scores': consensus_scores,
        'consensus_above_threshold': consensus_above_threshold,
        'total_crs_steps': total_crs_steps
    }


def aggregate_metrics_by_benchmark(all_runs: list) -> dict:
    """
    Aggregate metrics across all runs, grouped by benchmark.

    Returns dict mapping benchmark name to aggregated metrics.
    """
    benchmarks = defaultdict(lambda: {
        'precision_values': [],
        'consensus_scores': [],
        'consensus_above_threshold': 0,
        'total_crs_steps': 0
    })

    for run in all_runs:
        result = extract_metrics_from_run(run)
        if result is None:
            continue

        exp_name = result['experiment_name']

        # Map experiment names to benchmark names
        if 'GSM8K' in exp_name:
            benchmark = 'GSM8K'
        elif 'MBPP' in exp_name or 'Humaneval' in exp_name:
            benchmark = 'MBPP'
        elif 'SealQA' in exp_name:
            benchmark = 'SealQA Hard'
        elif 'MedBrowseComp' in exp_name:
            benchmark = 'MedBrowseComp'
        elif 'BrowseComp' in exp_name:
            # BrowseComp without SealQA or Med prefix - skip or classify
            continue
        else:
            continue

        benchmarks[benchmark]['precision_values'].extend(result['precision_values'])
        benchmarks[benchmark]['consensus_scores'].extend(result['consensus_scores'])
        benchmarks[benchmark]['consensus_above_threshold'] += result['consensus_above_threshold']
        benchmarks[benchmark]['total_crs_steps'] += result['total_crs_steps']

    return benchmarks


def compute_final_metrics(benchmarks: dict) -> dict:
    """
    Compute final CRS Precision and Consensus Rate for each benchmark.
    """
    results = {}

    for benchmark, data in benchmarks.items():
        precision_values = data['precision_values']
        total_crs_steps = data['total_crs_steps']
        consensus_above_threshold = data['consensus_above_threshold']

        # CRS Precision = mean of precision values
        if precision_values:
            crs_precision = sum(precision_values) / len(precision_values)
        else:
            crs_precision = None

        # Consensus Rate = fraction of steps with consensus >= 0.5
        if total_crs_steps > 0:
            consensus_rate = consensus_above_threshold / total_crs_steps
        else:
            consensus_rate = None

        results[benchmark] = {
            'crs_precision': crs_precision,
            'consensus_rate': consensus_rate,
            'num_precision_samples': len(precision_values),
            'num_consensus_samples': total_crs_steps
        }

    return results


def main():
    # File paths
    base_dir = Path(__file__).parent.parent / 'db_migrations'
    mbpp_file = base_dir / 'mbpp_results.json'
    result_file = base_dir / 'result_experiment.json'

    all_runs = []

    # Load MBPP results
    if mbpp_file.exists():
        print(f"Loading {mbpp_file}...")
        mbpp_data = load_json_file(mbpp_file)
        all_runs.extend(mbpp_data)
        print(f"  Loaded {len(mbpp_data)} runs")

    # Load other experiment results
    if result_file.exists():
        print(f"Loading {result_file}...")
        result_data = load_json_file(result_file)
        all_runs.extend(result_data)
        print(f"  Loaded {len(result_data)} runs")

    print(f"\nTotal runs: {len(all_runs)}")

    # Aggregate by benchmark
    benchmarks = aggregate_metrics_by_benchmark(all_runs)

    # Compute final metrics
    results = compute_final_metrics(benchmarks)

    # Print results
    print("\n" + "="*70)
    print("Attribution Metrics for Table 5")
    print("="*70)
    print(f"{'Benchmark':<20} {'CRS Precision':>15} {'Consensus Rate':>15} {'N (prec)':>10} {'N (cons)':>10}")
    print("-"*70)

    benchmark_order = ['GSM8K', 'MBPP', 'SealQA Hard', 'MedBrowseComp']

    for benchmark in benchmark_order:
        if benchmark in results:
            r = results[benchmark]
            prec = f"{r['crs_precision']:.2f}" if r['crs_precision'] is not None else "N/A"
            cons = f"{r['consensus_rate']:.2f}" if r['consensus_rate'] is not None else "N/A"
            print(f"{benchmark:<20} {prec:>15} {cons:>15} {r['num_precision_samples']:>10} {r['num_consensus_samples']:>10}")
        else:
            print(f"{benchmark:<20} {'N/A':>15} {'N/A':>15} {'0':>10} {'0':>10}")

    print("="*70)

    # Print LaTeX-ready output
    print("\nLaTeX table rows:")
    print("-"*70)
    for benchmark in benchmark_order:
        if benchmark in results:
            r = results[benchmark]
            prec = f"{r['crs_precision']:.2f}" if r['crs_precision'] is not None else "---"
            cons = f"{r['consensus_rate']:.2f}" if r['consensus_rate'] is not None else "---"
            print(f"{benchmark} & {prec} & {cons} \\\\")
        else:
            print(f"{benchmark} & --- & --- \\\\")


if __name__ == '__main__':
    main()
