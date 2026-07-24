# CausalFlow Results Snapshot

## Selection rules

- Use one primary run per benchmark.
- Count the unique persisted traces in `passing_traces` and `failing_traces`; do not substitute the configured dataset size when records are missing.
- Treat `stats.fixed` as the number of baseline-failing traces with at least one successful repair under that benchmark's evaluator.
- Recompute all rates from the raw counts below. The historical `stats.accuracy` field is not semantically consistent across runs.

## Runs

| Benchmark | Run ID | Configured | Persisted | Baseline pass | Baseline fail | Failures with a successful repair | Failure repair rate | Effective success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | `run_GSM8K_2025-12-20T07:38:08.801930` | 1,319 | 1,219 | 989 | 230 | 173 | 75.2% | 95.3% |
| MBPP | `run_MBPP_2025-12-11T00:49:23.675982` | 974 | 971 | 523 | 448 | 201 | 44.9% | 74.6% |
| SealQA Hard | `run_SealQA_2025-12-18T02:41:05.742380` | 240 | 240 | 100 | 140 | 26 | 18.6% | 52.5% |
| MedBrowseComp | `run_MedBrowseComp_2025-12-19T21:55:32.749095` | 405 | 405 | 135 | 270 | 67 | 24.8% | 49.9% |
| **Aggregate** | — | **2,938** | **2,835** | **1,747** | **1,088** | **467** | **42.9%** | **78.1%** |


## What “successful repair” means

The validation path differs by benchmark and must accompany any detailed claim:

- **GSM8K:** the baseline answer is numerically graded, but CausalFlow attribution and repair pass `reexecutor=None`; counterfactual outcomes are predicted by an LLM.
- **MBPP:** candidate repairs continue through the code agent and are checked by Dockerized tests, providing deterministic task validation.
- **SealQA Hard and MedBrowseComp:** candidate repairs resume the browsing agent; final answers are evaluated by the benchmark's LLM grader.

Accordingly, the aggregate is best described as **benchmark-evaluator-measured repair**, not as a single uniform deterministic metric. A “fixed” trace means at least one candidate repair succeeded under that evaluator; it is not the number of repaired steps or generated repair candidates.