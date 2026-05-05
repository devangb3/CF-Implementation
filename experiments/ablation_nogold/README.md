# No-gold repair-prompt ablation (Half A)

Responds to the reviewer concern that placing the gold answer inside
the repair prompt raises leakage concerns. This ablation strips gold from
the counterfactual-repair prompt only (attribution and critique remain
unchanged) and measures the paired repair-success-rate delta against the
existing minimality-ablation runs, which serve as the with-gold control.


## What changes

- `counterfactual_repair.py` — new `use_gold_in_prompts: bool = True`
  flag on `CounterfactualRepair.__init__`. When `False`:
  - The system message drops the "DO NOT reference the gold answer" rule.
  - `_create_repair_prompt` drops the `Correct Answer (FOR REFERENCE ONLY)`
    line, the associated "DO NOT incorporate" constraints, and the
    BAD/GOOD examples that name a specific gold string.
- `causal_flow.py` — `analyze_trace` accepts and forwards the flag.
- Attribution (`causal_attribution.py`) and critique
  (`multi_agent_critique.py`) are intentionally **not** touched — leaving
  step selection comparable to the main paper.

Evaluation still uses gold as the off-line oracle (`reexecution_utils.py`,
`browsecomp_eval.py` grader). Gold never enters an LLM prompt when the
flag is `False`.

## Sample

Same repairable-trace sample as the minimality ablation, so the two arms
are paired proposal-for-proposal (matched on `problem_id`). Counts reflect
what actually exists in `causal_flow_dups` — SealQA and MedBrowseComp
have fewer repairable traces than the original plan assumed, so the runner
takes all of them:

| Benchmark      | Traces | Source run (`runs` collection)                       |
|----------------|--------|------------------------------------------------------|
| GSM8K          | 100    | `run_GSM8K_2025-12-20T07:38:08.801930`               |
| MBPP           | 100    | `run_MBPP_2025-12-11T00:49:23.675982`                |
| SealQA Hard    | 26     | `run_SealQA_2025-12-18T02:41:05.742380`              |
| MedBrowseComp  | 67     | `run_MedBrowseComp_2025-12-19T21:55:32.749095`       |
| **Total**      | **293**|                                                      |

`K=5` proposals per causal step, deterministic ordering inherited from
the source run.

## Running

```bash
python -m experiments.ablation_nogold.run_ablation \
    --benchmark gsm8k \
    --source-run-id run_GSM8K_2025-12-20T07:38:08.801930

python -m experiments.ablation_nogold.run_ablation \
    --benchmark mbpp \
    --source-run-id run_MBPP_2025-12-11T00:49:23.675982

python -m experiments.ablation_nogold.run_ablation \
    --benchmark sealqa \
    --source-run-id run_SealQA_2025-12-18T02:41:05.742380

python -m experiments.ablation_nogold.run_ablation \
    --benchmark medbrowse \
    --source-run-id run_MedBrowseComp_2025-12-19T21:55:32.749095
```

Add `--dry-run` to verify loading + sampling without any LLM or MongoDB
writes. Add `--limit 2` for a live smoke test against MongoDB before
committing to the full run.

The runner targets MongoDB database `causal_flow_dups` by default (the DB
that holds the paired minimality-ablation runs). Override with `--db-name`
if needed. Writes into `experiment_name = ablation_nogold_{benchmark}`.

## Expected output

A single primary number: paired delta in per-proposal repair success rate
between with-gold (reused from `ablation_minimality_{benchmark}`) and
no-gold (this ablation). No minimality numbers are reported here — that
belongs to the separate minimality ablation.
