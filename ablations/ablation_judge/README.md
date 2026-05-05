# Ablation: Judge Accuracy for Predictive Re-Execution

Addresses reviewer GNVB's concern that the LLM-based validator used at repair SealQA Hard, and MedBrowseComp could inflate reported repair
rates if the judge is wrong.

MBPP and HumanEval use deterministic executors and are out of scope.

## Pipeline

1. `sample_for_review.py` — pull repairs from local MongoDB
   (`causal_flow_dups`, collection `runs`), stratify by `success_predicted`,
   and write a JSON sample manifest per dataset.
2. `build_review_csv.py` — expand the manifest into a reviewer CSV containing
   the problem, gold answer, original final answer, repaired final answer,
   step context, judge verdict, and an empty `human_label` column.
3. Human review (binary: `correct` / `incorrect` / `unclear`), saved as
   `<dataset>_labeled.csv` next to the original.
4. `analyze.py` — compute precision, recall, overall agreement, Cohen's κ,
   Wilson 95% CI on precision, and an inflation-adjusted repair rate
   (`original_repair_rate × precision_on_success`) per dataset. Emit a single
   summary Markdown table + JSON.

## Sample sizes (current)

| Dataset         | judge-success | judge-failure | pool avail. |
| --------------- | ------------- | ------------- | ----------- |
| MedBrowseComp   | 30            | 0             | 126         |
| SealQA Hard     | 30            | 0             | 49          |

Judge-success is the load-bearing stratum — GNVB's concern is about
false-positive validation inflating the repair rate. The original headline
runs do not persist judge-failure proposals (`all_proposals_by_step` is
empty), so we audit the success stratum directly from the headline runs
rather than mixing in the ablation_minimality rerun, which uses a different
judge (`_llm_predict_outcome` uniformly).

## Run IDs used (from local `causal_flow_dups`)

| Dataset         | run_id                                                        | failing |
| --------------- | ------------------------------------------------------------- | ------: |
| SealQA          | `run_SealQA_2025-12-18T02:41:05.742380` (Gemini 3 flash)      | 140     |
| MedBrowseComp   | `run_MedBrowseComp_2025-12-19T21:55:32.749095` (Gemini 3)     | 270     |

## What the reviewer sees per row

- `problem_statement`, `gold_answer`
- `original_final_answer` (agent's failed answer)
- `repaired_final_answer` (the re-executed final answer as the judge saw it).
  - **SealQA**:     Reviewer judges whether the proposed tool-call/reasoning would plausibly
    reach the gold answer.
  - **MedBrowseComp**: populated — direct answer-vs-gold comparison.
- `original_step_text` and `repaired_step_text`: what changed.
- `downstream_context`: a few post-step lines from the original failing trace.

## Conventions

- "Judge" = whatever produced `success_predicted` on the repair at
  experiment time.
  For SealQA / MedBrowseComp it is the `GRADER_TEMPLATE` check against the
  gold answer.
- We audit *the judge as it ran*, so the "repaired final answer" shown to the
  human reviewer is the one the judge actually saw (cached in
  `analysis.counterfactual_repair.successful_repairs.<step>.repaired_trace`).
- Labels are binary (`correct` / `incorrect`); `unclear` is allowed but
  excluded from the main precision/recall numbers and reported separately.

## Usage

```bash
# 1. discover run IDs
python -m ablations.ablation_judge.sample_for_review --list-runs

# 2. sample per dataset (writes review/<dataset>_sample.json)
python -m ablations.ablation_judge.sample_for_review \
    --dataset seal_qa \
    --run-id <seal_qa_run_id>

# 3a. build visual review UI (preferred — writes review/<dataset>_review.html)
python -m ablations.ablation_judge.build_review_html --dataset sealqa

# 3b. or build a plain CSV (writes review/<dataset>_review.csv)
python -m ablations.ablation_judge.build_review_csv --dataset sealqa

# 4. in the HTML UI, press 1 (correct) / 2 (incorrect) / 3 (unclear),
#    n/p to navigate, then click "Download labeled CSV" and save it as
#    review/<dataset>_labeled.csv. Labels persist in browser localStorage,
#    so you can close/reopen the page without losing progress.

# 5. compute metrics (writes RESULTS.md and results.json)
python -m ablations.ablation_judge.analyze
```

## Output

`RESULTS.md` contains the per-dataset table and the adjusted repair rates
that should be referenced in the rebuttal.
