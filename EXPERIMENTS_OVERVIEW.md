# CausalFlow Experiments

Design and metrics align with **§4–§5** of [`research/CausalFlow.pdf`](research/CausalFlow.pdf): sequential interventions, CRS, minimality-scored repairs, optional multi-agent consensus, and (per domain) deterministic re-execution vs. **predictive** validation.

## Framework snapshot

- **Trace logging** — Typed steps and explicit dependencies (`trace_logger.py`) match the paper’s “dependencies logged from the agent runtime” (tool response → tool call; reasoning chains; environment observations after actions).
- **CRS** — Step-level interventions replace a candidate step; **descendants are re-executed**; the verifier \(V\) on the final outcome defines success (implementation in `causal_attribution.py`).
- **Repairs** — For steps with CRS indicating an outcome flip, `counterfactual_repair.py` proposes edits and ranks by **minimality** (token overlap; optional semantic minimality flag).
- **Critique** — `multi_agent_critique.py` implements ensemble validation of attributions; some benchmarks pass `skip_critique=True` when deterministic execution already grounds validation (e.g. MBPP with Docker).
- **Persistence** — Traces and run metadata go to MongoDB via `mongodb_storage.py` when experiments are configured to use it.

**Re-execution vs. prediction** (paper §4.1): where a **domain executor** exists (e.g. Python tests in Docker), intervened traces are evaluated deterministically; otherwise an **LLM predicts** whether the intervention would flip the outcome. This split is per-task, not “one mode for the whole codebase.”

## Implemented Experiments
- **GSM8K** (`experiments/gsm8k/run_gsm8k_experiment.py`)
  - Data: HuggingFace `gsm8k` test split (1,319 problems).
  - Agent: `GSM8KAgent` uses `MathReexecutor` for **in-episode** calculator tools and **final numeric grading**. For **CausalFlow attribution and repair**, the GSM8K run script passes `reexecutor=None` to `analyze_trace`, so **intervened traces are judged via LLM outcome prediction** (paper protocol for this benchmark), not by re-running the symbolic executor over the branched trace.
  - Model in run script: `google/gemini-2.0-flash-lite-001` (commented to avoid contamination); critique enabled by default.
- **MBPP** (`experiments/mbpp/run_mbpp_experiment.py`)
  - Data: HF `mbpp` with train/test/validation/prompt splits merged by `MBPPDataLoader`.
  - Agent: Reuses Humaneval-style code generator (`HumanevalAgent`) with docker execution for tests and deterministic `HumanevalReexecutor`. CausalFlow runs only on failures, skipping critique because docker results provide ground truth; repairs are attempted via agent branching.
  - Model in run script: `openai/gpt-5-chat`.
- **HumanEval** (`experiments/humaneval/run_humaneval_experiment.py`)
  - Same code-generation + Docker execution pattern as MBPP (`HumanevalAgent` / `HumanevalReexecutor`); included for HumanEval-style splits beyond the paper’s MBPP-focused code results.
- **Browse-based QA** (shared `BrowseCompAgent` in `experiments/browsecomp/`)
  - Agent: Structured tool policy over `web_search`, `web_fetch`, `extract`, `answer`; deterministic caching via `WebEnvironment` and Serper search. Grading uses an LLM checker (`grade_response`). CausalFlow analyzes failures with interventions on tool/LLM/reasoning; web logs are truncated before analysis.
  - Datasets/runs:
    - **Seal QA Hard** (`run_sealqa_experiment.py`): HF `vtllms/sealqa` (`seal_hard`), defaults to `google/gemini-3-flash-preview`.
    - **MedBrowseComp** (`run_medbrowsecomp_experiment.py`): HF `AIM-Harvard/MedBrowseComp_CUA`, defaults to `google/gemini-3-flash-preview`, max 10 steps.

## Results Snapshot (from provided sheet)
| Experiment | Dataset | Model used | Total | Passed | Failed | CausalFlow fixes | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k/viewer/main/test | google/gemini-2.0 | 1319 | 989 | 330 | 173 | Used LLM as predictor; deterministic repair was impractical. |
| MBPP | https://huggingface.co/datasets/Muennighoff/mbpp | GPT 5 Chat | 947 | 523 | 488 | 201 | Deterministic docker reexecution; critique skipped. |
| Seal QA Hard | https://huggingface.co/datasets/vtllms/sealqa/viewer/seal_hard | Gemini 3 Flash | 254 | 108 | 146 | 32 | Web-search agent with LLM grading. |
| Med BrowseComp | https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp_CUA | Gemini 3 Flash | 484 | 149 | 335 | 149 | Web-search agent with medical domain queries. |

## Operational notes

- Requires `OPENROUTER_SECRET_KEY` for model access; web experiments need `SERPER_API_KEY`. Docker is required for MBPP/HumanEval **deterministic** execution paths.
- CausalFlow storage and run metadata are typically persisted in MongoDB (`mongodb_storage.py`); run scripts update accuracy, analyzed failures, and repair counts.
- Cached web fetch/search results live under `.cache/browsecomp/` for deterministic replays.

## Ablations and extra analyses

Scripts and READMEs for reviewer-style analyses (judge calibration, minimality, no-gold prompts, stochasticity) live under [`ablations/`](ablations/) (Python module paths such as `python -m ablations.ablation_judge.sample_for_review`).
