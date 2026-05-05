# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python 3.8+, an OpenRouter API key, and optionally MongoDB and Docker (for code execution experiments).

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in keys
```

`.env` keys (match `.env.example`):
- `OPENROUTER_SECRET_KEY` — LLM API access via OpenRouter (do **not** use `OPENROUTER_API_KEY`; code expects `OPENROUTER_SECRET_KEY`)
- `MONGODB_URI` — MongoDB connection (default: `mongodb://localhost:27017/causalflow`)
- `SERPER_API_KEY` — Web search (required for BrowseComp/SealQA/MedBrowseComp)

**Paper / spec:** [`research/CausalFlow.pdf`](research/CausalFlow.pdf) — CRS, sequential re-execution, minimality, multi-agent consensus (§4–§5).

## Running Experiments

**Core CausalFlow experiments:**
```bash
python experiments/gsm8k/run_gsm8k_experiment.py
python experiments/mbpp/run_mbpp_experiment.py
python experiments/humaneval/run_humaneval_experiment.py
python experiments/browsecomp/run_browsecomp_experiment.py
python experiments/browsecomp/run_sealqa_experiment.py
python experiments/browsecomp/run_medbrowsecomp_experiment.py
```

**Baseline comparison experiments:**
```bash
python baseline_comparison/experiments/run_gsm8k_baselines.py --workers 8
python baseline_comparison/experiments/run_mbpp_baselines.py --workers 8
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --workers 4 --resume
python baseline_comparison/experiments/run_sealqa_baselines.py --workers 4
```

Common baseline flags: `--num_rows N`, `--workers N`, `--resume [FILE]`, `--baselines direct self_refine self_reflection`, `--model MODEL_ID`.

**Quick examples:**
```bash
python examples/demo.py
python examples/complex_example.py
```

## Architecture

CausalFlow is a post-hoc failure analysis framework for LLM agents (see `research/CausalFlow.pdf`, §4). When an agent fails, it:
1. Builds a causal DAG from the recorded execution trace
2. Identifies causal steps via **sequential** intervention and re-execution of downstream steps (CRS)
3. Generates minimal counterfactual repairs and validates them with the task verifier (executor or LLM **outcome prediction**)
4. Optionally runs **multi-agent critique** to confirm attributions (`skip_critique=True` when a deterministic executor already validates repairs, e.g. MBPP)

### Core pipeline (`causal_flow.py`)

`CausalFlow.analyze_trace(trace, reexecutor)` orchestrates the full pipeline:

```
TraceLogger → CausalGraph → CausalAttribution → CounterfactualRepair → MultiAgentCritique
```

### Key modules

| File | Role |
|---|---|
| `trace_logger.py` | Records agent execution as typed steps (REASONING, TOOL_CALL, TOOL_RESPONSE, LLM_RESPONSE, MEMORY_ACCESS, ENVIRONMENT_ACTION, ENVIRONMENT_OBSERVATION, FINAL_ANSWER). Each `Step` has an id, type, dependency list, text, and state snapshot. |
| `causal_graph.py` | Builds a NetworkX DAG from step dependencies. Provides ancestor/descendant queries and `get_critical_steps()`. |
| `causal_attribution.py` | Computes a Causal Responsibility Score (CRS) per step: generate an LLM intervention → reexecute from that step → CRS = 1.0 if outcome flips to success. |
| `counterfactual_repair.py` | For each high-CRS step, generates 3 minimal repair candidates with `minimality_score` and `success_predicted`. Uses optional deterministic reexecutor for validation. |
| `multi_agent_critique.py` | Three LLM agents independently assess causal attribution and produce a `consensus_score` per step. |
| `llm_client.py` | OpenRouter wrapper using the OpenAI SDK. `generate()` for free-form, `generate_structured()` for JSON-schema-constrained outputs. |
| `schemas.py` | Pydantic models for all structured LLM outputs: `InterventionOutput`, `RepairOutput`, `CritiqueOutput`, etc. |
| `mongodb_storage.py` | Persists runs, traces, and results to MongoDB. Truncates large fields automatically. |
| `reexecution_utils.py` | `BranchRunnableAgent` protocol and `AgentBranchExecutor`: clone an agent's history, inject an intervened step, and resume execution. |

### Reexecutors

Experiments use different validation for **intervened** traces:
- **GSM8K** — Agent uses `MathReexecutor` for calculator tools and final numeric grading; the GSM8K experiment passes `reexecutor=None` into `analyze_trace`, so **CRS/repair validation uses LLM outcome prediction** (not deterministic reexecution of the branched trace).
- **MBPP / HumanEval**: `HumanevalReexecutor` (Docker tests) for attribution and repair validation when wired into `analyze_trace`.
- **Web QA** (BrowseComp, SealQA, MedBrowseComp): LLM-based outcome prediction for interventions.

### Baselines (`baseline_comparison/`)

Three baselines compared against CausalFlow:
- **Direct**: Single-pass chain-of-thought (1 LLM call)
- **Self-Refine**: Generate → feedback → refine loop (stops early on `[STOP]`, up to `max_iter`)
- **Self-Reflection**: Generate → if wrong, produce 5-part reflection → re-answer with reflection injected

For web QA baselines, one `BrowseCompAgent` run is shared per problem; baselines then refine using LLM-only calls (no new web searches) to isolate the refinement strategy.

Results are written as JSON to `baseline_comparison/results/<task>_<timestamp>.json`.

### Web search caching

BrowseComp-based experiments cache web results in `.cache/browsecomp/` via `WebEnvironment` for deterministic reruns.

### Ablations (`ablations/`)

Reviewer-facing analyses (LLM judge calibration, minimality / no-gold / stochasticity) live under `ablations/` with package-style entry points, e.g. `python -m ablations.ablation_judge.sample_for_review`. See each subdirectory’s README.
