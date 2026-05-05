# CausalFlow

Implementation of **CausalFlow**, an interventional framework for **causal attribution** and **counterfactual repair** on multi-step LLM agent traces. : Execution is modeled as a dependent sequence of steps; for failed traces, step-level interventions and sequential re-execution identify **Causal Responsibility Scores (CRS)**; high-CRS steps receive **minimal edits** that are validated under the task verifier, yielding **contrastive pairs** \((\text{wrong step}, \text{corrected step})\) useful for deploy-time repair and offline training (e.g. preference optimization).

## What’s in this repository

| Area | Path | Purpose |
|------|------|--------|
| Core library | Repository root (`causal_flow.py`, `trace_logger.py`, `causal_graph.py`, `causal_attribution.py`, `counterfactual_repair.py`, `multi_agent_critique.py`, …) | End-to-end `CausalFlow.analyze_trace(...)` pipeline |
| Task experiments | `experiments/` | GSM8K, MBPP, HumanEval, BrowseComp / SealQA Hard / MedBrowseComp agents and run scripts |
| Baselines | `baseline_comparison/` | Direct, Self-Refine, Self-Reflection (and pilot Tree-of-Thoughts code) vs. paper §5 |
| Ablations | `ablations/` | Judge accuracy, minimality, no-gold prompts, stochasticity (see each subfolder’s README) |
| DPO / learning | `DPO/` | Sketch and scripts for using repaired contrasts for preference-style training |
| Examples | `examples/` | Small demos and synthetic traces |

**High-level pipeline** (matches paper Figure 1):

1. **Trace** — Agent runtime logs typed steps (`REASONING`, `TOOL_CALL`, `TOOL_RESPONSE`, …) and explicit dependencies (`trace_logger.py`).
2. **DAG** — Steps and edges become a graph for attribution (`causal_graph.py`).
3. **CRS** — For candidate steps, the framework proposes interventions, re-executes **downstream** steps only, and checks whether the task verifier flips failure → success (`causal_attribution.py`). Where no deterministic executor exists, an **LLM predicts** whether the intervened trace would succeed (same distinction as paper §4.1 / §5.4).
4. **Repairs** — For causal steps, generate candidate repairs and prefer **minimal** edits; validation uses the same re-execution or predictive mechanism (`counterfactual_repair.py`).
5. **Multi-agent validation** — Optional ensemble of LLM critics confirms attributions (`multi_agent_critique.py`). Experiments that rely purely on deterministic execution (e.g. MBPP with Docker tests) often **skip** critique via `skip_critique=True`.

Runs and traces can be persisted with **MongoDB** (`mongodb_storage.py`). Web tasks cache Serper/fetch results under `.cache/browsecomp/` for repeatable browsing.

Agent-facing guidance for contributors: [`CLAUDE.md`](CLAUDE.md).

## Installation

### Prerequisites

- Python 3.8+
- **OpenRouter** API key for LLM calls ([openrouter.ai/keys](https://openrouter.ai/keys))
- **MongoDB** (optional but typical for experiment scripts that log runs)
- **Docker** (for MBPP / HumanEval-style code execution and deterministic repair)
- **Serper** API key for BrowseComp-class agents ([serper.dev](https://serper.dev)) — set `SERPER_API_KEY` in `.env`

### Setup

1. **Clone and enter the repo**
   ```bash
   git clone <repository-url>
   cd CausalFlow
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables**
   ```bash
   cp .env.example .env
   # Edit .env: at minimum OPENROUTER_SECRET_KEY; add MONGODB_URI, SERPER_API_KEY as needed.
   ```

   Required / common keys (see `.env.example`):

   - `OPENROUTER_SECRET_KEY` — all LLM traffic (code reads this name consistently)
   - `MONGODB_URI` — default `mongodb://localhost:27017/causalflow`
   - `SERPER_API_KEY` — web search for BrowseComp / SealQA / MedBrowseComp

## Quick start

**Examples (small traces, minimal dependencies):**
```bash
python examples/demo.py
python examples/complex_example.py
```

**Main experiments** (need `.env`; web tasks need `SERPER_API_KEY`; code tasks need Docker):
```bash
python experiments/gsm8k/run_gsm8k_experiment.py
python experiments/mbpp/run_mbpp_experiment.py
python experiments/humaneval/run_humaneval_experiment.py
python experiments/browsecomp/run_browsecomp_experiment.py
python experiments/browsecomp/run_sealqa_experiment.py
python experiments/browsecomp/run_medbrowsecomp_experiment.py
```

**Baseline comparison:**
```bash
python baseline_comparison/experiments/run_gsm8k_baselines.py --workers 8
python baseline_comparison/experiments/run_mbpp_baselines.py --workers 8
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --workers 4 --resume
python baseline_comparison/experiments/run_sealqa_baselines.py --workers 4
```

## Further reading

- **Experiment details and results snapshot:** [`EXPERIMENTS_OVERVIEW.md`](EXPERIMENTS_OVERVIEW.md)
- **Baseline methodology:** [`baseline_comparison/baseline_comparison.md`](baseline_comparison/baseline_comparison.md)
- **Sample textual report artifact:** `causalflow_report.txt` (from `CausalFlow.generate_full_report` in the library)

## Citation


