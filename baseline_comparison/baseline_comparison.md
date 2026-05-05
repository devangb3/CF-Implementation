# Baseline Comparison

Compares two iterative-reasoning baselines against a Direct (single-pass CoT) baseline across the same benchmarks used in [`research/CausalFlow.pdf`](../research/CausalFlow.pdf) (§5): GSM8K, MBPP, SealQA Hard, MedBrowseComp. CausalFlow itself is evaluated via the main `experiments/` scripts, not this folder.

Results are saved as JSON to `baseline_comparison/results/` after every problem so runs can be inspected or resumed mid-way.

---

## Baselines

### Direct
Single-pass Chain-of-Thought — one LLM call, no refinement. Serves as the lower bound.

### Self-Refine
> Madaan et al., 2023 — https://arxiv.org/abs/2303.17651

Generate → feedback → refine loop:
- Initial generation, then up to `max_iter` rounds of (feedback + refinement)
- Stops early when the feedback LLM outputs `[STOP]` (model declares itself satisfied)
- ~1–9 LLM calls per problem (GSM8K / web tasks), ~1–7 calls per task (MBPP)

### Self-Reflection
> arXiv 2405.06682 — https://arxiv.org/pdf/2405.06682

Structured single-pass reflection on a wrong answer:
- Generate initial answer; if wrong, generate a 5-part reflection:
  1. Explanation of why wrong
  2. Error keywords
  3. Correct step-by-step reasoning
  4. Instructions for this problem type
  5. General advice
- Re-answer with reflection injected — gold answer is never revealed to the model
- 1 LLM call if initially correct, 3 calls otherwise

### Note on Tree of Thoughts
We implemented and ran a pilot of Tree of Thoughts (Yao et al., 2023 — https://arxiv.org/abs/2305.10601) as a third baseline. Preliminary results showed it underperformed Direct on GSM8K (48% vs 84% accuracy) despite consuming ~10× the LLM calls. BFS exploration over reasoning steps actively hurts sequential arithmetic reasoning rather than helping it. We exclude ToT from the full runs and comparison tables on this basis.

---

## Benchmarks, Models & Temperatures

| Benchmark | Model | Temperature | Notes |
|-----------|-------|-------------|-------|
| **GSM8K** | `google/gemini-2.0-flash-lite-001` | `0.7` all steps | Same model as CausalFlow GSM8K experiment |
| **MBPP** | `openai/gpt-5-chat` | `0.2` code generation / `0.7` feedback & reflection | Deterministic code eval via Docker |
| **MedBrowseComp** | `google/gemini-3-flash-preview` | `0.3` all agent steps | LLM grading via `grade_response` |
| **SealQA** | `google/gemini-3-flash-preview` | `0.0` solver (initial pass) / `0.3` refinement steps | Split temperature matches existing SealQA experiment |

---

## File Structure

```
baseline_comparison/
├── baselines/
│   ├── self_refine.py          # SelfRefine class
│   └── self_reflection.py      # SelfReflection class
├── experiments/
│   ├── run_gsm8k_baselines.py          # GSM8K — math reasoning
│   ├── run_mbpp_baselines.py           # MBPP — Python code generation
│   ├── run_medbrowsecomp_baselines.py  # MedBrowseComp — medical web QA
│   └── run_sealqa_baselines.py         # SealQA Hard — research web QA
└── results/                    # JSON output written here per run
```

---

## Design: Web QA Tasks (MedBrowseComp & SealQA)

All baselines share **one** `BrowseCompAgent` web-search run per problem. After the agent collects evidence and produces an initial answer, each refinement strategy re-processes the gathered context using LLM-only calls — no additional web searches.

This isolates the refinement strategy as the independent variable, making comparisons against CausalFlow's causal repair approach directly meaningful.

**SealQA temperature split:**
- `BrowseCompAgent` runs at T=0.0 (deterministic solver, identical to the CausalFlow SealQA experiment)
- Self-Refine / Self-Reflection refinement steps run at T=0.3

---

## Running the Experiments

### GSM8K
```bash
# Full run (1,319 problems)
python baseline_comparison/experiments/run_gsm8k_baselines.py --workers 8

# Resume an interrupted run
python baseline_comparison/experiments/run_gsm8k_baselines.py --resume --workers 8

# Specific baselines only
python baseline_comparison/experiments/run_gsm8k_baselines.py --baselines direct self_refine --workers 8
```

### MBPP
```bash
# Full run (~947 tasks) — requires Docker
python baseline_comparison/experiments/run_mbpp_baselines.py --workers 8

# Resume an interrupted run
python baseline_comparison/experiments/run_mbpp_baselines.py --resume --workers 8
```

### MedBrowseComp
```bash
# Full run (484 examples) — requires SERPER_API_KEY
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --workers 4

# Resume an interrupted run
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --resume --workers 4
```

### SealQA
```bash
# Full run (254 examples) — requires SERPER_API_KEY
python baseline_comparison/experiments/run_sealqa_baselines.py --workers 4

# Resume an interrupted run
python baseline_comparison/experiments/run_sealqa_baselines.py --resume --workers 4
```

---

## CLI Arguments (all experiments)

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_rows` / `--num_examples` | all | Take the first N problems (deterministic) |
| `--sample N` | — | Randomly sample N problems (seed=42). Overrides `--num_rows`/`--num_examples` |
| `--workers` | 1 | Parallel workers. Try `8` for GSM8K/MBPP, `4` for web tasks |
| `--resume [FILE]` | — | Resume from checkpoint. Omit FILE to auto-pick the latest result file |
| `--baselines` | all three | Space-separated subset: `direct self_refine self_reflection` |
| `--model` | benchmark default | OpenRouter model identifier |
| `--self_refine_max_iter` | 4 (GSM8K/web), 3 (MBPP) | Max refinement iterations for Self-Refine |
| `--max_steps` | 10–15 | Max web-browsing steps — MedBrowseComp / SealQA only |

---

## Output Format

Each run writes one JSON file to `baseline_comparison/results/`:

```
gsm8k_<timestamp>.json
mbpp_<timestamp>.json
medbrowsecomp_<timestamp>.json
sealqa_<timestamp>.json
```

**Top-level keys:**

```json
{
  "meta": {
    "experiment": "GSM8K Baseline Comparison",
    "model": "google/gemini-2.0-flash-lite-001",
    "num_problems": 1319,
    "baselines": ["direct", "self_refine", "self_reflection"],
    "started_at": "20260426_143000"
  },
  "summary": {
    "direct":          { "accuracy": 0.88, "correct": 44, "total": 50, "avg_llm_calls": 1.0, "avg_elapsed_seconds": 4.3 },
    "self_refine":     { "accuracy": 0.88, "correct": 44, "total": 50, "avg_llm_calls": 2.0, "avg_elapsed_seconds": 8.0 },
    "self_reflection": { "accuracy": 0.92, "correct": 46, "total": 50, "avg_llm_calls": 1.2, "avg_elapsed_seconds": 7.1 }
  },
  "problems": { ... }
}
```

The summary table is also printed to the terminal at the end of each run.

---

## Environment Variables Required

```
OPENROUTER_SECRET_KEY=sk-or-v1-...   # Required for all experiments
SERPER_API_KEY=...                    # Required for MedBrowseComp and SealQA
```
