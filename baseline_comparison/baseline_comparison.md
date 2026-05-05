# Baseline Comparison

Compares three iterative-reasoning baselines against a Direct (single-pass CoT) baseline across four benchmarks. Results are saved as JSON to `baseline_comparison/results/` after every problem so runs can be inspected or resumed mid-way.

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

### Tree of Thoughts (BFS)
> Yao et al., 2023 — https://arxiv.org/abs/2305.10601

Explores a beam of intermediate reasoning states:
- Depth T=3, beam width b=2, k=3 candidates per state
- Each level: propose k thoughts per surviving state (1 call), batch-evaluate (1 call), keep top-b
- Final: generate answer from each surviving chain, select by cumulative score
- ~12 LLM calls per problem (math/web), ~5 calls per task (MBPP code strategies)

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
│   ├── self_refine.py          # SelfRefine class (GSM8K)
│   ├── self_reflection.py      # SelfReflection class (GSM8K)
│   └── tree_of_thoughts.py     # TreeOfThoughts BFS class (GSM8K)
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
- Self-Refine / Self-Reflection / Tree-of-Thoughts refinement steps run at T=0.3

---

## Running the Experiments

### GSM8K
```bash
# Random 50-problem pilot (fast, reproducible seed=42)
python baseline_comparison/experiments/run_gsm8k_baselines.py --sample 50 --workers 8

# Full run, all 4 baselines (1 319 problems)
python baseline_comparison/experiments/run_gsm8k_baselines.py --workers 8

# Resume an interrupted run
python baseline_comparison/experiments/run_gsm8k_baselines.py --resume --workers 8

# Specific baselines only
python baseline_comparison/experiments/run_gsm8k_baselines.py --baselines direct self_refine --sample 100
```

### MBPP
```bash
# Random 50-task pilot — requires Docker
python baseline_comparison/experiments/run_mbpp_baselines.py --sample 50 --workers 8

# Full run (~947 tasks)
python baseline_comparison/experiments/run_mbpp_baselines.py --workers 8

# Resume an interrupted run
python baseline_comparison/experiments/run_mbpp_baselines.py --resume --workers 8
```

### MedBrowseComp
```bash
# Random 50-example pilot — requires SERPER_API_KEY
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --sample 50 --workers 4

# Full run (484 examples)
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --workers 4

# Resume an interrupted run
python baseline_comparison/experiments/run_medbrowsecomp_baselines.py --resume --workers 4
```

### SealQA
```bash
# Random 50-example pilot — requires SERPER_API_KEY
python baseline_comparison/experiments/run_sealqa_baselines.py --sample 50 --workers 4

# Full run (254 examples)
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
| `--baselines` | all four | Space-separated subset: `direct self_refine self_reflection tree_of_thoughts` |
| `--model` | benchmark default | OpenRouter model identifier |
| `--self_refine_max_iter` | 4 (GSM8K/web), 3 (MBPP) | Max refinement iterations for Self-Refine |
| `--tot_candidates` | 3 | Tree of Thoughts: candidate thoughts/strategies per state (k) |
| `--tot_beam` | 2 | Tree of Thoughts: beam width between BFS levels (b) |
| `--tot_depth` | 3 | Tree of Thoughts: BFS depth (T) — GSM8K only |
| `--max_steps` | 10–15 | Max web-browsing steps — MedBrowseComp / SealQA only |

---

## Summary Tables (50-example pilot, seed=42)

### Table 1 — Repair performance per method across benchmarks

Repair Rate is computed over initially-failed traces only. Minimality measures position-wise token similarity between original and repaired output (equation 3; higher = smaller edit). Direct and Tree of Thoughts have no repair loop so their rows are omitted.

| Method | Benchmark | Total | Passed | Failed | Repairs | Min. |
|---|---|---|---|---|---|---|
| Self-Refine | GSM8K | 50 | 42 | 8 | 0 (0.0%) | 1.000 |
| Self-Reflection | GSM8K | 50 | 39 | 11 | 4 (36.4%) | 0.801 |
| Self-Refine | MBPP | 50 | 23 | 27 | 18 (66.7%) | 0.635 |
| Self-Reflection | MBPP | 50 | 24 | 26 | 12 (46.2%) | 0.614 |
| Self-Refine | MedBrowseComp | 50 | 13 | 37 | 2 (5.4%) | 0.790 |
| Self-Reflection | MedBrowseComp | 50 | 13 | 37 | 3 (8.1%) | 0.677 |
| Tree of Thoughts | MedBrowseComp | 50 | 13 | 37 | 5 (13.5%) | 0.397 |
| Self-Refine | SealQA | 50 | 17 | 33 | 6 (18.2%) | 0.100 |
| Self-Reflection | SealQA | 50 | 17 | 33 | 8 (24.2%) | 0.000 |
| Tree of Thoughts | SealQA | 50 | 17 | 33 | 8 (24.2%) | 0.183 |

> Tree of Thoughts has no discrete initial attempt on GSM8K/MBPP (BFS from scratch), so Passed/Failed/Repairs are not applicable. For web tasks (MedBrowseComp/SealQA) all baselines share the same agent run, so initial Passed/Failed are identical across methods.

---

### Table 2 — Direct (baseline) accuracy vs post-refinement accuracy per method

Post-Repair Accuracy reflects applying each refinement strategy to the agent's initial answer. Direct = no refinement (lower bound).

| Method | GSM8K | MBPP | MedBrowseComp | SealQA |
|---|---|---|---|---|
| Direct | 84.0% | 46.0% | 26.0% | 34.0% |
| Self-Refine | 84.0% | **82.0%** | 24.0% | 32.0% |
| Self-Reflection | **86.0%** | 72.0% | 26.0% | **36.0%** |
| Tree of Thoughts | 48.0% | 50.0% | **32.0%** | 32.0% |

---

## Pilot Results (50 random samples, seed=42)

### GSM8K — `google/gemini-2.0-flash-lite-001`, T=0.7

| Baseline | Accuracy | Correct | Total | Repair Rate | Minimality | Avg Calls | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Direct | 84.0% | 42 | 50 | 0% | 1.000 | 1.0 | 4.3 |
| Self-Refine | 84.0% | 42 | 50 | 0% | 1.000 | 2.0 | 8.0 |
| Self-Reflection | **86.0%** | 43 | 50 | **36.4%** | **0.801** | 1.4 | 7.1 |
| Tree of Thoughts | 48.0% | 24 | 50 | N/A | N/A | 10.1 | 32.5 |

> **Repair Rate** = fraction of initially-wrong answers successfully corrected. N/A for Tree of Thoughts (no discrete "initial attempt" in BFS).
> **Minimality** = position-wise token similarity between initial and repaired solution per equation (3) (1.0 = identical = maximally minimal edit). N/A for ToT.

Key findings:
- Self-Reflection is the only method that repairs wrong answers (36.4% repair rate) with surgical edits (0.801)
- Self-Refine never fires: the feedback model outputs `[STOP]` on wrong solutions (avg 2.0 calls = 1 generate + 1 feedback that halts), so no refinement happens and repair rate is 0%
- Tree of Thoughts collapses to 48% — BFS exploration introduces noise on sequential arithmetic despite 10× the calls

### MBPP — `openai/gpt-5-chat`, T=0.2 code / T=0.7 reasoning

| Baseline | Accuracy | Correct | Total | Repair Rate | Minimality | Avg Calls | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Direct | 46.0% | 23 | 50 | 0% | 1.000 | 1.0 | 1.6 |
| Self-Refine | **82.0%** | 41 | 50 | **66.7%** | 0.635 | 3.1 | 6.0 |
| Self-Reflection | 72.0% | 36 | 50 | 46.2% | **0.614** | 2.0 | 4.7 |
| Tree of Thoughts | 50.0% | 25 | 50 | N/A | N/A | 3.9 | 6.1 |

Key findings:
- Self-Refine dominates code (82%, 66.7% repair) — exact Docker test failure output gives precise, actionable feedback
- Self-Reflection (72%, 46.2% repair) falls behind: one abstract reflection pass is less precise than seeing real test errors
- Lower minimality on both methods (0.635 / 0.614) vs. GSM8K reflects that code repairs tend to be more extensive rewrites
- ToT (50%) barely beats Direct (46%) — abstract strategy proposals don't overcome implementation-level bugs

### MedBrowseComp — `google/gemini-3-flash-preview`, T=0.3

| Baseline | Accuracy | Correct | Total | Repair Rate | Minimality | Avg Calls | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Direct | 26.0% | 13 | 50 | 0% | 1.000 | — | 1.2 |
| Self-Refine | 24.0% | 12 | 50 | 5.4% | 0.790 | 2.1 | 5.8 |
| Self-Reflection | 26.0% | 13 | 50 | 8.1% | 0.677 | 2.0 | 7.5 |
| Tree of Thoughts | **32.0%** | **16** | 50 | **13.5%** | 0.397 | 2.0 | 4.2 |

> Avg Calls = refinement-only calls (agent web-browsing calls not counted in this column — see design note).

Key findings:
- Tree of Thoughts is the best on web QA (32%) — exploring multiple answer interpretations of the gathered evidence outperforms single-pass refinement
- Self-Refine **hurts** accuracy (24% < 26% direct) — iterative LLM feedback on ambiguous medical evidence leads the model away from the correct answer
- All repair rates are low (5–14%) because the binding constraint is evidence quality, not reasoning ability; LLM-only refinement without new web searches has limited headroom

### SealQA — `google/gemini-3-flash-preview`, T=0.0 solver / T=0.3 refinement

| Baseline | Accuracy | Correct | Total | Repair Rate | Minimality | Avg Calls | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Direct | 34.0% | 17 | 50 | 0% | 1.000 | — | 1.1 |
| Self-Refine | 32.0% | 16 | 50 | 18.2% | 0.100 | 4.5 | 18.5 |
| Self-Reflection | **36.0%** | **18** | 50 | **24.2%** | 0.000 | 2.0 | 9.7 |
| Tree of Thoughts | 32.0% | 16 | 50 | 24.2% | 0.183 | 2.0 | 4.2 |

Key findings:
- No method beats Direct by more than 2% — all methods are bottlenecked by the quality of collected web evidence
- Near-zero minimality scores (0.000–0.183) reveal that "repairs" are full answer replacements, not targeted edits; for short precise answers the position-wise formula is 0 when even one token differs at each position
- Self-Refine burns 4.5 calls on average and still ends up below Direct — iterative feedback loops are counterproductive without new web searches
- Self-Reflection edges ahead by 1 correct answer (36% vs 34%) at the cost of completely replacing the initial answer

---

## Output Format

Each run writes one JSON file to `baseline_comparison/results/`:

```
gsm8k_20260426_143000.json
mbpp_20260426_143000.json
medbrowsecomp_20260426_143000.json
sealqa_20260426_143000.json
```

**Top-level keys:**

```json
{
  "meta": {
    "experiment": "GSM8K Baseline Comparison",
    "model": "google/gemini-2.0-flash-lite-001",
    "num_problems": 1319,
    "baselines": ["direct", "self_refine", "self_reflection", "tree_of_thoughts"],
    "started_at": "20260426_143000"
  },
  "summary": {
    "direct":          { "accuracy": 0.88, "correct": 44, "total": 50, "avg_llm_calls": 1.0 },
    "self_refine":     { "accuracy": 0.88, "correct": 44, "total": 50, "avg_llm_calls": 2.0 },
    "self_reflection": { "accuracy": 0.92, "correct": 46, "total": 50, "avg_llm_calls": 1.2 },
    "tree_of_thoughts":{ "accuracy": 0.66, "correct": 33, "total": 50, "avg_llm_calls": 12.0 }
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
MONGODB_URI=...                       # Not needed — baselines save to JSON only
```
