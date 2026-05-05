# CausalFlow DPO Fine-Tuning Proposal
**Experiment: Training a Small Open Model on CausalFlow-Generated Supervision**

---

## Motivation

A reviewer concern anticipated for the new submission is that CausalFlow's repair pipeline
requires gold labels to flag failures and includes the gold answer inside the repair prompt,
raising two linked objections: (1) gold availability at deployment time is unrealistic, and
(2) placing gold inside the LLM prompt constitutes leakage that inflates repair quality.

The standard response to both objections in the alignment literature is the same: gold
is consumed at training-data-generation time, not at inference time. This is the exact
structure of every preference dataset used in RLHF and DPO, from InstructGPT to
Llama-2-Chat. CausalFlow's validated contrastive pairs sit in that same regime. The
proposed experiment operationalizes this argument with a concrete downstream result: a
small open model fine-tuned on CausalFlow supervision performs step-level repair at
inference with no gold access at all.

---

## Why DPO Specifically

Direct Preference Optimization (Rafailov et al., 2023, already cited in the paper) is the
appropriate fine-tuning objective for three reasons specific to this setting.

**Structural alignment with CausalFlow output.** CausalFlow's validated repairs naturally
produce contrastive pairs of the form (wrong step, corrected step) for the same input
context. DPO is designed exactly for this data format. No additional annotation or reward
modeling step is required.

**No reward model needed.** The alternative RLHF pipeline would require training a
separate reward model on CausalFlow pairs before running PPO. DPO collapses this into
a single fine-tuning pass, which is reproducible, cheaper, and makes the contribution
cleaner for the paper.

**Inference-time gold independence.** A DPO-trained model learns to produce corrected
steps from the trace context alone. Once fine-tuned, it receives no gold answer, no
success signal, and no failure label. This directly closes the reviewer's deployment
realism concern.

---

## Data: What We Are Taking from Previous Runs

Source run: `run_MBPP_2025-12-11T00:49:23.675982`
Model: GPT-5 Chat, deterministic Docker-based validation.

From the BSON analysis of your existing runs, we extracted **257 validated contrastive
pairs** from the MBPP benchmark. These are step-level pairs where:

- The **rejected** completion is the original step `si` that caused the agent to fail
- The **chosen** completion is the minimally-edited repair `si*` that was deterministically
  validated via Docker re-execution to flip the verifier outcome from failure to success

Pair statistics from the run:

| Field | Value |
|---|---|
| Total pairs | 257 |
| Step type: llm_response | 179 (69.6%) |
| Step type: reasoning | 78 (30.4%) |
| Average minimality score | 0.747 |
| Min minimality | 0.000 |
| Max minimality | 1.000 |
| Source failed traces | 448 |
| Source fixed traces | 201 (44.9% repair rate) |

We use only pairs where Docker re-execution confirmed `success_predicted = True`, meaning
every chosen completion in the training set is a deterministically validated repair. There
is no label noise from LLM-based outcome prediction in this subset.

MBPP is selected as the benchmark for this experiment for three reasons: it has
deterministic evaluation (Docker execution, no LLM grader), the highest data volume of
validated pairs, and the cleanest step structure (code generation steps with explicit
input/output semantics).

Train/test split: 90/10 stratified by step type, preserving the llm_response/reasoning
ratio in both splits. This yields approximately 231 training pairs and 26 held-out pairs.

---

## Model: What to Run

**Base model: `meta-llama/Llama-3.2-3B-Instruct`**

Rationale for this specific choice:

- 3B parameters is runnable on a single A100 or equivalent in under 4 hours with LoRA,
  making results reproducible without multi-GPU infrastructure
- The Instruct variant already follows the prompt format needed for step-level repair
  instructions without additional supervised fine-tuning warmup
- LLaMA 3.2 is recent enough to be credible in a 2026 submission but well-documented
  enough to be easily cited and reproduced
- If compute is constrained, `Llama-3.2-1B-Instruct` is a valid fallback with comparable
  architecture, at the cost of somewhat lower baseline performance

**Fine-tuning method: LoRA via TRL DPOTrainer**

Full fine-tuning on 3B is unnecessary given 257 training pairs and risks catastrophic
forgetting. LoRA with rank 16 applied to the attention layers is the standard setup for
this data regime.

Recommended hyperparameters as a starting point:

| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Learning rate | 5e-5 |
| Beta (DPO temperature) | 0.1 |
| Batch size | 4 |
| Gradient accumulation steps | 4 |
| Epochs | 3 |
| Max sequence length | 1024 |

---

## Evaluation Protocol

The fine-tuned model is evaluated on held-out failed MBPP traces with no gold in any
prompt. The model receives only the problem statement and prior trace context and must
produce a corrected step. Correctness is determined by Docker re-execution of the
intervened trace.

Primary metric: **repair rate on held-out failed traces** compared to the zero-shot
no-gold baseline from the existing ablation runs (which produced 0% validated repairs
because the no-gold pipeline cannot verify its own proposals without execution feedback).

The fair comparison is:

| Condition | Gold in prompt | Repair rate |
|---|---|---|
| CausalFlow (main) | Yes | 44.9% (MBPP) |
| No-gold ablation (zero-shot) | No | ~0% validated |
| DPO fine-tuned LLaMA 3B (proposed) | No | TBD |

A positive result (any validated repair rate above 0% without gold) demonstrates the
training-time supervision paradigm: gold is used to generate training data offline,
and the fine-tuned model then operates entirely without gold at inference. This closes
the deployment realism concern at the experimental level.

---

## What This Adds to the Paper

This experiment adds one paragraph and one table to Section 7.6 (Future Work, currently
vague) or a new Section 6.6, converting a speculative direction into a demonstrated
result. The argument becomes:

> CausalFlow-generated contrastive pairs constitute step-level preference data directly
> usable for DPO on a small open model. After fine-tuning on 205 MBPP pairs, LLaMA 3B
> achieves [X]% repair rate on held-out failed traces with no gold access at inference,
> demonstrating that gold dependency is a training-time rather than deployment-time
> requirement.

This is structurally identical to how the alignment literature handles the same objection
for every preference-tuned model, and it gives reviewers a concrete result rather than a
theoretical claim.

---

## Files Needed for Tomorrow

- `mbpp_dpo_pairs.json` - 257 validated pairs, located in DPO\mbpp_dpo_pairs.json
- TRL library: `pip install trl transformers peft`
- HuggingFace access token for LLaMA 3.2 gated weights

---

*Prepared April 29, 2026*
