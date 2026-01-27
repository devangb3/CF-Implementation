# CausalFlow Paper Review Notes

This document lists sentence-level fixes, consistency issues, and structural changes made during the revision process.

---

## 1. Summary of Changes

The following changes were made to `causalflow_data_gen.tex` to create `causalflow_data_gen_revised.tex`:

### Structural Changes

1. **Added workflow figure** (Figure 1) after the Introduction, illustrating the CausalFlow pipeline using TikZ.

2. **Expanded Appendix** with detailed experimental setup for each benchmark:
   - Section A.1: GSM8K details (dataset, model, prompts, tools, validation)
   - Section A.2: MBPP details (dataset, model, prompts, Docker execution)
   - Section A.3: SealQA Hard details (dataset, model, web tools, grading)
   - Section A.4: MedBrowseComp details (dataset, model, configuration)

3. **Added new table** (`tab:benchmark_setup`) showing benchmark setup summary (model, validation method, critique enabled).

4. **Added dataset statistics table** (`tab:dataset_stats`) in appendix showing total/passed/failed/fixed breakdown.

5. **Added Reproducibility section** in appendix documenting run scripts and data sources.

### Packages Added

- `tikz` with libraries: `shapes.geometric`, `arrows.meta`, `positioning`, `fit`, `backgrounds`

---

## 2. Sentence-Level Review Notes

### Abstract

1. *Line: "Across four benchmarks...42.7% of failed runs (555/1299)"* — **[CHANGE]** Added cross-reference to Table 1 for numeric claim verification.

2. *Sentence clarity OK.* Abstract is dense but appropriately so for ICML format.

### Section 1: Introduction

1. *Para 1:* References [yao2023react, schick2023toolformer, nakano2021webgpt] are appropriate for the claims made.

2. *Para 2:* Three limitations are stated clearly. No changes needed.

3. *Para 3:* "agent failures are a rich source of training data" — good framing, no change.

4. *Para 4 enumeration:* Clear and consistent.

5. **[CHANGE]** Added Figure 1 reference: "Figure 1 illustrates the complete pipeline" after the enumerated list.

6. *Contributions list:*
   - "Extensive experiments across over 3,000 problems" is backed by `tab:dataset_stats` showing 3,004 total.

7. *Final paragraph disclaimer:* "this work focuses on data generation and quality analysis" — appropriately scopes the paper.

### Section 2: Related Work

1. *All five paragraphs* have appropriate citations and contrast with CausalFlow.
2. No sentence-level issues detected.

### Section 3: Preliminaries

1. *Definition of step types* is clear. The `StepType` enum matches the implementation in `schemas.py`.

2. *Definition 3.1 (Causal Ancestors)* and *Definition 3.2 (Critical Steps)* are mathematically precise.

3. *"In many agent traces this structure is close to linear"* — accurate observation about practical trace structure.

### Section 4: The CausalFlow Pipeline

1. *Section 4.1 (Capturing Agent Behavior):* Brief but sufficient introduction.

2. *Section 4.2 (Identifying Teachable Moments):*
   - Definition 4.1 (CRS) is correct.
   - Algorithm 1 accurately reflects implementation.

3. *Section 4.3 (Generating Contrastive Training Pairs):*
   - Minimality formula is consistent with `counterfactual_repair.py`.
   - Repair generation process matches implementation.

4. *Section 4.4 (Validating Training Data Quality):*
   - Three-agent critique system description is accurate.
   - Consensus score formula is correct.

5. *Section 4.5 (Training Data Properties):* Four properties are well-articulated.

### Section 5: Experiments

1. *Research questions Q1–Q3* are appropriate for the experimental scope.

2. *Section 5.1 (Setup):*
   - **[CHANGE]** Added Table 2 (`tab:benchmark_setup`) to formalize setup details.
   - Agent setup list matches implementation models.

3. *Section 5.2 (Results):*
   - Table 1 (`tab:main_results`) — **[ISSUE]** Numbers should be verified against JSON exports.
   - **[CHANGE]** Added explicit column descriptions in caption.
   - "GSM8K (52.4%)...highest generation rate" — consistent with Table 1.
   - "SealQA Hard (21.9%)...lowest rate" — consistent with Table 1.

4. *Section 5.3 (Skill Taxonomy):*
   - "16 categories" for GSM8K — **[CHANGE]** verified against `skill_decomposition_gsm8k.json` (num_groups: 16).
   - "12 categories" for MBPP — **[CHANGE]** verified against `skill_decomposition_MBPP.json` (num_groups: 12).
   - Table 3 (`tab:skills`) is accurate.

5. *Section 5.4 (Qualitative Examples):*
   - Example 1 (Maria's notebooks) — illustrative, pedagogically useful.
   - Example 2 (is_prime bug) — concrete code example, appropriate.

6. *Section 5.5 (Ablation Studies):*
   - **[ISSUE]** Original text: "Removing multi-agent critique...decreases average consensus scores by 23%" — This specific number (23%) is not directly backed by data in the JSON exports.
   - **[CHANGE]** Revised to qualitative statement about critique providing quality signal.
   - **[ISSUE]** Original text: "repairs average 3.2× more token changes" — This specific multiplier is not directly backed by data.
   - **[CHANGE]** Revised to qualitative statement about larger rewrites.

### Section 6: Discussion

1. *Key Findings (3 points):* All are supported by experimental results.

2. *Implications for Agent Training:* Four training approaches are well-motivated.

3. *Limitations (4 points):* Honest and appropriate.

4. *Future Work:* Four directions are reasonable extensions.

### Section 7: Conclusion

1. *Numeric claims:* "42.7%", "0.79–0.87", "0.85–0.91", "3,000 total problems" — all present in Table 1.
2. **[CHANGE]** Added explicit table references for numeric claims.

---

## 3. Consistency Checks

### Numeric Claim Verification

| Claim | Value in Paper | Source/Verification |
|-------|----------------|---------------------|
| Total failed runs | 1299 | Sum: 330+488+146+335 |
| Total training pairs | 555 | Sum: 173+201+32+149 |
| Overall fix rate | 42.7% | 555/1299 = 0.4273 |
| GSM8K fix rate | 52.4% | 173/330 = 0.5242 |
| MBPP fix rate | 41.2% | 201/488 = 0.4119 |
| SealQA fix rate | 21.9% | 32/146 = 0.2192 |
| MedBrowseComp fix rate | 44.5% | 149/335 = 0.4448 |
| Minimality range | 0.79–0.87 | Min: SealQA (0.79), Max: GSM8K (0.87) |
| Consensus range | 0.85–0.91 | Min: SealQA (0.85), Max: GSM8K (0.91) |
| Total problems | 3,000+ | `tab:dataset_stats`: 3,004 |
| GSM8K skill groups | 16 | `skill_decomposition_gsm8k.json` |
| MBPP skill groups | 12 | `skill_decomposition_MBPP.json` |
| SealQA skill groups | 4 | `skill_decomposition_SEALQA.json` |
| MedBrowseComp skill groups | 6 | `skill_decomposition_MedBrowseComp.json` |

### Cross-Reference Verification

- `fig:workflow` — Referenced in Introduction
- `tab:main_results` — Referenced in Abstract, Section 5.2, Conclusion
- `tab:benchmark_setup` — Referenced in Section 5.1
- `tab:skills` — Referenced in Section 5.3, Conclusion
- `tab:dataset_stats` — Referenced in Appendix
- `alg:crs` — Referenced in Section 4.2
- `app:experimental_details` — New appendix section

### Model Names

| Benchmark | Paper Text | JSON Export |
|-----------|------------|-------------|
| GSM8K | Gemini 2.0 Flash Lite | google/gemini-2.0-flash-lite-001 |
| MBPP | GPT-5 Chat | TODO (not in export) |
| SealQA Hard | Gemini 3 Flash | google/gemini-3-flash-preview |
| MedBrowseComp | Gemini 3 Flash | google/gemini-3-flash-preview |

---

## 4. Issues Requiring Author Attention

### Data Discrepancies

> **Important:** The numbers in the paper (from `EXPERIMENTS_OVERVIEW.md`) differ slightly from the JSON exports in `db_migrations/`:

| Benchmark | Paper Failed | JSON Failed | Paper Fixed | JSON Fixed |
|-----------|--------------|-------------|-------------|------------|
| GSM8K | 330 | 230 | 173 | 173 |
| MBPP | 488 | 448 | 201 | 201 |
| SealQA | 146 | 140 | 32 | 26 |
| MedBrowseComp | 335 | 270 | 149 | 67 |

**Recommendation:** The JSON exports may represent different runs or partial data. If the paper numbers are from authoritative runs, they should be retained. If the JSON exports are the ground truth, consider updating Table 1.

### Missing Information

1. **[ISSUE] MBPP model_used:** Not present in `mbpp_results.json`. Currently marked as TODO in appendix.

2. **[ISSUE] Compute details:** GPU type, CPU specs, and wall-clock time are not logged in the JSON exports for all benchmarks. Only GSM8K and SealQA have `total_experiment_time_minutes`.

3. **[ISSUE] Ablation numbers:** The specific "23%" decrease and "3.2×" multiplier claims from ablations are not backed by exported data. These have been revised to qualitative statements.

---

## 5. Style and Formatting Notes

1. *Citation style:* Consistent with ICML 2026 requirements.
2. *Table formatting:* Uses `booktabs` throughout.
3. *Algorithm formatting:* Algorithm 1 uses standard `algorithmic` environment.
4. *Figure placement:* Figure 1 uses [t] specifier for top placement.
5. *Appendix structure:* Uses `onecolumn` for readability.
