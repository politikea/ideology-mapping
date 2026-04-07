# Methodology

This document describes the full methodology pipeline used in Politikea Phase 1.
For the full analysis with results, see [PAPER.pdf](../PAPER.pdf) (arXiv submission pending).
For the axis definitions, see `docs/axes.md` and `prompts/label_8axis_v1.txt`.

---

## Overview

1. Proposals are collected from the Politikea platform (Spanish, 20–200 words each)
2. Each proposal is scored by the primary GPT model across 8 ideological axes, N≈13 times
3. Multi-run scores are aggregated and filtered for reliability
4. The resulting clean labels are analyzed for dimensionality, semantic validity, and cross-model agreement

---

## Step 1: Scoring

**Model**: the primary GPT model with structured JSON output enforced by the API.

**Prompt**: `prompts/label_8axis_v1.txt` — passed verbatim with the proposal text substituted at `{{TEXT}}` and the JSON schema at `{{JSON_SCHEMA}}`.

**Runs per proposal**: mean 13.1 (range: 10–20). Multiple runs per proposal are essential — they give:
- A stability estimate per axis (std, sign agreement)
- An ICC(2,1) reliability measure
- A validity flag for filtering

**Output schema per run** (flat structure — see `prompts/label_8axis_v1_schema.json` for the full JSON Schema):
```json
{
  "axis_aequitas_libertas": -35.0,
  "axis_imperium_anarkhia": 10.0,
  "...": "(...6 more axis scores, each -100 to +100)",
  "conf_axis_aequitas_libertas": 0.8,
  "conf_axis_imperium_anarkhia": 0.5,
  "...": "(...6 more per-axis confidences, each 0 to 1)",
  "global_confidence": 0.75,
  "flags": [],
  "rationale_spans": [
    {"axis": "axis_aequitas_libertas", "start": 12, "end": 58}
  ]
}
```

---

## Step 2: Cleaning (`cli.py clean`)

**Inputs**: consolidated multi-run annotations (one row per run per proposal)

**Process**:
1. Filter runs by confidence: global ≥ 0.5, per-axis ≥ 0.4
2. Aggregate per item: mean and std per axis, ICC(2,1) per axis, sign agreement
3. Flag item as valid if:
   - ≥ 10 runs completed
   - ≥ 6 of 8 axes are stable (std ≤ 30.0 AND sign agreement ≥ 0.6)

**Outputs**: `labels_clean.parquet` — one row per proposal with aggregated scores and validity flag

**Typical result**: 70–85% of proposals pass the validity gate. See the paper for Phase 1 specific numbers.

**Threshold sensitivity**: results are robust across a wide range of threshold configurations. The paper (Appendix B) summarises the full sensitivity grid.

---

## Step 3: Semantic Validation (`cli.py validate`)

**What it measures**: does text similarity predict ideological (8D) similarity?

**Method**: Spearman correlation between pairwise cosine similarity in text-embedding space and pairwise cosine similarity in 8D ideological space, across 500 randomly sampled pairs.

**Text embeddings**: `paraphrase-multilingual-mpnet-base-v2` (or any sentence-transformers model).

**8D similarity**: cosine similarity of the 8-dimensional axis-mean vectors.

**Two directions**:
1. Primary (text → 8D): build text-space nearest-neighbor graph, compute 8D similarity for each pair
2. Diagnostic (8D → text): build 8D nearest-neighbor graph, compute text similarity for each pair

**Key finding**: global ρ is positive but misleading. Within-category ρ is much lower — the global signal is confounded by category. The joint filter (text ≥ 0.70 AND 8D ≥ 0.70) is the reliable signal — it eliminates both noise modes simultaneously. See the paper for specific values.

---

## Step 4: Dimensionality (`cli.py dimensionality`)

**What it measures**:
1. PCA: how many components are needed to explain the 8D variance?
2. Axis collinearity: VIF for each axis
3. 2-axis reconstruction: which 2-axis pair best reconstructs the full 8D space?
4. (Optional) Predictive AUC: how much signal is lost at 1D vs 2D vs 8D for vote prediction?

**PCA method**: standardized PCA on the 8 axis-mean columns across all valid proposals.

**Reconstruction R²**: for each of the 28 2-axis pairs, fit Ridge regression from the 2-axis subspace to all 8 axes, compute mean R². This measures how much of the full ideological structure is captured by each pair.

Run `cli.py dimensionality` on your data to get corpus-specific PCA and R² results. See the paper for Phase 1 benchmarks.

<img src="figures/axis_r2_heatmap.png" width="690" alt="R² heatmap — all 2-axis pairs">

*Reconstruction R² for all 28 two-axis pairs (rows) predicting all 8 axes (columns). Green = high fidelity, red = poor reconstruction.*

---

## Step 5: Cross-Model Triangulation (advanced)

Available via `cli.py triangulate`. Requires audit model labels (see `docs/calibration-protocol.md`).

**Protocol**:
1. Draw stratified sample of N proposals
2. Re-score each with audit model (3 repeats, temperature 0.2)
3. Compute directional agreement and MAE vs. primary labels
4. Gate: directional agreement ≥ 0.85 AND MAE ≤ 15

**Phase 1 audit models**:
- Gemini 2.5 Flash (399 items): FAIL — systematic positivity bias, correctable
- llama3.1:8b (310 items): FAIL — task overload + coarse quantization, structurally different failure

**Key finding**: AI models carry model-specific calibration biases detectable only through cross-model comparison. Single-model annotation without this audit is methodologically insufficient.

**On few-shot examples**: Phase 1 prompt ablation (Variant E) found that naive few-shot examples *hurt* cross-model agreement — the audit model pattern-matched on the tone of the examples rather than using them as scale anchors. Few-shot calibration remains a promising path to cross-model alignment, but requires *calibration-targeted* examples (anchoring the scale endpoints and polarity per axis) rather than illustrative ones. This is an open research direction; see the paper's Future Work section.

---

## Reproducibility

This toolkit enables methodology replication, not results replication. You can:
- Apply the same 8-axis prompt to your own proposals
- Run the same cleaning, validation, and dimensionality pipeline
- Compare your findings with Phase 1 numbers (see `docs/results_summary.md`)

You cannot replicate Phase 1 exact numbers without the private Politikea corpus. This is intentional and documented.
