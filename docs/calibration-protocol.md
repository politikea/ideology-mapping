# Cross-Model Calibration Protocol

This document describes how to audit your primary model's labels using a second model, detect calibration biases, and interpret the results.

---

## Why Cross-Model Triangulation?

AI models carry model-specific calibration biases that are consistent within a model but only detectable through comparison with a second model. A single model can produce highly reliable scores (high ICC, high sign agreement) that are systematically shifted relative to another model's interpretation of the same axes.

The triangulation protocol answers: *"Would a different model agree with these labels?"*

---

## Step 1: Choose an Audit Model

The audit model should differ from your primary model in architecture or training. Good choices:

| Primary model | Audit model options |
|---------------|-------------------|
| GPT-4o / GPT-5 | Gemini, Claude, llama (via Ollama or vLLM) |
| Claude | GPT-4o, Gemini |
| Local llama | GPT-4o (cloud) |

The audit model does not need to be as capable as the primary model. Its role is to provide an independent signal for comparison.

---

## Step 2: Draw a Stratified Sample

You do not need to re-score your entire corpus. A stratified sample of 200-400 items is sufficient.

Stratify by:
- **Policy category** (if available) — ensures coverage across topics
- **Confidence quantile** — ensures you test both high-confidence and borderline items

```bash
# The triangulation module handles sampling internally, or you can pre-sample:
python -c "
import pandas as pd
from analysis.triangulation import stratified_subset

labels = pd.read_parquet('labels_clean.parquet')
items = pd.read_csv('proposals.csv')
sample = stratified_subset(labels, items, sample_n=400)
sample.to_parquet('triangulation_sample.parquet', index=False)
"
```

---

## Step 3: Score with the Audit Model

Re-score each sampled proposal with the audit model. Use the same prompt template.

For OpenAI-compatible APIs, use `scripts/score_proposals.py`:

```bash
python scripts/score_proposals.py \
    --input triangulation_sample.parquet \
    --output audit_annotations.parquet \
    --model llama3.1:8b \
    --base-url http://localhost:11434/v1 \
    --api-key ollama \
    --n-runs 3 \
    --temperature 0.2
```

Then clean the audit labels:

```bash
python cli.py clean \
    --input audit_annotations.parquet \
    --output audit_labels_clean.parquet \
    --min-runs 3
```

---

## Step 4: Run Triangulation

```bash
python cli.py triangulate \
    --reference labels_clean.parquet \
    --audit audit_labels_clean.parquet \
    --output-dir triangulation_results/
```

This computes per-axis directional agreement and MAE, then applies the publication gate.

---

## Step 5: Interpret the Results

### Gate thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Mean directional agreement | >= 0.85 | At least 85% of items should have same-sign scores |
| Mean MAE | <= 15.0 | Average magnitude difference across all axes |
| Max axis MAE | <= 20.0 | No single axis should diverge more than 20 points on average |

### Common failure patterns

**Acquiescence bias** (e.g., Gemini):
- Directional agreement is moderate (~0.6-0.7) but not catastrophic
- MAE is high, driven by a systematic offset in one direction
- All or most audit scores are positive (or all negative)
- **Correctable** via per-axis offset normalization

**Task overload** (e.g., small llama models):
- High instability across runs (large within-item std)
- Scores clustered at round numbers (0, ±10, ±50, ±100)
- Low directional agreement on complex axes
- **Partially correctable** via prompt simplification or task decomposition (see `docs/prompt-engineering.md`)

**Axis confusion**:
- One or two axes have very low directional agreement while others are fine
- The audit model may be interpreting those axes differently
- Check per-axis metrics in the report

### If the gate fails

A gate failure does not mean your primary labels are wrong. It means the two models disagree beyond the threshold. Options:

1. **Accept with caveat**: report that cross-model validation was attempted and note the disagreement
2. **Calibrate**: apply offset correction to the audit model and re-evaluate
3. **Investigate**: examine the highest-disagreement items to understand the pattern
4. **Try a different audit model**: the failure may be model-specific

---

## Advanced: Calibration Correction

If the audit model shows a systematic offset (acquiescence bias), you can attempt correction:

```python
import pandas as pd
from analysis.label_io import AXES

ref = pd.read_parquet("labels_clean.parquet")
audit = pd.read_parquet("audit_labels_clean.parquet")
merged = ref.merge(audit, on="item_id", suffixes=("_ref", "_audit"))

for axis in AXES:
    ref_col = f"{axis}_mean_ref"
    audit_col = f"{axis}_mean_audit"
    offset = merged[audit_col].mean() - merged[ref_col].mean()
    print(f"{axis}: offset = {offset:+.1f}")
    # Apply correction:
    audit[f"{axis}_mean"] -= offset
```

After correction, re-run `cli.py triangulate` to see if the gate passes. If it does, the bias was systematic and correctable.

---

*See the paper for the full Phase 1 triangulation results and discussion.*
