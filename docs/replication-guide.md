# Replication Guide

This guide walks you through scoring your own political proposals and running the full analysis pipeline, from raw text to publishable results.

---

## Prerequisites

```bash
git clone https://github.com/politikea/ideology-mapping.git
cd ideology-mapping
pip install -e .                   # installs all analysis dependencies
pip install -e ".[scoring]"        # also installs openai SDK (needed for scoring step)
```

You will need:
- Python 3.9+
- An API key for an OpenAI-compatible LLM (GPT-4o, GPT-5, Azure OpenAI, or local vLLM/Ollama) — only for the scoring step
- A dataset of political proposals (CSV or parquet with at least a `text` column)

---

## Step 1: Prepare Your Data

Your input file should have at minimum:
- `item_id` — unique identifier per proposal (generated automatically if missing)
- `text` — the full text of each proposal

Optional but recommended:
- `category` — policy category (e.g., "Economy", "Health", "Security")

```
item_id,text,category
prop_001,"Increase the minimum wage to $25/hour",Economy
prop_002,"Mandate body cameras for all police officers",Security
...
```

Aim for 100+ proposals. The methodology works best with 200-1000 items spanning multiple policy categories.

---

## Step 2: Score Proposals

Score each proposal N=13 times with the 8-axis prompt:

```bash
python scripts/score_proposals.py \
    --input my_proposals.csv \
    --output annotations.parquet \
    --model claude-sonnet-4-20250514 \
    --n-runs 13 \
    --temperature 0.2 \
    --cache-dir cache/labels/
```

This produces a parquet with one row per (item, run) — approximately 13x the number of proposals.

**Cost estimate**: ~400 tokens per call. For 200 proposals x 13 runs = 2,600 calls. At GPT-4o rates this is roughly $2-5 USD.

**Caching**: use `--cache-dir` to avoid re-scoring on restarts. The cache is keyed by item text + model + run index.

---

## Step 3: Clean and Aggregate Labels

```bash
python cli.py clean \
    --input annotations.parquet \
    --output labels_clean.parquet
```

This filters by confidence, aggregates across runs, computes ICC per axis, and flags valid items. You should see 70-85% of items passing the validity gate. If significantly fewer pass, check your model's temperature setting and prompt compliance.

---

## Step 4: Validate (Semantic Consistency)

```bash
python cli.py validate \
    --labels labels_clean.parquet \
    --items my_proposals.csv \
    --output-dir results/ \
    --text-col text
```

This runs the H1-H4 validation suite:
- **H1/H2**: Text-8D Spearman correlation (global and within-category)
- **H3**: Per-axis linguistic encoding strength
- **H4**: Proposal deduplication via joint text+8D filtering

Output: `results/validation_report.md`

---

## Step 5: Dimensionality Analysis

```bash
python cli.py dimensionality \
    --labels labels_clean.parquet \
    --output-dir results/ \
    --skip-predictive
```

This runs PCA, VIF collinearity analysis, and the 28-pair reconstruction R² ranking.

Output: `results/dimensionality_report.md`

---

## Step 6: Insights

```bash
python cli.py insights \
    --labels labels_clean.parquet \
    --items my_proposals.csv \
    --output-dir results/
```

This computes category centroids, axis correlation matrix, VIF table, and K-means clustering.

Output: `results/insights_report.md`

---

## Step 7: Cross-Model Triangulation (Optional)

Score a subset with a second model and compare:

```bash
# Score sample with audit model
python scripts/score_proposals.py \
    --input my_proposals.csv \
    --output audit_annotations.parquet \
    --model llama3.1:8b \
    --base-url http://localhost:11434/v1 \
    --api-key ollama \
    --n-runs 3

# Clean audit labels
python cli.py clean \
    --input audit_annotations.parquet \
    --output audit_labels_clean.parquet \
    --min-runs 3

# Compare
python cli.py triangulate \
    --reference labels_clean.parquet \
    --audit audit_labels_clean.parquet \
    --output-dir results/
```

Output: `results/triangulation_report.md`

See `docs/calibration-protocol.md` for how to interpret and correct failures.

---

## Step 8: Visualize

```python
from analysis.visualize import correlation_heatmap, pca_biplot, score_distributions
import pandas as pd

labels = pd.read_parquet("labels_clean.parquet")

fig = correlation_heatmap(labels)
fig.savefig("results/correlation_heatmap.png", dpi=150, bbox_inches="tight")

fig = pca_biplot(labels)
fig.savefig("results/pca_biplot.png", dpi=150, bbox_inches="tight")

fig = score_distributions(labels)
fig.savefig("results/score_distributions.png", dpi=150, bbox_inches="tight")
```

---

## Comparing with Phase 1 Benchmarks

Your results will differ from the paper's Phase 1 numbers. This is expected. Key comparisons:

| Metric | What to compare | Where to look |
|--------|----------------|---------------|
| Valid item rate | 70-85% typical | `cli.py clean` output |
| ICC range | >= 0.85 across axes | `labels_clean.parquet` columns |
| PCA structure | 2-3 PCs for > 80% variance | `dimensionality_report.md` |
| Top axis pair | Which 2 axes best reconstruct 8D | `dimensionality_report.md` |

See `docs/results_summary.md` for detailed metric interpretation.

---

## Troubleshooting

**Low valid item rate (<60%)**:
- Check temperature (should be 0.2, not 0.0)
- Check n_runs (need >= 10 for stable ICC)
- Check model compliance — some models produce malformed JSON

**High axis correlation**:
- Expected for economic axes (aequitas, market, inequality, risk)
- If all axes are highly correlated, the data may lack ideological diversity

**Triangulation gate fails**:
- See `docs/calibration-protocol.md` for diagnosis
- Try a different audit model or apply prompt engineering (see `docs/prompt-engineering.md`)
