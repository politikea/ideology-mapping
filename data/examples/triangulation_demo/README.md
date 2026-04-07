# Triangulation Demo — Synthetic Mock Data

This directory contains **purely synthetic** datasets for demonstrating cross-model triangulation in the ideology-mapping toolkit. No real user data is included.

## Files

| File | Description |
|------|-------------|
| `reference_labels.parquet` | Synthetic reference model labels (50 items, 8 axes). Columns: `item_id`, `{axis}_mean`, `valid`. |
| `audit_gemini_mock.parquet` | Simulated Gemini audit labels exhibiting **acquiescence bias** (all scores shifted positive, no negatives). |
| `audit_llama_mock.parquet` | Simulated Llama audit labels exhibiting **quantization and task overload** (coarse rounding, high noise, extreme values). |

## Bias Patterns

### Gemini mock — acquiescence bias
All axis scores are shifted positive by +25 to +45 points relative to the reference, then clamped so no score falls below 0. This produces a systematic positivity bias where the audit model "agrees" with everything, yielding inflated scores across the board.

### Llama mock — quantization and task overload
Scores are rounded to coarse bins (-100, -50, -20, -10, 0, 10, 20, 50, 100) and perturbed with large random noise (std ~30). Approximately 10% of scores are slammed to extreme values. This simulates a smaller model struggling with the task — producing noisy, discretized outputs.

## Running Triangulation

From the `ideology-mapping/` root:

```bash
# Evaluate Gemini audit against reference
python cli.py triangulate \
    --reference data/examples/triangulation_demo/reference_labels.parquet \
    --audit data/examples/triangulation_demo/audit_gemini_mock.parquet \
    --output-dir results/

# Evaluate Llama audit against reference
python cli.py triangulate \
    --reference data/examples/triangulation_demo/reference_labels.parquet \
    --audit data/examples/triangulation_demo/audit_llama_mock.parquet \
    --output-dir results/
```

## Regenerating

To regenerate these files (deterministic, seed = 2024):

```bash
python scripts/generate_triangulation_mocks.py
```
