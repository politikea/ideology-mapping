#!/usr/bin/env bash
# run_pipeline.sh — convenience wrapper for the full ideology-mapping pipeline
#
# Usage:
#   ./scripts/run_pipeline.sh \
#       --input path/to/annotations.parquet \
#       --items path/to/proposals.parquet \
#       --output-dir results/
#
# Steps:
#   1. clean        — aggregate multi-run labels and flag valid items
#   2. validate     — text vs 8D Spearman correlation and joint-filter analysis
#   3. dimensionality — PCA, VIF, and 2-axis reconstruction R²
#
# Prerequisites:
#   pip install pandas pyarrow scikit-learn sentence-transformers scipy matplotlib seaborn
#
# Set --skip-predictive if you have no interaction/vote data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(dirname "$SCRIPT_DIR")"
CLI="$TOOLKIT_DIR/cli.py"

INPUT=""
ITEMS=""
OUTPUT_DIR="results"
SKIP_PREDICTIVE=""

usage() {
    echo "Usage: $0 --input <annotations.parquet> --items <proposals.parquet> [--output-dir <dir>] [--skip-predictive]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)          INPUT="$2";       shift 2 ;;
        --items)          ITEMS="$2";       shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";  shift 2 ;;
        --skip-predictive) SKIP_PREDICTIVE="--skip-predictive"; shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$INPUT" ]] && { echo "Error: --input is required"; usage; }
[[ -z "$ITEMS" ]] && { echo "Error: --items is required"; usage; }

LABELS_CLEAN="$OUTPUT_DIR/labels_clean.parquet"

echo "=== Step 1: clean ==="
python "$CLI" clean \
    --input "$INPUT" \
    --output "$LABELS_CLEAN"

echo ""
echo "=== Step 2: validate ==="
python "$CLI" validate \
    --labels "$LABELS_CLEAN" \
    --items "$ITEMS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 3: dimensionality ==="
python "$CLI" dimensionality \
    --labels "$LABELS_CLEAN" \
    --output-dir "$OUTPUT_DIR" \
    $SKIP_PREDICTIVE

echo ""
echo "=== Done. Results written to $OUTPUT_DIR/ ==="
