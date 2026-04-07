#!/usr/bin/env python3
"""
score_proposals.py — Batch-score political proposals via any OpenAI-compatible API.

This is the "step 0" before running the analysis pipeline. It takes a CSV/parquet
of proposals, scores each one N times with the 8-axis prompt, and outputs a
consolidated multi-run annotations parquet ready for `cli.py clean`.

Usage:
    python scripts/score_proposals.py \
        --input data/examples/politikas_100.csv \
        --output annotations.parquet \
        --model claude-sonnet-4-20250514 \
        --n-runs 13 \
        --api-key $OPENAI_API_KEY

    # Or with a custom base URL (Azure, local vLLM, etc.)
    python scripts/score_proposals.py \
        --input proposals.csv \
        --output annotations.parquet \
        --model claude-sonnet-4-20250514 \
        --base-url http://localhost:8000/v1 \
        --api-key $API_KEY
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score proposals with the 8-axis labeling prompt via OpenAI-compatible API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to proposals CSV or parquet.")
    parser.add_argument("--output", required=True, help="Path to write consolidated annotations parquet.")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name (default: claude-sonnet-4-20250514).")
    parser.add_argument("--n-runs", type=int, default=13, help="Scoring runs per proposal (default: 13).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    parser.add_argument("--text-col", default="text", help="Column containing proposal text (default: text).")
    parser.add_argument("--item-id-col", default="item_id", help="Column for proposal IDs (default: item_id).")
    parser.add_argument("--api-key", default=None, help="API key (default: $OPENAI_API_KEY).")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API.")
    parser.add_argument("--cache-dir", default=None, help="Directory for response caching (default: no cache).")
    parser.add_argument("--rate-limit-delay", type=float, default=0.1, help="Delay between proposals in seconds.")
    args = parser.parse_args()

    # Lazy imports so --help is fast
    _has_anthropic = False
    _has_openai = False
    try:
        import anthropic
        _has_anthropic = True
    except ImportError:
        pass
    try:
        from openai import OpenAI
        _has_openai = True
    except ImportError:
        pass
    if not _has_anthropic and not _has_openai:
        print("Error: install anthropic or openai package. pip install -e '.[scoring]'", file=sys.stderr)
        return 1

    import pandas as pd

    # Add toolkit root to path
    toolkit_dir = Path(__file__).resolve().parent.parent
    if str(toolkit_dir) not in sys.path:
        sys.path.insert(0, str(toolkit_dir))

    from analysis.labeler import score_proposals

    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"[score] Loaded {len(df)} proposals from {input_path}")

    if args.text_col not in df.columns:
        print(f"Error: column '{args.text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        return 1

    # Create item_id if missing
    if args.item_id_col not in df.columns:
        df[args.item_id_col] = [f"item_{i:05d}" for i in range(len(df))]
        print(f"[score] Generated item_id column ({args.item_id_col})")

    # Build client — prefer Anthropic, fall back to OpenAI
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY / OPENAI_API_KEY", file=sys.stderr)
        return 1

    if _has_anthropic and not args.base_url and "claude" in args.model:
        client = anthropic.Anthropic(api_key=api_key)
    elif _has_openai:
        client_kwargs = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)
    else:
        client = anthropic.Anthropic(api_key=api_key)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Score
    print(f"[score] Model: {args.model}, runs/proposal: {args.n_runs}, temperature: {args.temperature}")
    result = score_proposals(
        df=df,
        client=client,
        text_col=args.text_col,
        item_id_col=args.item_id_col,
        model=args.model,
        n_runs=args.n_runs,
        temperature=args.temperature,
        cache_dir=cache_dir,
        rate_limit_delay=args.rate_limit_delay,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    n_items = result["item_id"].nunique()
    print(f"[score] Wrote {len(result)} rows ({n_items} items x ~{args.n_runs} runs) -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
