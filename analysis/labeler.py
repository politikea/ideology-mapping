"""
labeler.py — Score political proposals via any OpenAI-compatible API.

This module provides the "step 1" of the pipeline: calling an LLM to score
proposals across the 8 ideological axes. It supports any OpenAI-compatible
endpoint (OpenAI, Azure OpenAI, local vLLM/Ollama with OpenAI-compat mode).

Usage:
    from analysis.labeler import score_proposal, score_proposals

    client = OpenAI(api_key="...")
    result = score_proposal("Increase minimum wage to $25/hour", client=client)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .label_io import AXES, CONF_AXES

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROMPT_VERSION = "label_8axis_v1"
_PROMPT_TEMPLATE: str | None = None


def _load_prompt_template() -> str:
    """Load the prompt template from the prompts/ directory."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "label_8axis_v1.txt"
        _PROMPT_TEMPLATE = prompt_path.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def parse_response(raw_json: dict | str) -> dict:
    """Parse and validate a single LLM response into a flat dict with axis scores and confidence.

    Args:
        raw_json: Either a parsed dict or a JSON string from the LLM.

    Returns:
        Flat dict with axis scores, confidence scores, global_confidence, flags,
        and rationale_spans.

    Raises:
        ValueError: If the response is missing required axis fields.
    """
    if isinstance(raw_json, str):
        raw_json = json.loads(raw_json)

    result = {}

    # Extract axis scores
    missing_axes = []
    for axis in AXES:
        if axis in raw_json:
            score = float(raw_json[axis])
            result[axis] = max(-100.0, min(100.0, score))
        else:
            missing_axes.append(axis)

    if missing_axes:
        raise ValueError(f"Response missing axis fields: {missing_axes}")

    # Extract confidence scores (optional — default to 0.5 if missing)
    for conf_axis in CONF_AXES:
        if conf_axis in raw_json:
            result[conf_axis] = max(0.0, min(1.0, float(raw_json[conf_axis])))
        else:
            result[conf_axis] = 0.5

    result["global_confidence"] = max(0.0, min(1.0, float(raw_json.get("global_confidence", 0.5))))
    result["flags"] = raw_json.get("flags", [])
    result["rationale_spans"] = raw_json.get("rationale_spans", [])

    return result


def _cache_key(item_id: str, text: str, prompt_version: str, model: str, run_idx: int) -> str:
    """Deterministic cache key for a single scoring run."""
    payload = f"{prompt_version}::{model}::{item_id}::{run_idx}::{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_from_cache(cache_dir: Path, key: str) -> dict | None:
    """Load a cached response if it exists."""
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_to_cache(cache_dir: Path, key: str, data: dict) -> None:
    """Save a response to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def score_proposal(
    text: str,
    client: Any,
    model: str = "claude-sonnet-4-20250514",
    n_runs: int = 13,
    temperature: float = 0.2,
    prompt_template: str | None = None,
    item_id: str | None = None,
    cache_dir: Path | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Score a single proposal N times and return a list of parsed responses.

    Args:
        text: The political proposal text.
        client: An OpenAI-compatible client (must have client.chat.completions.create).
        model: Model name to use.
        n_runs: Number of repeated scoring runs.
        temperature: Sampling temperature.
        prompt_template: Custom prompt template (uses default if None).
        item_id: Optional identifier for caching.
        cache_dir: Directory for caching responses. None disables caching.
        max_retries: Max retries per API call on failure.
        retry_delay: Base delay between retries (exponential backoff).

    Returns:
        List of parsed response dicts, one per successful run.
    """
    template = prompt_template or _load_prompt_template()
    prompt = template.replace("{{TEXT}}", text)
    _item_id = item_id or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    results = []
    for run_idx in range(n_runs):
        # Check cache
        if cache_dir is not None:
            key = _cache_key(_item_id, text, PROMPT_VERSION, model, run_idx)
            cached = _load_from_cache(cache_dir, key)
            if cached is not None:
                results.append(cached)
                continue

        # Call the API with retries (supports both Anthropic and OpenAI clients)
        parsed = None
        for attempt in range(max_retries):
            try:
                if hasattr(client, "messages"):
                    # Anthropic SDK
                    resp = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = resp.content[0].text
                else:
                    # OpenAI SDK (or compatible)
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    raw = resp.choices[0].message.content
                parsed = parse_response(raw)
                parsed["item_id"] = _item_id
                parsed["run_idx"] = run_idx
                parsed["model"] = model
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"[labeler] Failed after {max_retries} attempts for item {_item_id} "
                          f"run {run_idx}: {exc}")

        if parsed is not None:
            results.append(parsed)
            if cache_dir is not None:
                _save_to_cache(cache_dir, key, parsed)

    return results


def score_proposals(
    df: pd.DataFrame,
    client: Any,
    text_col: str = "text",
    item_id_col: str = "item_id",
    model: str = "claude-sonnet-4-20250514",
    n_runs: int = 13,
    temperature: float = 0.2,
    prompt_template: str | None = None,
    cache_dir: Path | None = None,
    rate_limit_delay: float = 0.1,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Score a DataFrame of proposals and return consolidated multi-run annotations.

    Args:
        df: DataFrame with at least a text column and an item_id column.
        client: An OpenAI-compatible client.
        text_col: Column name containing proposal text.
        item_id_col: Column name for proposal IDs.
        model: Model name to use.
        n_runs: Number of repeated scoring runs per proposal.
        temperature: Sampling temperature.
        prompt_template: Custom prompt template (uses default if None).
        cache_dir: Directory for caching responses. None disables caching.
        rate_limit_delay: Delay in seconds between proposals.
        max_retries: Max retries per API call on failure.

    Returns:
        DataFrame with one row per (item_id, run_idx), containing all axis scores,
        confidence values, and metadata. Ready for `cli.py clean`.
    """
    all_rows = []
    items = df[[item_id_col, text_col]].drop_duplicates(subset=[item_id_col])

    iterator = items.iterrows()
    if tqdm is not None:
        iterator = tqdm(items.iterrows(), total=len(items), desc="scoring proposals")

    for _, row in iterator:
        item_id = str(row[item_id_col])
        text = str(row[text_col])

        runs = score_proposal(
            text=text,
            client=client,
            model=model,
            n_runs=n_runs,
            temperature=temperature,
            prompt_template=prompt_template,
            item_id=item_id,
            cache_dir=cache_dir,
            max_retries=max_retries,
        )

        for run in runs:
            run_row = {
                "item_id": item_id,
                "run_id": f"{item_id}_run{run.get('run_idx', 0):03d}",
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "global_confidence": run.get("global_confidence", 0.5),
            }
            for axis in AXES:
                run_row[axis] = run.get(axis, np.nan)
            for conf in CONF_AXES:
                run_row[conf] = run.get(conf, 0.5)
            all_rows.append(run_row)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return pd.DataFrame(all_rows)
