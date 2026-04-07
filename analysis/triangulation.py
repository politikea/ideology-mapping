"""
triangulation.py — Cross-model label triangulation (GPT vs local Ollama or Gemini).

This module is intentionally lightweight and reproducible:
- stratified subset sampling for budget-aware relabeling
- local Ollama relabeling (e.g. llama2:7b) or Gemini API relabeling
- directional + magnitude-tolerance agreement checks
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .category_mapping import map_category_id_to_name
from .label_io import AXES

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class TriangulationRule:
    min_directional_agreement: float = 0.85
    max_mean_abs_delta: float = 15.0
    max_axis_mean_abs_delta: float = 20.0


def stratified_subset(
    labels_df: pd.DataFrame,
    items_df: pd.DataFrame,
    sample_n: int = 400,
    category_col: str = "category",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows stratified by category and confidence quantiles."""
    df = labels_df.copy()
    if "valid" in df.columns:
        df = df[df["valid"]].copy()
    if "item_id" not in df.columns:
        raise ValueError("labels_df must include item_id")

    merge_cols = ["item_id"]
    has_category = category_col in items_df.columns
    has_category_id = "category_id" in items_df.columns
    if has_category:
        merge_cols.append(category_col)
    if has_category_id:
        merge_cols.append("category_id")
    work = df.merge(items_df[merge_cols], on="item_id", how="left")
    if category_col not in work.columns:
        work[category_col] = None
    if has_category_id:
        # Fill category labels from category_id mapping when category is missing.
        mapped = work["category_id"].map(map_category_id_to_name)
        work[category_col] = work[category_col].where(work[category_col].notna(), mapped)
    work[category_col] = work[category_col].fillna("uncategorized")
    work["_conf_bin"] = pd.qcut(
        work.get("mean_global_conf", pd.Series(np.ones(len(work)))),
        q=4,
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)

    strata = work.groupby([category_col, "_conf_bin"], dropna=False)
    rng = np.random.default_rng(random_state)
    pieces = []
    # Proportional allocation with floor=1
    for _, grp in strata:
        take = max(1, int(round(sample_n * (len(grp) / max(len(work), 1)))))
        take = min(take, len(grp))
        idx = rng.choice(grp.index.to_numpy(), size=take, replace=False)
        pieces.append(work.loc[idx])
    sampled = pd.concat(pieces, axis=0).drop_duplicates(subset=["item_id"])
    if len(sampled) > sample_n:
        sampled = sampled.sample(n=sample_n, random_state=random_state)
    return sampled.reset_index(drop=True)


def _extract_json_block(text: str) -> dict:
    t = text.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Ollama response.")
    return json.loads(t[start : end + 1])


def relabel_with_ollama(
    sampled_df: pd.DataFrame,
    items_df: pd.DataFrame,
    model: str = "llama2:7b",
    timeout_s: int = 90,
    text_col: str = "text_norm",
    cache_dir: "Path | None" = None,
    disable_cache: bool = False,
    repeats: int = 1,
    temperature: float = 0.0,
    seed: "int | None" = 42,
    return_stats: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """Relabel sampled items with local Ollama into 8 axes."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    text_lookup = items_df.set_index("item_id")[text_col].to_dict()
    prompt_version = "triangulate_8axis_v1"

    conn: "sqlite3.Connection | None" = None
    if cache_dir is not None and not disable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "triangulation_labels.sqlite3"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triangulation_cache (
                cache_key TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                record_json TEXT NOT NULL,
                source TEXT NOT NULL,
                generated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()

    n_total = 0
    n_cache_hit = 0
    n_generated = 0
    n_errors = 0
    rows = []
    work = sampled_df[["item_id"]].drop_duplicates().reset_index(drop=True)
    total_calls = len(work) * repeats
    progress_iter = range(total_calls)
    if tqdm is not None and total_calls >= 50:
        progress_iter = tqdm(progress_iter, total=total_calls, desc="triangulate_ollama", leave=False)
    for idx in progress_iter:
        row_idx = idx // repeats
        repeat_idx = idx % repeats
        n_total += 1
        item_id = work.iloc[row_idx]["item_id"]
        text = str(text_lookup.get(item_id, ""))[:4000]
        if not text:
            continue
        cache_key = hashlib.sha256(
            f"{prompt_version}::{model}::{temperature}::{seed}::{repeat_idx}::{item_id}::{text}".encode("utf-8")
        ).hexdigest()
        if conn is not None:
            cached = conn.execute(
                "SELECT record_json FROM triangulation_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if cached is not None:
                try:
                    rec = json.loads(str(cached[0]))
                    rec["_cache_source"] = "cache_hit"
                    rows.append(rec)
                    n_cache_hit += 1
                    continue
                except Exception:
                    pass
        prompt = (
            "Classify the political proposal on these 8 axes in range [-100, 100]:\n"
            "- axis_aequitas_libertas\n"
            "- axis_imperium_anarkhia\n"
            "- axis_universalism_particularism\n"
            "- axis_market_collective_allocation\n"
            "- axis_inequality_acceptance_correction\n"
            "- axis_individual_socialized_risk\n"
            "- axis_progressivism_preservation\n"
            "- axis_technocracy_populism\n\n"
            "Return strict JSON with exactly those keys and numeric values only.\n\n"
            f"Proposal:\n{text}"
        )
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": float(temperature)},
            }
            if seed is not None:
                payload["options"]["seed"] = int(seed) + int(repeat_idx)
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            raw = str(resp.json().get("response", ""))
            obj = _extract_json_block(raw)
            rec = {"item_id": item_id, "repeat_idx": int(repeat_idx)}
            for axis in AXES:
                rec[f"{axis}_ollama"] = float(obj.get(axis, np.nan))
            rec["_cache_source"] = "ollama"
            rows.append(rec)
            n_generated += 1
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO triangulation_cache
                    (cache_key, item_id, model, prompt_version, record_json, source, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        str(item_id),
                        model,
                        prompt_version,
                        json.dumps({k: rec[k] for k in rec if not k.startswith("_")}, ensure_ascii=False),
                        "ollama",
                        int(time.time()),
                    ),
                )
                conn.commit()
        except Exception as exc:
            n_errors += 1
            err_rec = {
                "item_id": item_id,
                "repeat_idx": int(repeat_idx),
                "error": f"{type(exc).__name__}: {exc}",
                "_cache_source": "error",
            }
            rows.append(err_rec)
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO triangulation_cache
                    (cache_key, item_id, model, prompt_version, record_json, source, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        str(item_id),
                        model,
                        prompt_version,
                        json.dumps({k: err_rec[k] for k in ("item_id", "repeat_idx", "error")}, ensure_ascii=False),
                        "error",
                        int(time.time()),
                    ),
                )
                conn.commit()
    if conn is not None:
        conn.close()
    df = pd.DataFrame(rows)
    stats = {
        "n_total": n_total,
        "n_cache_hit": n_cache_hit,
        "n_generated": n_generated,
        "n_errors": n_errors,
        "repeats": int(repeats),
        "temperature": float(temperature),
        "seed": seed,
        "model": model,
        "prompt_version": prompt_version,
        "cache_backend": "sqlite" if cache_dir is not None and not disable_cache else "none",
    }
    if return_stats:
        return df, stats
    return df


def relabel_with_gemini(
    sampled_df: pd.DataFrame,
    items_df: pd.DataFrame,
    api_key: str,
    model: str = "gemini-2.5-flash",
    timeout_s: int = 60,
    text_col: str = "text_norm",
    cache_dir: "Path | None" = None,
    disable_cache: bool = False,
    repeats: int = 1,
    temperature: float = 0.0,
    return_stats: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """Relabel sampled items with Google Gemini API into 8 axes.

    Uses the Gemini generateContent REST API directly (no extra SDK required).
    Axis scores are stored in columns named ``{axis}_ollama`` to stay compatible
    with the rest of the pipeline (aggregate_ollama_relabels, evaluate_agreement,
    write_report, etc.).
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    text_lookup = items_df.set_index("item_id")[text_col].to_dict()
    prompt_version = "triangulate_8axis_gemini_v1"

    conn: "sqlite3.Connection | None" = None
    if cache_dir is not None and not disable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "triangulation_labels.sqlite3"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triangulation_cache (
                cache_key TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                record_json TEXT NOT NULL,
                source TEXT NOT NULL,
                generated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()

    n_total = 0
    n_cache_hit = 0
    n_generated = 0
    n_errors = 0
    rows = []
    work = sampled_df[["item_id"]].drop_duplicates().reset_index(drop=True)
    total_calls = len(work) * repeats
    progress_iter = range(total_calls)
    if tqdm is not None and total_calls >= 10:
        progress_iter = tqdm(progress_iter, total=total_calls, desc="triangulate_gemini", leave=False)

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    for idx in progress_iter:
        row_idx = idx // repeats
        repeat_idx = idx % repeats
        n_total += 1
        item_id = work.iloc[row_idx]["item_id"]
        text = str(text_lookup.get(item_id, ""))[:4000]
        if not text:
            continue
        cache_key = hashlib.sha256(
            f"{prompt_version}::{model}::{temperature}::{repeat_idx}::{item_id}::{text}".encode("utf-8")
        ).hexdigest()
        if conn is not None:
            cached = conn.execute(
                "SELECT record_json FROM triangulation_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if cached is not None:
                try:
                    rec = json.loads(str(cached[0]))
                    rec["_cache_source"] = "cache_hit"
                    rows.append(rec)
                    n_cache_hit += 1
                    continue
                except Exception:
                    pass
        prompt = (
            "Classify the political proposal on these 8 axes in range [-100, 100].\n"
            "Use the full continuous range — do NOT round to multiples of 10.\n"
            "Return ONLY a strict JSON object with exactly these 8 keys and numeric float values. "
            "No explanation, no markdown, no code fences.\n\n"
            "Keys: axis_aequitas_libertas, axis_imperium_anarkhia, axis_universalism_particularism, "
            "axis_market_collective_allocation, axis_inequality_acceptance_correction, "
            "axis_individual_socialized_risk, axis_progressivism_preservation, axis_technocracy_populism\n\n"
            f"Proposal:\n{text}"
        )
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": 512,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            resp = requests.post(api_url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            raw_content = resp.json()
            raw_text = (
                raw_content.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            obj = _extract_json_block(raw_text)
            rec = {"item_id": item_id, "repeat_idx": int(repeat_idx)}
            for axis in AXES:
                rec[f"{axis}_ollama"] = float(obj.get(axis, np.nan))
            rec["_cache_source"] = "gemini"
            rows.append(rec)
            n_generated += 1
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO triangulation_cache
                    (cache_key, item_id, model, prompt_version, record_json, source, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        str(item_id),
                        model,
                        prompt_version,
                        json.dumps({k: rec[k] for k in rec if not k.startswith("_")}, ensure_ascii=False),
                        "gemini",
                        int(time.time()),
                    ),
                )
                conn.commit()
        except Exception as exc:
            n_errors += 1
            err_rec = {
                "item_id": item_id,
                "repeat_idx": int(repeat_idx),
                "error": f"{type(exc).__name__}: {exc}",
                "_cache_source": "error",
            }
            rows.append(err_rec)
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO triangulation_cache
                    (cache_key, item_id, model, prompt_version, record_json, source, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        str(item_id),
                        model,
                        prompt_version,
                        json.dumps({k: err_rec[k] for k in ("item_id", "repeat_idx", "error")}, ensure_ascii=False),
                        "error",
                        int(time.time()),
                    ),
                )
                conn.commit()

    if conn is not None:
        conn.close()
    df = pd.DataFrame(rows)
    stats = {
        "n_total": n_total,
        "n_cache_hit": n_cache_hit,
        "n_generated": n_generated,
        "n_errors": n_errors,
        "repeats": int(repeats),
        "temperature": float(temperature),
        "seed": None,
        "model": model,
        "prompt_version": prompt_version,
        "cache_backend": "sqlite" if cache_dir is not None and not disable_cache else "none",
    }
    if return_stats:
        return df, stats
    return df


def aggregate_ollama_relabels(
    relabel_df: pd.DataFrame,
    instability_threshold: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Aggregate repeated Ollama labels per item via median and report instability stats.

    Instability threshold applies to mean within-item axis SD.
    """
    if relabel_df.empty:
        return pd.DataFrame(columns=["item_id"] + [f"{a}_ollama" for a in AXES]), {
            "n_items": 0,
            "mean_repeats_per_item": 0.0,
            "mean_within_item_axis_sd": float("nan"),
            "instability_threshold": instability_threshold,
            "instability_rate": float("nan"),
        }

    valid = relabel_df.copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna()].copy()
    axis_cols = [f"{axis}_ollama" for axis in AXES if f"{axis}_ollama" in valid.columns]
    if not axis_cols:
        return pd.DataFrame(columns=["item_id"] + [f"{a}_ollama" for a in AXES]), {
            "n_items": 0,
            "mean_repeats_per_item": 0.0,
            "mean_within_item_axis_sd": float("nan"),
            "instability_threshold": instability_threshold,
            "instability_rate": float("nan"),
        }

    grouped = valid.groupby("item_id", dropna=False)
    agg = grouped[axis_cols].median().reset_index()

    repeat_counts = grouped.size().rename("repeat_count")
    per_axis_sd = grouped[axis_cols].std(ddof=0).fillna(0.0)
    item_sd = per_axis_sd.mean(axis=1) if len(per_axis_sd.columns) > 0 else pd.Series([], dtype=float)
    instability_rate = float(np.mean(item_sd > instability_threshold)) if len(item_sd) else float("nan")
    mean_within_item_axis_sd = float(np.mean(item_sd)) if len(item_sd) else float("nan")

    repeat_stats = {
        "n_items": int(agg["item_id"].nunique()),
        "mean_repeats_per_item": float(np.mean(repeat_counts.to_numpy())) if len(repeat_counts) else 0.0,
        "mean_within_item_axis_sd": mean_within_item_axis_sd,
        "instability_threshold": float(instability_threshold),
        "instability_rate": instability_rate,
    }
    return agg, repeat_stats


def evaluate_agreement(
    sampled_df: pd.DataFrame,
    ollama_df: pd.DataFrame,
    rule: Optional[TriangulationRule] = None,
) -> dict:
    """Directional + tolerance agreement metrics and gate decision."""
    rule = rule or TriangulationRule()
    merged = sampled_df.merge(ollama_df, on="item_id", how="inner")
    if merged.empty:
        return {"pass_gate": False, "reason": "No merged labels available.", "axis_metrics": []}

    axis_metrics = []
    dir_scores = []
    mae_scores = []
    for axis in AXES:
        g_col = f"{axis}_mean" if f"{axis}_mean" in merged.columns else axis
        o_col = f"{axis}_ollama"
        if g_col not in merged.columns or o_col not in merged.columns:
            continue
        g = merged[g_col].astype(float).to_numpy()
        o = merged[o_col].astype(float).to_numpy()
        valid = np.isfinite(g) & np.isfinite(o)
        if not np.any(valid):
            continue
        g = g[valid]
        o = o[valid]
        same_sign = np.mean(np.sign(g) == np.sign(o))
        mae = float(np.mean(np.abs(g - o)))
        axis_metrics.append(
            {"axis": axis, "directional_agreement": float(same_sign), "mae": mae, "n": int(len(g))}
        )
        dir_scores.append(float(same_sign))
        mae_scores.append(mae)

    mean_directional = float(np.mean(dir_scores)) if dir_scores else np.nan
    mean_mae = float(np.mean(mae_scores)) if mae_scores else np.nan
    max_axis_mae = float(np.max(mae_scores)) if mae_scores else np.nan

    pass_gate = (
        np.isfinite(mean_directional)
        and np.isfinite(mean_mae)
        and mean_directional >= rule.min_directional_agreement
        and mean_mae <= rule.max_mean_abs_delta
        and max_axis_mae <= rule.max_axis_mean_abs_delta
    )

    return {
        "pass_gate": bool(pass_gate),
        "summary": {
            "mean_directional_agreement": mean_directional,
            "mean_abs_delta": mean_mae,
            "max_axis_mean_abs_delta": max_axis_mae,
            "n_items_compared": int(len(merged)),
        },
        "rule": {
            "min_directional_agreement": rule.min_directional_agreement,
            "max_mean_abs_delta": rule.max_mean_abs_delta,
            "max_axis_mean_abs_delta": rule.max_axis_mean_abs_delta,
        },
        "axis_metrics": axis_metrics,
    }
