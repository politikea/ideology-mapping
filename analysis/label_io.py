"""
label_io.py — I/O helpers and shared constants for the analysis layer.

All paths are passed explicitly as arguments so scripts can be invoked
from any working directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# ── Canonical axis names ────────────────────────────────────────────────────

AXES: list[str] = [
    "axis_aequitas_libertas",
    "axis_imperium_anarkhia",
    "axis_universalism_particularism",
    "axis_market_collective_allocation",
    "axis_inequality_acceptance_correction",
    "axis_individual_socialized_risk",
    "axis_progressivism_preservation",
    "axis_technocracy_populism",
]

CONF_AXES: list[str] = [f"conf_{a}" for a in AXES]

# Expected columns in the consolidated multi-run parquet
_REQUIRED_COLS: set[str] = {
    "item_id",
    "text_hash",
    "run_id",
    "global_confidence",
    *AXES,
    *CONF_AXES,
}


# ── Config loader ────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    """Load config.yaml and return the raw dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Parquet loaders ──────────────────────────────────────────────────────────

def load_all_runs(path: str | Path) -> pd.DataFrame:
    """
    Load the consolidated multi-run annotations parquet.

    Validates that the expected schema columns are present and that
    `run_id` exists (i.e. pull_aml_labels.py has already been run).

    Returns a DataFrame with one row per (item_id, run_id, rank).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated labels parquet not found: {path}\n"
            "Ensure you have scored your proposals and saved the multi-run annotations parquet."
        )
    df = pd.read_parquet(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Consolidated parquet is missing expected columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns)}"
        )
    return df


def load_clean_items(path: str | Path) -> pd.DataFrame:
    """
    Load the filtered clean items parquet produced by the pipeline's filter step.

    Expected columns: item_id, text_norm, text_hash, plus any category/metadata
    columns present in the original CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Clean items parquet not found: {path}\n"
            "Run `python scripts/run_pipeline.py filter` first."
        )
    df = pd.read_parquet(path)
    if "item_id" not in df.columns:
        raise ValueError(f"Clean items parquet missing `item_id` column: {path}")
    return df


def load_labels_clean(path: str | Path) -> pd.DataFrame:
    """
    Load the cleaned/aggregated labels produced by 01_clean.py.

    One row per item_id; columns include axis mean/std/ICC per axis,
    `valid` boolean, `n_runs` count.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Clean labels parquet not found: {path}\n"
            "Run `python cli.py clean --input your_annotations.parquet --output labels_clean.parquet` first."
        )
    return pd.read_parquet(path)


def load_interactions(path: str | Path) -> pd.DataFrame:
    """
    Load interactions CSV using the project's standard CSV format
    (semicolon-separated, decimal comma, standard NA values).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Interactions CSV not found: {path}")
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        na_values=["Null", "NULL", ""],
        keep_default_na=False,
        dtype=str,
    )


# ── Output path helpers ──────────────────────────────────────────────────────

def default_output_root(base_dir: str | Path = ".") -> Path:
    """
    Return the default analysis output root relative to a given base directory.
    Defaults to `./outputs/analysis/` in the current working directory.
    """
    return Path(base_dir).resolve() / "outputs" / "analysis"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
