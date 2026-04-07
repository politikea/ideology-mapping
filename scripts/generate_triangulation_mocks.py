"""
generate_triangulation_mocks.py

Generates synthetic mock datasets for demonstrating cross-model triangulation.
All data is purely synthetic — no real user data or internal numbers are used.

Two audit-model mock patterns are produced:
  - Gemini acquiescence bias: scores shifted positive, no negatives.
  - Llama quantization/task overload: coarse rounding and high noise.

Run from the ideology-mapping/ root:
    python scripts/generate_triangulation_mocks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AXES = [
    "axis_aequitas_libertas",
    "axis_imperium_anarkhia",
    "axis_universalism_particularism",
    "axis_market_collective_allocation",
    "axis_inequality_acceptance_correction",
    "axis_individual_socialized_risk",
    "axis_progressivism_preservation",
    "axis_technocracy_populism",
]

N_ITEMS = 50
SEED = 2024
OUTPUT_DIR = ROOT / "data" / "examples" / "triangulation_demo"


def load_item_ids() -> list[str]:
    """Read item_ids from the existing synthetic proposals CSV."""
    csv_path = ROOT / "data" / "examples" / "politikas_100.csv"
    df = pd.read_csv(csv_path)
    return df["item_id"].tolist()[:N_ITEMS]


def generate_reference_scores(
    item_ids: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Generate synthetic reference labels centered around 0 with std ~40."""
    rows = []
    for item_id in item_ids:
        row: dict = {"item_id": item_id}
        scores = rng.normal(0, 40, size=len(AXES))
        scores = np.clip(scores, -100, 100)
        for ax, sc in zip(AXES, scores):
            row[f"{ax}_mean"] = float(np.round(sc, 1))
        row["valid"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def generate_gemini_audit(
    reference_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Simulate Gemini acquiescence bias: positive shift, no negative scores."""
    rows = []
    for _, ref_row in reference_df.iterrows():
        row: dict = {"item_id": ref_row["item_id"]}
        for ax in AXES:
            ref_score = ref_row[f"{ax}_mean"]
            # Add positive offset between +25 and +45
            offset = rng.uniform(25, 45)
            # Small additional noise
            noise = rng.normal(0, 5)
            score = ref_score + offset + noise
            # Clamp: no negative scores (acquiescence bias), cap at 100
            score = np.clip(score, 0, 100)
            row[f"{ax}_ollama"] = float(np.round(score, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def generate_llama_audit(
    reference_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Simulate llama quantization and task overload patterns."""
    coarse_bins = np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100])
    rows = []
    for _, ref_row in reference_df.iterrows():
        row: dict = {"item_id": ref_row["item_id"]}
        for ax in AXES:
            ref_score = ref_row[f"{ax}_mean"]
            # Add large noise (task overload / instability)
            noise = rng.normal(0, 30)
            score = ref_score + noise
            # Quantize to nearest coarse bin (simulates rounding behavior)
            idx = np.argmin(np.abs(coarse_bins - score))
            score = float(coarse_bins[idx])
            # Occasionally slam to extreme values
            if rng.random() < 0.10:
                score = float(rng.choice([-100, -50, 0, 50, 100]))
            row[f"{ax}_ollama"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading item IDs from politikas_100.csv (first {N_ITEMS}) ...")
    item_ids = load_item_ids()
    print(f"  Found {len(item_ids)} item IDs.")

    print("Generating synthetic reference labels ...")
    reference_df = generate_reference_scores(item_ids, rng)
    ref_path = OUTPUT_DIR / "reference_labels.parquet"
    reference_df.to_parquet(ref_path, index=False)
    print(f"  Saved {len(reference_df)} rows -> {ref_path.relative_to(ROOT)}")

    print("Generating Gemini acquiescence-bias audit mock ...")
    gemini_df = generate_gemini_audit(reference_df, rng)
    gemini_path = OUTPUT_DIR / "audit_gemini_mock.parquet"
    gemini_df.to_parquet(gemini_path, index=False)
    neg_count = sum((gemini_df[[f"{ax}_ollama" for ax in AXES]] < 0).any(axis=1))
    print(f"  Saved {len(gemini_df)} rows -> {gemini_path.relative_to(ROOT)}")
    print(f"  Negative-score rows: {neg_count} (should be 0)")

    print("Generating Llama quantization/overload audit mock ...")
    llama_df = generate_llama_audit(reference_df, rng)
    llama_path = OUTPUT_DIR / "audit_llama_mock.parquet"
    llama_df.to_parquet(llama_path, index=False)
    unique_vals = set()
    for ax in AXES:
        unique_vals.update(llama_df[f"{ax}_ollama"].unique())
    print(f"  Saved {len(llama_df)} rows -> {llama_path.relative_to(ROOT)}")
    print(f"  Unique score values: {sorted(unique_vals)}")

    print("\nAll mock files written to:")
    print(f"  {OUTPUT_DIR.relative_to(ROOT)}/")
    for f in sorted(OUTPUT_DIR.glob("*.parquet")):
        print(f"    {f.name}")
    print("Done.")


if __name__ == "__main__":
    main()
