"""
generate_example_labels.py

Generates realistic mock annotations and clean labels for the 100 synthetic
example proposals in data/examples/politikas_100.csv.

Scores are hand-derived from the semantic content of each proposal using the
8-axis framework. Multi-run noise is added to simulate N_RUNS=13 repeated
API calls at temperature 0.2, matching the Phase 1 protocol.

Axis polarities (negative = left pole, positive = right pole):
  aequitas_libertas:              -100 redistribution  /  +100 individual freedom
  imperium_anarkhia:              -100 state authority  /  +100 decentralisation
  universalism_particularism:     -100 universal rules  /  +100 group-specific
  market_collective_allocation:   -100 market          /  +100 collective
  inequality_acceptance_correction: -100 inequality ok  /  +100 must correct
  individual_socialized_risk:     -100 individual risk  /  +100 socialised risk
  progressivism_preservation:     -100 reform/change   /  +100 tradition
  technocracy_populism:           -100 expert-led      /  +100 popular will

Run from the ideology-mapping/ root:
    python scripts/generate_example_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.cleaning import compute_item_stability, filter_by_confidence, flag_valid_items

AXES = [
    "aequitas_libertas",
    "imperium_anarkhia",
    "universalism_particularism",
    "market_collective_allocation",
    "inequality_acceptance_correction",
    "individual_socialized_risk",
    "progressivism_preservation",
    "technocracy_populism",
]

# fmt: off
# Hand-scored base values for all 100 proposals.
# Columns: [aeq, imp, uni, mkt, ineq, risk, prog, tech]
# Each score is the "true" ideological position; noise is added per run.
BASE_SCORES: dict[str, list[int]] = {
    # ── Economy & Taxation ────────────────────────────────────────────────────
    "ex_001": [-75, -10, -15,  65,  72,  70, -50,   5],  # Universal basic income
    "ex_002": [-60, -25, -10,  55,  60,  45, -40,   0],  # Housing expropriation
    "ex_004": [ 45,  20,   0, -45, -35, -40,  20,  10],  # Spending cuts 30%
    "ex_011": [ 55,  10,   5, -40, -45, -35,  25,  -5],  # Abolish inheritance tax
    "ex_012": [-50,  -5,  -5,  50,  55,  45, -35,  -5],  # Windfall profits tax
    "ex_047": [-70, -35, -10,  70,  65,  60, -45, -10],  # Nationalise electricity grid
    "ex_048": [ 60,  35,   0, -65, -50, -55,  15,  10],  # Remove energy price controls
    "ex_049": [-45,  -5,  -5,  45,  55,  35, -30,   0],  # Progressive corporate tax 25%
    "ex_050": [ 55,  15,   5, -50, -45, -40,  15,   5],  # Flat corporate tax 15%
    "ex_081": [-60,  -5, -10,  60,  65,  55, -35,   0],  # National wealth tax 1%
    "ex_082": [ 30,   5,   5, -25, -20, -15,  10,   0],  # Income tax cuts lower brackets

    # ── Infrastructure & Housing ──────────────────────────────────────────────
    "ex_017": [-50, -20,  -5,  45,  50,  40, -35,  -5],  # 20% social housing mandate
    "ex_018": [ 45,  25,   5, -40, -35, -35,  15,   5],  # Deregulate urban density
    "ex_056": [-65, -30, -10,  60,  60,  55, -40, -10],  # National social housing authority
    "ex_057": [-40, -15,  -5,  40,  45,  35, -25,   0],  # Vacancy tax
    "ex_058": [ 50,  15,   5, -45, -40, -40,  20,   5],  # Abolish rent control
    "ex_059": [-30, -15,  -5,  30,  30,  25, -35,  -5],  # High-speed rail extension
    "ex_060": [ 40,  25,   5, -40, -30, -30,  15,   5],  # Rail privatisation concessions
    "ex_083": [-45, -20,  -5,  45,  45,  40, -30,  -5],  # Free public transport
    "ex_084": [ 50,  30,   5, -50, -35, -40,  20,   5],  # Privatise national rail operator

    # ── Institutions & Government ─────────────────────────────────────────────
    "ex_003": [ 10,  35,   0, -10,  -5, -10,   5,  30],  # Eliminate party public funding
    "ex_019": [-15, -20, -10,  15,  20,  10, -20, -15],  # Beneficial ownership registry
    "ex_020": [ 20,  15,   5, -15, -10, -10,  15,  20],  # Reduce parliament size 40%
    "ex_021": [-20,  50, -15,  15,  10,  10, -25,  45],  # Citizens assembly sortition
    "ex_022": [-10,  10,  -5,   5,   5,   5,  -5,  15],  # Electoral reform mixed system
    "ex_065": [-20, -15, -10,  15,  20,  10, -20, -20],  # Anti-corruption authority
    "ex_066": [-10, -10,  -5,  10,   5,   5,   0,  25],  # Parliamentary judicial hearings
    "ex_067": [-15,  10, -10,  10,  10,   5, -20,  25],  # Lower voting age to 16
    "ex_068": [-10, -15, -10,   5,   5,   5,  10, -15],  # Compulsory voting
    "ex_091": [-25, -35, -10,  20,  15,  10, -15, -20],  # State digital identity platform
    "ex_092": [-10,  40, -10,   5,   5,   5, -10,  20],  # Ban centralised biometric data

    # ── Labour & Job Market ───────────────────────────────────────────────────
    "ex_005": [-45, -10,  -5,  45,  45,  50, -30,  10],  # 32-hour week
    "ex_015": [-40, -10,  -5,  40,  40,  45, -25,   0],  # Permanent public contracts
    "ex_016": [ 40,  20,   5, -35, -30, -35,  15,   5],  # Direct hour agreements small firms
    "ex_051": [-25,  -5, -20,  20,  30,  15, -30,  -5],  # Gender quota 40% boards
    "ex_052": [ 25,   5,  20, -20, -25, -10,  20,   5],  # Abolish gender quotas
    "ex_053": [-35,  -5,  -5,  35,  40,  45, -25,   0],  # Parental leave 24 weeks
    "ex_054": [ 40,  20,  10, -35, -30, -40,  15,   5],  # Flexible gig contracting
    "ex_055": [-55, -15,  -5,  55,  55,  65, -35,   5],  # Gig workers = employees
    "ex_087": [-45, -10,  -5,  45,  45,  55, -25,   5],  # Suspend delivery apps storms
    "ex_088": [-35,  -5,  -5,  35,  35,  45, -20,   0],  # Reduce transport compensation period
    "ex_089": [-55,  -5,  -5,  55,  60,  60, -35,   5],  # Mandatory profit sharing
    "ex_090": [ 45,  20,   5, -40, -35, -45,  20,   5],  # Allow redundancy without auth

    # ── Public Services ───────────────────────────────────────────────────────
    "ex_006": [ 35,  15,   5, -40, -25, -30,  15,   0],  # Hospital privatisation
    "ex_013": [-60, -25,  -5,  60,  55,  55, -35, -10],  # Public pharma company
    "ex_014": [-15, -20, -10,  10,  15,   5, -20, -20],  # Hospital waiting time transparency
    "ex_061": [-30, -15,  -5,  30,  30,  30, -20, -10],  # GP 4-year continuity contract
    "ex_062": [ 30,  10,   5, -35, -20, -25,  10,   5],  # Vouchers private clinics
    "ex_063": [-50, -15, -10,  45,  50,  45, -30,  -5],  # Free school meals universal
    "ex_064": [ 35,  10,   5, -30, -35, -30,  15,   5],  # Means testing social benefits
    "ex_093": [-55, -20,  -5,  50,  55,  55, -30, -10],  # Universal long-term care
    "ex_094": [-20,  25,   0,  20,  20,  15, -10,  -5],  # Elderly care to local authorities

    # ── Security & Defence ────────────────────────────────────────────────────
    "ex_007": [-10, -50,  15,  10,  -5,  10,  30,  -5],  # More police 20%
    "ex_023": [-15,  35, -15,   5,  15,   5, -25,  15],  # Ban facial recognition police
    "ex_024": [-20, -30, -10,  10,  15,  10, -15, -10],  # Body cameras all police
    "ex_025": [-10, -45,  10,  10, -10,  10,  20, -15],  # Military spending 2% GDP
    "ex_026": [-30,  20,  -5,  20,  20,  15, -30,  10],  # Defence budget to diplomacy
    "ex_071": [-15, -30,  -5,   5,  10,   5, -10,  10],  # Ban assault-style firearms
    "ex_072": [ 25,  30,   5, -15, -10,  -5,  10,  10],  # Self-defence firearm licence

    # ── Rights & Freedoms ─────────────────────────────────────────────────────
    "ex_008": [-10,  45, -25,  -5,  20,  10, -50,  25],  # Legalise cannabis
    "ex_027": [-15,  15, -35,   0,  25,  10, -45,  15],  # Legalise same-sex marriage
    "ex_028": [  5,  -5,  25,   0, -15,  -5,  40, -10],  # Restrict gender-affirming minors
    "ex_029": [-20,  30, -20,   5,  20,  10, -30,  20],  # Digital bill of rights
    "ex_030": [-15,  40, -20,   5,  20,  10, -50,  25],  # Decriminalise all drug use
    "ex_069": [-30, -10, -15,  25,  30,  20, -20,  -5],  # Free legal representation
    "ex_070": [-15, -15, -10,   5,  10,   5,  -5,  -5],  # Abolish statute limitations minors

    # ── Science, Energy & Environment ─────────────────────────────────────────
    "ex_009": [-20, -10, -10,  20,  25,  15, -40,  -5],  # Ban single-use plastics
    "ex_031": [-40, -15, -10,  35,  40,  35, -50, -10],  # 80% renewable by 2030
    "ex_032": [  5,  10,  10, -10,  -5,  -5,  30,   5],  # Halt wind farm approvals
    "ex_033": [-30, -10,  -5,  25,  30,  25, -35, -20],  # Public science funding 1.5%
    "ex_034": [-25, -15,  -5,  20,  25,  20, -35, -15],  # Net-zero government buildings
    "ex_073": [-45, -10,  -5,  40,  45,  40, -40,  -5],  # Home retrofit programme
    "ex_074": [-10, -15,   0,  10,  10,  10,  10, -25],  # Small modular nuclear reactors
    "ex_095": [-35, -10,  -5,  30,  35,  25, -45,  -5],  # Replace short-haul flights rail
    "ex_096": [ 20,  10,   5, -20, -15, -15,  25,   5],  # Oppose flight restrictions

    # ── Culture & Civic Education ─────────────────────────────────────────────
    "ex_010": [  5,  -5,  -5,   0,   5,   0,  -5,  -5],  # Personal finance in schools
    "ex_035": [-10, -10, -10,   5,  10,   5, -10, -10],  # Civic education standalone subject
    "ex_036": [  5,  -5,  20,  -5,  -5,  -5,  40,  -5],  # Mandatory religious education
    "ex_037": [-25, -10, -10,  20,  20,  15, -20,  -5],  # Libraries as digital hubs
    "ex_038": [-20,  -5, -10,  15,  15,  10, -25,  10],  # Civic storytelling schools
    "ex_075": [-30, -10, -10,  25,  25,  20, -25, -10],  # Digital literacy adults
    "ex_076": [-20,  -5, -15,  15,  15,  10, -20,  10],  # 30% budget young artists
    "ex_085": [-65, -20, -10,  60,  55,  50, -40,  -5],  # Prohibit private school fees
    "ex_086": [ 60,  40,   5, -55, -50, -50,  15,  10],  # Deregulate education system

    # ── Identity & Social Cohesion ────────────────────────────────────────────
    "ex_039": [ 20,  -5,  25, -15,  -5,  -5,  30,   5],  # Points-based immigration
    "ex_040": [-35,  10, -20,  25,  30,  20, -30,  10],  # Permanent residency undocumented
    "ex_041": [-25,  -5, -15,  20,  20,  15, -20,  -5],  # National integration programme
    "ex_042": [  5, -20,  35, -10,  -5,  -5,  40,   0],  # Official language only
    "ex_077": [-20,  20, -35,  10,  15,  10, -25,  10],  # Expand co-official languages
    "ex_078": [ 10, -25,  40, -10,  -5,  -5,  45,   0],  # Constitutional language prohibition
    "ex_097": [-40,   5, -30,  35,  50,  30, -35,  10],  # Reparations marginalised communities
    "ex_098": [ 30,   5,  30, -25, -35, -15,  25,   5],  # Oppose group-based preferences

    # ── International Relations ───────────────────────────────────────────────
    "ex_043": [-25,  10, -20,  20,  20,  15, -25,   5],  # Mediterranean diplomatic forum
    "ex_044": [-20,  30,  -5,  10,  10,  10, -10,  20],  # Withdraw from NATO command
    "ex_045": [-30,  15, -20,  20,  25,  15, -30,  10],  # Ratify nuclear weapons ban
    "ex_046": [ 20,  30,  10, -20,  -5,  -5,  15,   5],  # Bilateral free trade agreements
    "ex_079": [-15,   5, -20,  10,  10,   5, -20,   5],  # Youth exchange programme
    "ex_080": [  0, -10,  15, -10,   5,   5,  15,  -5],  # 25% agricultural tariff
    "ex_099": [-20, -30, -20,  15,  15,  10, -20, -20],  # EU banking union accession
    "ex_100": [  5,  25,  10,  -5,  -5,  -5,  10,  25],  # Referendum EU single market
}
# fmt: on

N_RUNS = 13
NOISE_STD = 7.0  # Simulate temperature=0.2 call variation


def generate_annotations(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for item_id, base in BASE_SCORES.items():
        for run_i in range(N_RUNS):
            noise = rng.normal(0, NOISE_STD, size=8)
            scores = np.clip(np.array(base, dtype=float) + noise, -100, 100)
            # Confidence correlates weakly with score extremity (more extreme = more confident)
            mean_extremity = np.mean(np.abs(base)) / 100
            base_conf = 0.65 + 0.20 * mean_extremity
            row: dict = {
                "item_id": item_id,
                "run_id": run_i,
                "global_confidence": float(np.clip(base_conf + rng.normal(0, 0.05), 0.3, 1.0)),
            }
            for ax, sc, bc in zip(AXES, scores, base):
                axis_conf = 0.60 + 0.25 * (abs(bc) / 100) + rng.normal(0, 0.05)
                row[f"axis_{ax}"] = float(sc)
                row[f"conf_axis_{ax}"] = float(np.clip(axis_conf, 0.3, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(42)

    print("Generating mock annotations for 100 proposals …")
    annotations = generate_annotations(rng)

    # Add text_hash (required by cleaning pipeline)
    annotations["text_hash"] = annotations["item_id"].apply(lambda x: x.encode().hex())

    ann_path = ROOT / "data" / "examples" / "annotations_mock.parquet"
    annotations.to_parquet(ann_path, index=False)
    print(f"  Saved {len(annotations):,} rows → {ann_path.relative_to(ROOT)}")

    print("Cleaning and computing stability …")
    filtered = filter_by_confidence(annotations, global_threshold=0.5, axis_threshold=0.4)
    stability = compute_item_stability(filtered)
    stability = flag_valid_items(
        stability,
        min_stable_axes=6,
        min_runs=10,
        std_threshold=30.0,
        sign_agreement_threshold=0.6,
    )

    lbl_path = ROOT / "data" / "examples" / "labels_clean.parquet"
    stability.to_parquet(lbl_path, index=False)

    n_valid = stability["valid"].sum()
    print(f"  Valid: {n_valid} / {len(stability)} ({100 * n_valid / len(stability):.0f}%)")
    print(f"  Saved → {lbl_path.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
