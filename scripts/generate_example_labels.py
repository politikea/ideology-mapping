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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.cleaning import compute_item_stability, filter_by_confidence, flag_valid_items  # noqa: E402

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
    "ex_001": [-70, -24, -12, 62, 76, 68, -66, 7],  # Universal basic income
    "ex_002": [-60, -25, -10, 55, 60, 45, -40, 0],  # Housing expropriation
    "ex_004": [50, 6, 3, -48, -31, -42, 4, 12],  # Spending cuts 30%
    "ex_011": [60, -4, 8, -43, -41, -37, 9, -3],  # Abolish inheritance tax
    "ex_012": [-45, -19, -2, 47, 59, 43, -51, -3],  # Windfall profits tax
    "ex_047": [-65, -49, -7, 67, 69, 58, -61, -8],  # Nationalise electricity grid
    "ex_048": [65, 21, 3, -68, -46, -57, -1, 12],  # Remove energy price controls
    "ex_049": [-40, -19, -2, 42, 59, 33, -46, 2],  # Progressive corporate tax 25%
    "ex_050": [60, 1, 8, -53, -41, -42, -1, 7],  # Flat corporate tax 15%
    "ex_081": [-55, -19, -7, 57, 69, 53, -51, 2],  # National wealth tax 1%
    "ex_082": [35, -9, 8, -28, -16, -17, -6, 2],  # Income tax cuts lower brackets

    # ── Infrastructure & Housing ──────────────────────────────────────────────
    "ex_017": [-50, -20, -5, 45, 50, 40, -35, -5],  # 20% social housing mandate
    "ex_018": [45, 25, 5, -40, -35, -35, 15, 5],  # Deregulate urban density
    "ex_056": [-65, -30, -10, 60, 60, 55, -40, -10],  # National social housing authority
    "ex_057": [-40, -15, -5, 40, 45, 35, -25, 0],  # Vacancy tax
    "ex_058": [50, 15, 5, -45, -40, -40, 20, 5],  # Abolish rent control
    "ex_059": [-30, -15, -5, 30, 30, 25, -35, -5],  # High-speed rail extension
    "ex_060": [40, 25, 5, -40, -30, -30, 15, 5],  # Rail privatisation concessions
    "ex_083": [-45, -20, -5, 45, 45, 40, -30, -5],  # Free public transport
    "ex_084": [50, 30, 5, -50, -35, -40, 20, 5],  # Privatise national rail operator

    # ── Institutions & Government ─────────────────────────────────────────────
    "ex_003": [14, 22, 2, 3, 5, -2, -15, 35],  # Eliminate party public funding
    "ex_019": [-11, -33, -8, 28, 30, 18, -40, -10],  # Beneficial ownership registry
    "ex_020": [24, 2, 7, -2, 0, -2, -5, 25],  # Reduce parliament size 40%
    "ex_021": [-16, 37, -13, 28, 20, 18, -45, 50],  # Citizens assembly sortition
    "ex_022": [-6, -3, -3, 18, 15, 13, -25, 20],  # Electoral reform mixed system
    "ex_065": [-16, -28, -8, 28, 30, 18, -40, -15],  # Anti-corruption authority
    "ex_066": [-6, -23, -3, 23, 15, 13, -20, 30],  # Parliamentary judicial hearings
    "ex_067": [-11, -3, -8, 23, 20, 13, -40, 30],  # Lower voting age to 16
    "ex_068": [-6, -28, -8, 18, 15, 13, -10, -10],  # Compulsory voting
    "ex_091": [-21, -48, -8, 33, 25, 18, -35, -15],  # State digital identity platform
    "ex_092": [-6, 27, -8, 18, 15, 13, -30, 25],  # Ban centralised biometric data

    # ── Labour & Job Market ───────────────────────────────────────────────────
    "ex_005": [-53, -25, -3, 57, 64, 56, -55, 14],  # 32-hour week
    "ex_015": [-48, -25, -3, 52, 59, 51, -50, 4],  # Permanent public contracts
    "ex_016": [32, 5, 7, -23, -11, -29, -10, 9],  # Direct hour agreements small firms
    "ex_051": [-33, -20, -18, 32, 49, 21, -55, -1],  # Gender quota 40% boards
    "ex_052": [17, -10, 22, -8, -6, -4, -5, 9],  # Abolish gender quotas
    "ex_053": [-43, -20, -3, 47, 59, 51, -50, 4],  # Parental leave 24 weeks
    "ex_054": [32, 5, 12, -23, -11, -34, -10, 9],  # Flexible gig contracting
    "ex_055": [-63, -30, -3, 67, 74, 71, -60, 9],  # Gig workers = employees
    "ex_087": [-53, -25, -3, 57, 64, 61, -50, 9],  # Suspend delivery apps storms
    "ex_088": [-43, -20, -3, 47, 54, 51, -45, 4],  # Reduce transport compensation period
    "ex_089": [-63, -20, -3, 67, 79, 66, -60, 9],  # Mandatory profit sharing
    "ex_090": [37, 5, 7, -28, -16, -39, -5, 9],  # Allow redundancy without auth

    # ── Public Services ───────────────────────────────────────────────────────
    "ex_006": [25, -6, -7, -18, -5, -9, -8, 0],  # Hospital privatisation
    "ex_013": [-70, -46, -17, 82, 75, 76, -58, -10],  # Public pharma company
    "ex_014": [-25, -41, -22, 32, 35, 26, -43, -20],  # Hospital waiting time transparency
    "ex_061": [-40, -36, -17, 52, 50, 51, -43, -10],  # GP 4-year continuity contract
    "ex_062": [20, -11, -7, -13, 0, -4, -13, 5],  # Vouchers private clinics
    "ex_063": [-60, -36, -22, 67, 70, 66, -53, -5],  # Free school meals universal
    "ex_064": [25, -11, -7, -8, -15, -9, -8, 5],  # Means testing social benefits
    "ex_093": [-65, -41, -17, 72, 75, 76, -53, -10],  # Universal long-term care
    "ex_094": [-30, 4, -12, 42, 40, 36, -33, -5],  # Elderly care to local authorities

    # ── Security & Defence ────────────────────────────────────────────────────
    "ex_007": [-4, -85, 26, 26, -16, 15, 47, 4],  # More police 20%
    "ex_023": [-9, 0, -4, 21, 4, 10, -8, 24],  # Ban facial recognition police
    "ex_024": [-14, -65, 1, 26, 4, 15, 2, -1],  # Body cameras all police
    "ex_025": [-4, -80, 21, 26, -20, 15, 37, -6],  # Military spending 2% GDP
    "ex_026": [-24, -15, 6, 36, 10, 20, -13, 19],  # Defence budget to diplomacy
    "ex_071": [-9, -65, 6, 21, 0, 10, 7, 19],  # Ban assault-style firearms
    "ex_072": [31, -5, 16, 1, -20, 0, 27, 19],  # Self-defence firearm licence

    # ── Rights & Freedoms ─────────────────────────────────────────────────────
    "ex_008": [-1, 13, -27, 9, 25, 15, -50, 23],  # Legalise cannabis
    "ex_027": [-6, -17, -37, 14, 30, 15, -45, 13],  # Legalise same-sex marriage
    "ex_028": [14, -37, 23, 14, -10, 0, 40, -12],  # Restrict gender-affirming minors
    "ex_029": [-11, -2, -22, 19, 25, 15, -30, 18],  # Digital bill of rights
    "ex_030": [-6, 8, -22, 19, 25, 15, -50, 23],  # Decriminalise all drug use
    "ex_069": [-21, -42, -17, 39, 35, 25, -20, -7],  # Free legal representation
    "ex_070": [-6, -47, -12, 19, 15, 10, -5, -7],  # Abolish statute limitations minors

    # ── Science, Energy & Environment ─────────────────────────────────────────
    "ex_009": [-20, -10, -10, 20, 25, 15, -40, -5],  # Ban single-use plastics
    "ex_031": [-40, -15, -10, 35, 40, 35, -50, -10],  # 80% renewable by 2030
    "ex_032": [5, 10, 10, -10, -5, -5, 30, 5],  # Halt wind farm approvals
    "ex_033": [-30, -10, -5, 25, 30, 25, -35, -20],  # Public science funding 1.5%
    "ex_034": [-25, -15, -5, 20, 25, 20, -35, -15],  # Net-zero government buildings
    "ex_073": [-45, -10, -5, 40, 45, 40, -40, -5],  # Home retrofit programme
    "ex_074": [-10, -15, 0, 10, 10, 10, 10, -25],  # Small modular nuclear reactors
    "ex_095": [-35, -10, -5, 30, 35, 25, -45, -5],  # Replace short-haul flights rail
    "ex_096": [20, 10, 5, -20, -15, -15, 25, 5],  # Oppose flight restrictions

    # ── Culture & Civic Education ─────────────────────────────────────────────
    "ex_010": [5, -5, -5, 0, 5, 0, -5, -5],  # Personal finance in schools
    "ex_035": [-10, -10, -10, 5, 10, 5, -10, -10],  # Civic education standalone subject
    "ex_036": [5, -5, 20, -5, -5, -5, 40, -5],  # Mandatory religious education
    "ex_037": [-25, -10, -10, 20, 20, 15, -20, -5],  # Libraries as digital hubs
    "ex_038": [-20, -5, -10, 15, 15, 10, -25, 10],  # Civic storytelling schools
    "ex_075": [-30, -10, -10, 25, 25, 20, -25, -10],  # Digital literacy adults
    "ex_076": [-20, -5, -15, 15, 15, 10, -20, 10],  # 30% budget young artists
    "ex_085": [-65, -20, -10, 60, 55, 50, -40, -5],  # Prohibit private school fees
    "ex_086": [60, 40, 5, -55, -50, -50, 15, 10],  # Deregulate education system

    # ── Identity & Social Cohesion ────────────────────────────────────────────
    "ex_039": [11, -29, 57, 13, -4, 5, 43, 29],  # Points-based immigration
    "ex_040": [-44, -14, 12, 53, 31, 30, -17, 34],  # Permanent residency undocumented
    "ex_041": [-34, -29, 17, 48, 21, 25, -7, 19],  # National integration programme
    "ex_042": [-4, -44, 67, 18, -4, 5, 53, 24],  # Official language only
    "ex_077": [-29, -4, -3, 38, 16, 20, -12, 34],  # Expand co-official languages
    "ex_078": [1, -49, 72, 18, -4, 5, 58, 24],  # Constitutional language prohibition
    "ex_097": [-49, -19, 2, 63, 51, 40, -22, 34],  # Reparations marginalised communities
    "ex_098": [21, -19, 62, 3, -34, -5, 38, 29],  # Oppose group-based preferences

    # ── International Relations ───────────────────────────────────────────────
    "ex_043": [-25, 10, -20, 20, 20, 15, -25, 5],  # Mediterranean diplomatic forum
    "ex_044": [-20, 30, -5, 10, 10, 10, -10, 20],  # Withdraw from NATO command
    "ex_045": [-30, 15, -20, 20, 25, 15, -30, 10],  # Ratify nuclear weapons ban
    "ex_046": [20, 30, 10, -20, -5, -5, 15, 5],  # Bilateral free trade agreements
    "ex_079": [-15, 5, -20, 10, 10, 5, -20, 5],  # Youth exchange programme
    "ex_080": [0, -10, 15, -10, 5, 5, 15, -5],  # 25% agricultural tariff
    "ex_099": [-20, -30, -20, 15, 15, 10, -20, -20],  # EU banking union accession
    "ex_100": [5, 25, 10, -5, -5, -5, 10, 25],  # Referendum EU single market
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

    ann_path = ROOT / "data" / "examples" / "annotations_example.parquet"
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
