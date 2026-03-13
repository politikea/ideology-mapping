"""
cleaning.py — Data cleaning and inter-run reliability for multi-run labels.

Takes the raw consolidated parquet (one row per item×run) and produces:
  1. A per-item stability summary (mean, std, ICC, sign_agreement per axis).
  2. A `valid` flag marking items suitable for downstream analysis.

All functions are pure; callers handle I/O.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .label_io import AXES, CONF_AXES
from .stats_utils import benjamini_hochberg, binomial_sign_test, sign_agreement

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ── Confidence filtering ─────────────────────────────────────────────────────

# Patterns that indicate a non-political or junk item
_JUNK_PATTERNS = [
    r"^\s*[\d\W]+\s*$",                  # all digits/symbols (e.g. "67676767...")
    r"(?i)esto es una prueba",           # test items
    r"(?i)haced m[aá]s politikas",       # app feedback
    r"(?i)no puedo abrir",              # app bug reports
    r"(?i)^(si|no|sí)\s*[o\|/]\s*(no|sí|si)\s*[\?¿]?$",  # "si o no?"
]
import re as _re
_JUNK_RE = [_re.compile(p) for p in _JUNK_PATTERNS]


def flag_junk_items(
    items_df: pd.DataFrame,
    title_col: str = "title",
    desc_col: str = "description",
    min_words: int = 10,
) -> pd.DataFrame:
    """
    Mark items that are unlikely to be valid political proposals.

    Adds a boolean column `is_junk`. Callers decide whether to drop or flag.

    Criteria:
    - Total word count (title + description) below `min_words`
    - Title or description matches known noise patterns (test items, app bug
      reports, pure numeric/symbol strings)

    Returns a copy of `items_df` with an `is_junk` column added.
    """
    df = items_df.copy()
    title = df[title_col].fillna("").astype(str) if title_col in df.columns else pd.Series([""] * len(df))
    desc  = df[desc_col].fillna("").astype(str)  if desc_col  in df.columns else pd.Series([""] * len(df))
    combined = title + " " + desc

    word_count = combined.str.split().str.len().fillna(0)
    too_short  = word_count < min_words

    junk_pattern = combined.apply(
        lambda t: any(rx.search(t) for rx in _JUNK_RE)
    )

    df["is_junk"] = too_short | junk_pattern
    n_junk = df["is_junk"].sum()
    if n_junk:
        print(f"[cleaning] Flagged {n_junk} junk items "
              f"({n_junk / len(df):.1%} of {len(df):,}) — "
              f"short: {too_short.sum()}, pattern: {junk_pattern.sum()}")
    return df



def filter_by_confidence(
    df: pd.DataFrame,
    global_threshold: float = 0.5,
    axis_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Drop individual label rows that fall below confidence thresholds.

    - global_confidence < global_threshold → row dropped entirely.
    - For each axis, if conf_axis_X < axis_threshold, that axis score is
      set to NaN (the row is kept; only that axis value becomes unreliable).

    Args:
        df:               Raw multi-run labels DataFrame.
        global_threshold: Minimum global_confidence to keep a row.
        axis_threshold:   Per-axis confidence below which the score is nulled.

    Returns:
        Filtered DataFrame with low-confidence axis values set to NaN.
    """
    df = df.copy()

    # Drop rows with insufficient global confidence
    initial = len(df)
    df = df[df["global_confidence"].astype(float) >= global_threshold].copy()
    dropped = initial - len(df)
    if dropped:
        print(f"[cleaning] Dropped {dropped:,} rows with global_confidence < {global_threshold}")

    # Null out per-axis scores below axis_threshold
    for axis in AXES:
        conf_col = f"conf_{axis}"
        if conf_col in df.columns:
            mask = df[conf_col].astype(float) < axis_threshold
            if mask.any():
                df.loc[mask, axis] = np.nan

    return df


# ── Per-item cross-run stability ─────────────────────────────────────────────

def compute_item_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the multi-run label rows into one row per item_id.

    For each item × axis, computes across runs:
        - mean, std of axis scores
        - sign_agreement (fraction agreeing with majority sign)
        - sign_p_value (binomial p-value for directional labeling)
        - n_valid  (runs where that axis score is not NaN)
        - icc      (ICC(2,1) via pingouin if available; else NaN)

    Also computes:
        - n_runs           total label rows for this item
        - mean_global_conf mean global confidence across runs

    Returns one row per item_id with columns:
        item_id,
        <axis>_mean, <axis>_std, <axis>_sign_agreement,
        <axis>_sign_p, <axis>_n_valid, <axis>_icc,
        n_runs, mean_global_conf
    """
    rows = []
    grouped = df.groupby("item_id")

    # Try to import pingouin for ICC; fall back gracefully
    try:
        import pingouin as pg
        _has_pingouin = True
    except ImportError:
        _has_pingouin = False
        print("[cleaning] pingouin not installed — ICC values will be NaN. Install with: pip install pingouin")

    group_iter = grouped
    if tqdm is not None and grouped.ngroups >= 200:
        group_iter = tqdm(grouped, total=grouped.ngroups, desc="item_stability", leave=False)
    for item_id, group in group_iter:
        row: dict = {"item_id": item_id, "n_runs": len(group)}
        row["mean_global_conf"] = float(group["global_confidence"].astype(float).mean())

        for axis in AXES:
            vals = group[axis].astype(float).dropna().to_numpy()
            n_valid = len(vals)
            row[f"{axis}_n_valid"] = n_valid

            if n_valid == 0:
                row[f"{axis}_mean"] = np.nan
                row[f"{axis}_std"] = np.nan
                row[f"{axis}_sign_agreement"] = np.nan
                row[f"{axis}_sign_p"] = np.nan
                row[f"{axis}_icc"] = np.nan
                continue

            row[f"{axis}_mean"] = float(np.mean(vals))
            row[f"{axis}_std"] = float(np.std(vals))
            row[f"{axis}_sign_agreement"] = sign_agreement(vals)

            # Binomial sign test: k = number of runs agreeing with majority sign
            nonzero = vals[vals != 0.0]
            if len(nonzero) >= 2:
                majority = float(np.sign(np.median(nonzero)))
                k = int(np.sum(np.sign(nonzero) == majority))
                row[f"{axis}_sign_p"] = binomial_sign_test(k, len(nonzero))
            else:
                row[f"{axis}_sign_p"] = np.nan

            # ICC(2,1): requires ≥ 2 raters and ≥ 2 items per rater
            # We compute ICC across all items in the batch (done after loop)
            row[f"{axis}_icc"] = np.nan  # filled in by _compute_icc_batch below

        rows.append(row)

    stability = pd.DataFrame(rows)

    # Fill ICC values via batch computation (requires the full dataset)
    if _has_pingouin and len(df) > 0:
        stability = _compute_icc_batch(df, stability)

    # Per-item multiple-testing correction across 8 sign tests
    sign_cols = [f"{axis}_sign_p" for axis in AXES if f"{axis}_sign_p" in stability.columns]
    if sign_cols:
        q_arr = []
        rej_arr = []
        for _, row in stability[sign_cols].iterrows():
            bh = benjamini_hochberg(row.to_numpy(dtype=float), alpha=0.05)
            q_arr.append(bh["q_values"])
            rej_arr.append(bh["reject"])
        q_mat = np.vstack(q_arr) if q_arr else np.empty((0, len(sign_cols)))
        r_mat = np.vstack(rej_arr) if rej_arr else np.empty((0, len(sign_cols)), dtype=bool)
        for j, col in enumerate(sign_cols):
            stability[col.replace("_sign_p", "_sign_q")] = q_mat[:, j] if len(q_mat) else np.nan
            stability[col.replace("_sign_p", "_sign_reject_fdr")] = r_mat[:, j] if len(r_mat) else False

    return stability


def _compute_icc_batch(
    df: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute ICC(2,1) for each axis across all items using pingouin.

    pingouin.intraclass_corr expects long-format data with columns:
        targets (item_id), raters (run_id), ratings (axis value)

    We compute one global ICC per axis (not per-item), then broadcast
    that value back to stability to reflect overall axis reliability.
    The per-item ICC is not meaningfully estimable with only 13 raters
    unless we use a hierarchical model; we report the global value.

    To handle unbalanced data (different raters per item after confidence
    filtering) we restrict to items × rater pairs that appear in the
    most-covered balanced subset: items seen by the top-N raters.
    """
    import pingouin as pg

    stability = stability.copy()

    def _extract_icc_ci(icc_row: "pd.DataFrame") -> tuple[float, float]:
        """
        Pingouin changed CI column naming across versions.
        Accept known variants and fall back to NaN safely.
        """
        ci_candidates = ["CI95%", "CI95", "CI95%_lower", "CI95%_upper"]
        cols = set(icc_row.columns)

        # Variant A: tuple/list-like column
        for col in ("CI95%", "CI95"):
            if col in cols:
                raw = icc_row[col].iloc[0]
                if isinstance(raw, (list, tuple)) and len(raw) == 2:
                    return float(raw[0]), float(raw[1])
                if isinstance(raw, str):
                    # e.g. "[0.82, 0.90]"
                    cleaned = raw.strip().strip("[]")
                    parts = [p.strip() for p in cleaned.split(",")]
                    if len(parts) == 2:
                        try:
                            return float(parts[0]), float(parts[1])
                        except ValueError:
                            pass

        # Variant B: split columns
        if "CI95%_lower" in cols and "CI95%_upper" in cols:
            return float(icc_row["CI95%_lower"].iloc[0]), float(icc_row["CI95%_upper"].iloc[0])

        # Unknown variant
        return np.nan, np.nan

    for axis in AXES:
        subset = df[["item_id", "run_id", axis]].dropna(subset=[axis]).copy()
        subset = subset.rename(columns={axis: "ratings", "item_id": "targets", "run_id": "raters"})

        # Need at least 2 items and 2 raters
        if subset["targets"].nunique() < 2 or subset["raters"].nunique() < 2:
            continue

        # Build a balanced dataset: keep only (target, rater) pairs where
        # all selected raters have rated the same targets.
        # Strategy: find the set of raters that covers the most targets,
        # then keep only targets rated by ALL of those raters.
        rater_target_counts = subset.groupby("raters")["targets"].nunique()
        # Pick raters with the highest target coverage (top half by count)
        threshold = rater_target_counts.median()
        top_raters = rater_target_counts[rater_target_counts >= threshold].index
        sub = subset[subset["raters"].isin(top_raters)]
        # Keep only targets seen by ALL top raters
        target_rater_counts = sub.groupby("targets")["raters"].nunique()
        n_top = len(top_raters)
        balanced_targets = target_rater_counts[target_rater_counts == n_top].index
        sub = sub[sub["targets"].isin(balanced_targets)]

        if sub["targets"].nunique() < 2 or sub["raters"].nunique() < 2:
            continue

        try:
            icc_result = pg.intraclass_corr(
                data=sub,
                targets="targets",
                raters="raters",
                ratings="ratings",
            )
            # ICC(2,1) row
            icc_row = icc_result[icc_result["Type"] == "ICC2"]
            if len(icc_row) > 0:
                icc_val = float(icc_row["ICC"].iloc[0])
                icc_ci_low, icc_ci_high = _extract_icc_ci(icc_row)
                # Broadcast global ICC to all items for this axis
                stability[f"{axis}_icc"] = icc_val
                stability[f"{axis}_icc_ci_low"] = icc_ci_low
                stability[f"{axis}_icc_ci_high"] = icc_ci_high
        except Exception as e:
            print(f"[cleaning] ICC computation failed for {axis}: {e}")

    return stability


# ── Validity flagging ─────────────────────────────────────────────────────────

def flag_valid_items(
    stability: pd.DataFrame,
    min_stable_axes: int = 6,
    min_runs: int = 10,
    std_threshold: float = 30.0,
    sign_agreement_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Add a `valid` boolean column to the stability DataFrame.

    An item is marked valid if:
        1. n_runs >= min_runs  (enough runs to estimate variance)
        2. At least min_stable_axes of 8 axes are "stable"

    An axis is "stable" for an item if BOTH:
        - std < std_threshold  (scores don't swing wildly across runs)
        - sign_agreement >= sign_agreement_threshold
          (majority sign consistent enough for directional claims)

    Statistical anchor:
        With n=13 draws, sign_agreement >= 11/13 ≈ 0.846 → p < 0.02.
        The default 0.6 threshold is conservative (intentionally inclusive
        at the cleaning stage; stricter thresholds can be applied downstream).

    Returns the stability DataFrame with added columns:
        n_stable_axes, valid
    """
    stability = stability.copy()

    n_stable = np.zeros(len(stability), dtype=int)
    for axis in AXES:
        std_col = f"{axis}_std"
        sa_col = f"{axis}_sign_agreement"
        if std_col not in stability.columns or sa_col not in stability.columns:
            continue
        stable_mask = (
            stability[std_col].fillna(999).astype(float) < std_threshold
        ) & (
            stability[sa_col].fillna(0).astype(float) >= sign_agreement_threshold
        )
        n_stable += stable_mask.astype(int).to_numpy()

    stability["n_stable_axes"] = n_stable
    stability["valid"] = (
        (stability["n_runs"].astype(int) >= min_runs)
        & (stability["n_stable_axes"] >= min_stable_axes)
    )

    valid_count = stability["valid"].sum()
    total = len(stability)
    print(
        f"[cleaning] Valid items: {valid_count:,} / {total:,} "
        f"({100 * valid_count / total:.1f}%) "
        f"[min_runs={min_runs}, min_stable_axes={min_stable_axes}]"
    )
    return stability


# ── Summary helpers ───────────────────────────────────────────────────────────

def cleaning_summary(stability: pd.DataFrame) -> dict:
    """
    Return a dict of summary statistics for the cleaning report.
    """
    valid = stability[stability["valid"]]
    summary: dict = {
        "total_items": int(len(stability)),
        "valid_items": int(stability["valid"].sum()),
        "invalid_items": int((~stability["valid"]).sum()),
        "mean_n_runs": float(stability["n_runs"].mean()),
        "per_axis": {},
    }
    for axis in AXES:
        icc_col = f"{axis}_icc"
        std_col = f"{axis}_std"
        sa_col = f"{axis}_sign_agreement"
        summary["per_axis"][axis] = {
            "mean_std": float(stability[std_col].mean()) if std_col in stability.columns else None,
            "mean_sign_agreement": float(stability[sa_col].mean()) if sa_col in stability.columns else None,
            "icc": float(stability[icc_col].iloc[0]) if icc_col in stability.columns and stability[icc_col].notna().any() else None,
            "icc_ci_low": float(stability[f"{axis}_icc_ci_low"].iloc[0]) if f"{axis}_icc_ci_low" in stability.columns and stability[f"{axis}_icc_ci_low"].notna().any() else None,
            "icc_ci_high": float(stability[f"{axis}_icc_ci_high"].iloc[0]) if f"{axis}_icc_ci_high" in stability.columns and stability[f"{axis}_icc_ci_high"].notna().any() else None,
        }
    return summary


def threshold_sensitivity_grid(
    df_raw: pd.DataFrame,
    global_thresholds: list[float],
    axis_thresholds: list[float],
    std_thresholds: list[float],
    sign_agreement_thresholds: list[float],
    min_stable_axes: int = 6,
    min_runs: int = 10,
) -> pd.DataFrame:
    """
    Evaluate validity-retention tradeoffs across threshold combinations.

    Returns a grid with retained rows/items and valid-item rates. This is intended
    for methodological transparency, not for automatic hyper-optimization.
    """
    rows: list[dict] = []
    combinations = [
        (g, a, s, sa)
        for g in global_thresholds
        for a in axis_thresholds
        for s in std_thresholds
        for sa in sign_agreement_thresholds
    ]
    combo_iter = combinations
    if tqdm is not None and len(combinations) >= 20:
        combo_iter = tqdm(combinations, total=len(combinations), desc="threshold_grid", leave=False)

    # Cache per confidence-filter pair to avoid redundant repeated work
    conf_cache: dict[tuple[float, float], tuple[pd.DataFrame, int]] = {}
    for g, a, s, sa in combo_iter:
        conf_key = (float(g), float(a))
        if conf_key not in conf_cache:
            filtered = filter_by_confidence(df_raw, global_threshold=g, axis_threshold=a)
            stability = compute_item_stability(filtered)
            conf_cache[conf_key] = (stability, int(len(filtered)))
            n_rows_after_conf = int(len(filtered))
        else:
            stability, n_rows_after_conf = conf_cache[conf_key]

        flagged = flag_valid_items(
            stability,
            min_stable_axes=min_stable_axes,
            min_runs=min_runs,
            std_threshold=s,
            sign_agreement_threshold=sa,
        )
        n_total = int(len(flagged))
        n_valid = int(flagged["valid"].sum()) if n_total else 0
        rows.append(
            {
                "global_threshold": float(g),
                "axis_threshold": float(a),
                "std_threshold": float(s),
                "sign_agreement_threshold": float(sa),
                "n_rows_after_conf_filter": n_rows_after_conf,
                "n_items_total": n_total,
                "n_items_valid": n_valid,
                "valid_rate": float(n_valid / n_total) if n_total else np.nan,
            }
        )
    return pd.DataFrame(rows)
