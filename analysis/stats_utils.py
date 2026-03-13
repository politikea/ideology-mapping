"""
stats_utils.py — Small, standalone statistical primitives.

All functions are pure (no I/O, no side effects).  They are imported by
every other analysis module that needs significance tests or uncertainty bounds.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as scipy_stats


# ── Sign agreement ───────────────────────────────────────────────────────────

def sign_agreement(values: np.ndarray) -> float:
    """
    Fraction of draws that share the majority sign.

    For a vector of 13 continuous draws, this is the fraction that agree
    with sign(median(values)).  Zero values count as neither positive nor
    negative and reduce the denominator.

    Returns NaN if all values are zero.
    """
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return float("nan")
    majority = float(np.sign(np.median(nonzero)))
    return float(np.mean(np.sign(nonzero) == majority))


# ── Binomial sign test ───────────────────────────────────────────────────────

def binomial_sign_test(k: int, n: int) -> float:
    """
    Two-sided binomial test: p-value for observing >= k/n same-sign draws
    under the null hypothesis that the true probability is 0.5.

    Interpretation:
        k=11, n=13 → p ≈ 0.0225   (significant at alpha=0.05)
        k=10, n=13 → p ≈ 0.0923   (marginal)
        k=9,  n=13 → p ≈ 0.2668   (not significant)

    Uses scipy.stats.binomtest for exact p-value.
    """
    if n == 0:
        return float("nan")
    result = scipy_stats.binomtest(k, n, p=0.5, alternative="two-sided")
    return float(result.pvalue)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """
    Benjamini-Hochberg FDR control.

    Args:
        p_values: 1-D array of p-values (NaNs allowed).
        alpha: Target false discovery rate.

    Returns:
        Dict with:
          - q_values: adjusted p-values (NaN where input NaN)
          - reject: boolean rejection mask
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    reject = np.zeros(len(p), dtype=bool)

    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return {"q_values": q, "reject": reject}

    pv = p[valid_idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))

    # Monotone adjusted p-values from right to left
    adj = np.empty_like(ranked)
    running = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        rank = float(i + 1)
        value = (m / rank) * ranked[i]
        running = min(running, value)
        adj[i] = min(1.0, running)

    # Rejection rule
    thresholds = (np.arange(1, len(ranked) + 1, dtype=float) / m) * float(alpha)
    rej_ranked = ranked <= thresholds
    if np.any(rej_ranked):
        k_max = int(np.where(rej_ranked)[0].max())
        rej_ranked = np.arange(len(ranked)) <= k_max
    else:
        rej_ranked = np.zeros(len(ranked), dtype=bool)

    back_order = np.empty_like(order)
    back_order[order] = np.arange(len(order))
    q_valid = adj[back_order]
    reject_valid = rej_ranked[back_order]

    q[valid_idx] = q_valid
    reject[valid_idx] = reject_valid
    return {"q_values": q, "reject": reject}


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(
    fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for a scalar statistic.

    Args:
        fn:           Function that takes a 1-D array and returns a float.
        data:         1-D array of observations.
        n_boot:       Number of bootstrap resamples.
        ci:           Coverage (default 0.95 → 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data, dtype=float)
    stats = np.array([
        fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lower = float(np.nanpercentile(stats, 100 * alpha / 2))
    upper = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_ci_2d(
    fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI where data is a 2-D array (N × features).
    Resamples rows with replacement.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    boot_stats = np.array([
        fn(data[rng.integers(0, n, size=n)])
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


# ── Cohen's kappa for sign agreement (binary) ────────────────────────────────

def fleiss_kappa_sign(votes: np.ndarray) -> float:
    """
    Approximate Fleiss' kappa for binary sign agreement across N raters.

    `votes` is shape (n_items, n_raters) with values in {-1, 0, +1}.
    Zero votes are excluded from the agreement calculation per item.

    Returns kappa in [-1, 1]; values > 0.6 indicate substantial agreement.
    """
    votes = np.asarray(votes, dtype=float)
    n_items, n_raters = votes.shape

    # Per-item: fraction voting positive (ignoring zeros)
    pos_fracs = []
    ns = []
    for row in votes:
        nonzero = row[row != 0]
        if len(nonzero) < 2:
            continue
        pos_fracs.append(float(np.mean(nonzero > 0)))
        ns.append(len(nonzero))

    if not pos_fracs:
        return float("nan")

    pos_fracs = np.array(pos_fracs)
    p_bar = float(np.mean(pos_fracs))
    p_e = p_bar**2 + (1 - p_bar)**2  # expected agreement under independence

    # Per-item observed agreement
    p_o_items = pos_fracs**2 + (1 - pos_fracs)**2
    p_o = float(np.mean(p_o_items))

    if abs(1 - p_e) < 1e-9:
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))
