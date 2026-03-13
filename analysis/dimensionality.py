"""
dimensionality.py — Axis information content and dimensionality analysis.

Key questions answered:
  1. How much variance do the 8 axes explain jointly and individually?
  2. Which axes carry the most information (PCA loadings)?
  3. Does Politikea's 2D representation (axes 1+2) capture more variance
     than any other 2-axis combination?
  4. How much predictive power for user agree/disagree is lost going
     from 8D → 2D → 1D?

All functions are pure; callers handle I/O.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

from .label_io import AXES
from .stats_utils import bootstrap_ci_2d


# ── PCA ───────────────────────────────────────────────────────────────────────

def pca_analysis(df: pd.DataFrame, axis_cols: Optional[list[str]] = None) -> dict:
    """
    Run PCA on the N×8 axis score matrix and return a rich result dict.

    Args:
        df:        DataFrame with one row per item (mean axis scores).
        axis_cols: Columns to use (defaults to AXES).

    Returns dict with keys:
        explained_variance_ratio  list[float]  per-PC fraction (8 values)
        cumulative_variance       list[float]  cumulative
        loadings                  dict[axis → list[float]]  per-axis PC loadings
        scree_df                  pd.DataFrame  for plotting
        pc_scores                 np.ndarray  shape (N, 8)
        n_items                   int
        n_pcs_80pct               int  number of PCs needed to explain ≥ 80%
    """
    axis_cols = axis_cols or AXES
    data = df[axis_cols].astype(float).dropna().to_numpy()

    pca = PCA(n_components=len(axis_cols), random_state=42)
    pc_scores = pca.fit_transform(data)

    evr = pca.explained_variance_ratio_.tolist()
    cum_var = np.cumsum(evr).tolist()
    n_80 = int(np.searchsorted(cum_var, 0.80) + 1)

    loadings = {
        axis: pca.components_[:, i].tolist()
        for i, axis in enumerate(axis_cols)
    }

    scree_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_var,
    })

    return {
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_var,
        "loadings": loadings,
        "scree_df": scree_df,
        "pc_scores": pc_scores,
        "n_items": len(data),
        "n_pcs_80pct": n_80,
        "axis_cols": axis_cols,
        "pca_model": pca,
    }


# ── Reconstruction R² ─────────────────────────────────────────────────────────

def _r2_for_subset(
    X_full: np.ndarray,
    subset_indices: list[int],
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Fit LinearRegression from a subset of axes → all 8 axes.
    Returns per-axis R² and overall NMSE.
    """
    X_sub = X_full[:, subset_indices]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, X_full, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)

    var = float(np.var(y_te))
    mse = float(mean_squared_error(y_te, y_hat))
    nmse = mse / var if var > 0 else float("nan")

    r2_per_axis = {}
    for i, axis in enumerate(AXES):
        ss_res = float(np.sum((y_te[:, i] - y_hat[:, i]) ** 2))
        ss_tot = float(np.sum((y_te[:, i] - np.mean(y_te[:, i])) ** 2))
        r2_per_axis[axis] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"nmse": nmse, "r2_per_axis": r2_per_axis, "n_test": len(y_te)}


def reconstruction_r2_all_subsets(
    df: pd.DataFrame,
    max_k: int = 4,
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute linear reconstruction R² for every k-axis subset (k = 1..max_k).

    For each subset of k axes, fits a linear model that predicts all 8
    axis scores from those k axes.  This measures how much information
    is retained when only using k axes as the representation.

    The *mean R²* across the 8 output axes is the headline number.
    A high mean R² means the chosen k axes encode most of the 8D structure.

    Args:
        df:        Clean items DataFrame (one row per item, mean axis scores).
        max_k:     Maximum subset size to evaluate.
        axis_cols: Columns to use (defaults to AXES).

    Returns:
        DataFrame with columns:
            subset_axes   tuple of axis names
            k             subset size
            mean_r2       mean R² across 8 axes
            nmse          normalised MSE
            <axis>_r2     per-axis R² for each of the 8 axes
    """
    axis_cols = axis_cols or AXES
    data = df[axis_cols].astype(float).dropna().to_numpy()
    n_axes = len(axis_cols)

    rows = []
    for k in range(1, min(max_k, n_axes) + 1):
        for subset in combinations(range(n_axes), k):
            res = _r2_for_subset(data, list(subset))
            row = {
                "subset_axes": tuple(axis_cols[i] for i in subset),
                "k": k,
                "mean_r2": float(np.nanmean(list(res["r2_per_axis"].values()))),
                "nmse": res["nmse"],
            }
            for axis, r2 in res["r2_per_axis"].items():
                row[f"{axis}_r2"] = r2
            rows.append(row)

    result = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
    return result.reset_index(drop=True)


def bootstrap_r2(
    df: pd.DataFrame,
    subset_axes: list[str],
    n_boot: int = 1000,
    axis_cols: Optional[list[str]] = None,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for mean R² of a given axis subset.

    Args:
        df:           Clean items DataFrame.
        subset_axes:  Which axes to use as the predictor subset.
        n_boot:       Number of bootstrap resamples.
        axis_cols:    Full axis column list (defaults to AXES).

    Returns:
        (lower, upper) 95% CI on mean R².
    """
    axis_cols = axis_cols or AXES
    data = df[axis_cols].astype(float).dropna().to_numpy()
    subset_indices = [axis_cols.index(a) for a in subset_axes]

    def _mean_r2(sample: np.ndarray) -> float:
        res = _r2_for_subset(sample, subset_indices)
        return float(np.nanmean(list(res["r2_per_axis"].values())))

    return bootstrap_ci_2d(_mean_r2, data, n_boot=n_boot)


# ── Predictive AUC (agree/disagree from item representation) ─────────────────

def predictive_auc(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
    action_col: str = "action",
    politics_id_col: str = "politics_id",
    item_id_col: str = "item_id",
    test_size: float = 0.25,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict:
    """
    Predict interaction action (AGREE vs DISAGREE) from item representation.

    Compares three representations:
        1D: first axis only (axis_aequitas_libertas)
        2D: first two axes (aequitas + imperium)
        8D: all 8 axes

    Returns AUC and accuracy for each representation.
    """
    axis_cols = axis_cols or AXES

    inter = interactions_df.copy()
    inter = inter[inter[action_col].isin(["AGREE", "DISAGREE"])].copy()
    inter["_y"] = (inter[action_col] == "AGREE").astype(int)

    merged = inter.merge(
        items_df[[item_id_col] + axis_cols],
        left_on=politics_id_col,
        right_on=item_id_col,
        how="inner",
    ).dropna(subset=axis_cols)

    if len(merged) < 50:
        return {"1d": {}, "2d": {}, "8d": {}, "n_interactions": len(merged)}

    y = merged["_y"].to_numpy()
    groups = merged[politics_id_col].astype(str).to_numpy()

    def _run(cols: list[str]) -> dict:
        X = merged[cols].astype(float).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) >= max(2, n_splits):
            cv = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
            fold_rows = []
            for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups), 1):
                clf = LogisticRegression(max_iter=2000, random_state=random_state)
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                try:
                    auc = float(roc_auc_score(y[te], p))
                except ValueError:
                    auc = float("nan")
                acc = float((clf.predict(X[te]) == y[te]).mean())
                fold_rows.append({"fold": fold_idx, "auc": auc, "accuracy": acc, "n_test": int(len(te))})
            fold_df = pd.DataFrame(fold_rows)
            return {
                "auc": float(fold_df["auc"].mean()),
                "accuracy": float(fold_df["accuracy"].mean()),
                "auc_std": float(fold_df["auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                "accuracy_std": float(fold_df["accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                "n_test": int(fold_df["n_test"].sum()),
                "n_folds": int(len(fold_df)),
                "fold_metrics": fold_rows,
            }

        # Fallback when groups are insufficient
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, p))
        except ValueError:
            auc = float("nan")
        acc = float((clf.predict(X_te) == y_te).mean())
        return {"auc": auc, "accuracy": acc, "auc_std": float("nan"), "accuracy_std": float("nan"), "n_test": len(y_te), "n_folds": 1, "fold_metrics": []}

    return {
        "1d": _run([axis_cols[0]]),
        "2d": _run(axis_cols[:2]),
        "8d": _run(axis_cols),
        "n_interactions": len(merged),
    }
