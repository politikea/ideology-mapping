"""
visualize.py — Publication-quality charts for the ideology-mapping toolkit.

All functions return a :class:`matplotlib.figure.Figure` and accept an optional
*ax* parameter.  When *ax* is ``None`` a new figure is created automatically.

Designed for the 8-axis scoring system defined in :mod:`.label_io`.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .label_io import AXES

# ── Module-level style ──────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.9)

MODULE_PALETTE: list[str] = sns.color_palette(
    "Set3", n_colors=20
).as_hex()
"""Consistent academic palette used across all charts."""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _short_name(axis: str) -> str:
    """Strip the ``axis_`` prefix and replace underscores with spaces."""
    name = axis.removeprefix("axis_")
    return name.replace("_", " ")


def _mean_cols() -> list[str]:
    """Return the ``_mean`` column names for the 8 axes."""
    return [f"{a}_mean" for a in AXES]


def _resolve_axis_cols(df: pd.DataFrame) -> list[str]:
    """Pick *_mean* columns if present, otherwise fall back to raw AXES."""
    mean_cols = _mean_cols()
    if all(c in df.columns for c in mean_cols):
        return mean_cols
    return [a for a in AXES if a in df.columns]


def _ensure_ax(ax: Optional[plt.Axes]) -> tuple[Figure, plt.Axes]:
    """Return (fig, ax), creating a new figure when *ax* is ``None``."""
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()
    return fig, ax


# ── 1. Correlation heatmap ──────────────────────────────────────────────────

def correlation_heatmap(
    labels_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """8x8 Pearson correlation heatmap of axis scores.

    Parameters
    ----------
    labels_df : DataFrame
        One row per item with ``*_mean`` (or raw axis) columns.
    ax : matplotlib Axes, optional
        Target axes; a new figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = _resolve_axis_cols(labels_df)
    corr = labels_df[cols].corr(method="pearson")

    short_labels = [_short_name(c.removesuffix("_mean")) for c in cols]
    corr.index = short_labels
    corr.columns = short_labels

    fig, ax = _ensure_ax(ax)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Axis Correlation Matrix")
    return fig


# ── 2. PCA biplot ───────────────────────────────────────────────────────────

def pca_biplot(
    labels_df: pd.DataFrame,
    items_df: Optional[pd.DataFrame] = None,
    category_col: str = "category",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """PCA scatter on PC1/PC2 with axis-loading arrows.

    Parameters
    ----------
    labels_df : DataFrame
        One row per item with axis score columns.
    items_df : DataFrame, optional
        If provided and contains *category_col*, points are colored by
        category.
    category_col : str
        Column in *items_df* to use for coloring (default ``"category"``).
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = _resolve_axis_cols(labels_df)
    data = labels_df[cols].dropna().astype(float)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(data.values)

    fig, ax = _ensure_ax(ax)

    # Determine categories for coloring
    categories = None
    if items_df is not None and category_col in items_df.columns:
        merged = labels_df.loc[data.index].copy()
        if "item_id" in merged.columns and "item_id" in items_df.columns:
            cat_map = items_df.set_index("item_id")[category_col]
            categories = merged["item_id"].map(cat_map)
        elif category_col in merged.columns:
            categories = merged[category_col]

    if categories is not None:
        categories = categories.loc[data.index]
        unique_cats = categories.dropna().unique()
        palette = dict(zip(unique_cats, MODULE_PALETTE[: len(unique_cats)]))
        for cat in unique_cats:
            mask = categories == cat
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                label=cat,
                color=palette[cat],
                alpha=0.6,
                s=20,
            )
        ax.legend(fontsize=7, frameon=True)
    else:
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=20, color=MODULE_PALETTE[0])

    # Loading arrows
    loadings = pca.components_.T  # (n_features, 2)
    arrow_scale = np.abs(scores).max() * 0.8
    loading_norms = np.linalg.norm(loadings, axis=1, keepdims=True)
    loading_norms[loading_norms == 0] = 1.0
    scaled = loadings / loading_norms.max() * arrow_scale

    for i, col in enumerate(cols):
        ax.annotate(
            "",
            xy=(scaled[i, 0], scaled[i, 1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )
        ax.text(
            scaled[i, 0] * 1.08,
            scaled[i, 1] * 1.08,
            _short_name(col.removesuffix("_mean")),
            fontsize=6,
            ha="center",
            va="center",
            color="dimgray",
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("PCA Biplot")
    fig.set_tight_layout(True)
    return fig


# ── 3. Radar / polar chart ─────────────────────────────────────────────────

def category_radar(
    centroids_df: pd.DataFrame,
    axes: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Radar (polar) chart of category centroids across the 8 axes.

    Parameters
    ----------
    centroids_df : DataFrame
        One row per category.  Must contain a ``"category"`` column and
        one column per axis (``*_mean`` or raw axis name).
    axes : list[str], optional
        Subset of axes to plot (defaults to all 8).
    ax : matplotlib PolarAxes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Resolve columns
    mean_cols = _mean_cols()
    if all(c in centroids_df.columns for c in mean_cols):
        all_cols = mean_cols
    else:
        all_cols = [a for a in AXES if a in centroids_df.columns]

    if axes is not None:
        # Allow caller to pass short or full names
        col_set = set()
        for a in axes:
            if a in all_cols:
                col_set.add(a)
            elif f"{a}_mean" in all_cols:
                col_set.add(f"{a}_mean")
        all_cols = [c for c in all_cols if c in col_set]

    labels_display = [_short_name(c.removesuffix("_mean")) for c in all_cols]
    n_axes = len(all_cols)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), constrained_layout=True)
    else:
        fig = ax.get_figure()

    categories = centroids_df["category"].tolist() if "category" in centroids_df.columns else centroids_df.index.tolist()

    for idx, (_, row) in enumerate(centroids_df.iterrows()):
        values = row[all_cols].astype(float).tolist()
        values += values[:1]
        color = MODULE_PALETTE[idx % len(MODULE_PALETTE)]
        cat_label = categories[idx] if idx < len(categories) else f"Group {idx}"
        ax.plot(angles, values, linewidth=1.5, label=cat_label, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_display, fontsize=7)
    ax.set_title("Category Radar", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    return fig


# ── 3b. Category centroids line chart ──────────────────────────────────────

def category_centroids_line(
    centroids_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Line chart of category centroid profiles across the 8 axes.

    Each line connects a category's mean score on each axis, matching the
    paper's ``category_radar.png`` figure style.

    Parameters
    ----------
    centroids_df : DataFrame
        One row per category with a ``"category"`` column and axis score
        columns (``*_mean`` or raw axis names).
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean_cols = _mean_cols()
    if all(c in centroids_df.columns for c in mean_cols):
        all_cols = mean_cols
    else:
        all_cols = [a for a in AXES if a in centroids_df.columns]

    labels_display = [_short_name(c.removesuffix("_mean")) for c in all_cols]
    categories = centroids_df["category"].tolist() if "category" in centroids_df.columns else centroids_df.index.tolist()

    fig, ax = _ensure_ax(ax)

    x = np.arange(len(all_cols))
    for idx, (_, row) in enumerate(centroids_df.iterrows()):
        values = row[all_cols].astype(float).tolist()
        color = MODULE_PALETTE[idx % len(MODULE_PALETTE)]
        cat_label = categories[idx] if idx < len(categories) else f"Group {idx}"
        ax.plot(x, values, marker="o", markersize=4, linewidth=1.5,
                label=cat_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean axis score (-100 to +100)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Category Centroids")
    ax.legend(fontsize=7, loc="best", frameon=True)
    fig.set_tight_layout(True)
    return fig


# ── 4. Score distributions ─────────────────────────────────────────────────

def score_distributions(
    labels_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Small-multiples (2x4) histogram of each axis's score distribution.

    Parameters
    ----------
    labels_df : DataFrame
        One row per item with axis score columns.
    ax : matplotlib Axes, optional
        Ignored — a new 2x4 figure is always created.  Kept in the
        signature for API consistency.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = _resolve_axis_cols(labels_df)
    fig, axes_grid = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)

    for i, (col, cell_ax) in enumerate(zip(cols, axes_grid.flat)):
        values = labels_df[col].dropna()
        color = MODULE_PALETTE[i % len(MODULE_PALETTE)]
        cell_ax.hist(values, bins=30, color=color, edgecolor="white", alpha=0.85)
        cell_ax.set_title(_short_name(col.removesuffix("_mean")), fontsize=9)
        cell_ax.tick_params(labelsize=7)

    # Hide any surplus subplots
    for j in range(len(cols), len(axes_grid.flat)):
        axes_grid.flat[j].set_visible(False)

    fig.suptitle("Score Distributions", fontsize=12)
    return fig


# ── 5. Axis-pair scatter ────────────────────────────────────────────────────

def axis_pair_scatter(
    labels_df: pd.DataFrame,
    axis_x: str,
    axis_y: str,
    items_df: Optional[pd.DataFrame] = None,
    category_col: str = "category",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Scatter of two axes, optionally colored by category.

    Parameters
    ----------
    labels_df : DataFrame
        One row per item.
    axis_x, axis_y : str
        Column names (e.g. ``"axis_aequitas_libertas_mean"``).
    items_df : DataFrame, optional
        Merge source for *category_col*.
    category_col : str
        Column used for hue.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _ensure_ax(ax)

    # Resolve category series
    hue = None
    if items_df is not None and category_col in items_df.columns:
        if "item_id" in labels_df.columns and "item_id" in items_df.columns:
            cat_map = items_df.set_index("item_id")[category_col]
            hue = labels_df["item_id"].map(cat_map)
        elif category_col in labels_df.columns:
            hue = labels_df[category_col]
    elif category_col in labels_df.columns:
        hue = labels_df[category_col]

    if hue is not None:
        unique_cats = hue.dropna().unique()
        palette = dict(zip(unique_cats, MODULE_PALETTE[: len(unique_cats)]))
        for cat in unique_cats:
            mask = hue == cat
            ax.scatter(
                labels_df.loc[mask, axis_x],
                labels_df.loc[mask, axis_y],
                label=cat,
                color=palette[cat],
                alpha=0.6,
                s=20,
            )
        ax.legend(fontsize=7, frameon=True)
    else:
        ax.scatter(
            labels_df[axis_x],
            labels_df[axis_y],
            alpha=0.5,
            s=20,
            color=MODULE_PALETTE[0],
        )

    ax.set_xlabel(_short_name(axis_x.removesuffix("_mean")))
    ax.set_ylabel(_short_name(axis_y.removesuffix("_mean")))
    ax.set_title(f"{_short_name(axis_x.removesuffix('_mean'))}  vs  {_short_name(axis_y.removesuffix('_mean'))}")
    fig.set_tight_layout(True)
    return fig


# ── 6. R² leaderboard ──────────────────────────────────────────────────────

def r2_leaderboard(
    r2_results: pd.DataFrame,
    highlight_pair: Optional[tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Horizontal bar chart of 2-axis pair R² values.

    Parameters
    ----------
    r2_results : DataFrame
        Must contain columns ``"pair"`` (str label) and ``"r2"`` (float).
        Rows are sorted descending by R².
    highlight_pair : tuple[str, str], optional
        A pair to highlight in a contrasting colour (e.g. the default
        projection axes).
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = r2_results.sort_values("r2", ascending=True).copy()

    fig, ax = _ensure_ax(ax)

    colors = []
    highlight_label = None
    if highlight_pair is not None:
        highlight_label = " + ".join(
            _short_name(a.removesuffix("_mean")) for a in highlight_pair
        )

    for _, row in df.iterrows():
        if highlight_label and row["pair"] == highlight_label:
            colors.append(MODULE_PALETTE[1])
        else:
            colors.append(MODULE_PALETTE[0])

    ax.barh(df["pair"], df["r2"], color=colors, edgecolor="white")
    ax.set_xlabel("R²")
    ax.set_title("2-Axis Pair R² Leaderboard")
    ax.tick_params(axis="y", labelsize=7)
    fig.set_tight_layout(True)
    return fig
