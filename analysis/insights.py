"""
insights.py — Higher-level political structure analyses.

Four independent analyses:
  2.4.1  Category centroids and clustering
  2.4.2  Ideological convergence (filter bubble test)
  2.4.3  Cross-category contradictions
  2.4.4  Axis correlation structure and collinearity

All functions are pure; callers handle I/O.
"""
from __future__ import annotations

from typing import Optional

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analysis.label_io import AXES


# ── 2.4.1 — Category centroids and clustering ─────────────────────────────────

def category_centroids(
    labels_df: pd.DataFrame,
    items_df: pd.DataFrame,
    category_col: str = "category",
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute the mean 8D vector per category.

    Args:
        labels_df:    Clean labels (one row per item, columns = axis means).
        items_df:     Items DataFrame with item_id + category_col.
        category_col: Column in items_df identifying the category.
        axis_cols:    Axis columns to use (without _mean suffix).

    Returns:
        DataFrame with category + mean axis columns + n_items,
        sorted by n_items descending.
        If category_col is absent, returns a single-row "all" aggregate.
    """
    axis_cols = axis_cols or AXES

    merged = (
        labels_df.merge(items_df[["item_id", category_col]], on="item_id", how="left")
        if category_col in items_df.columns
        else labels_df.assign(**{category_col: "all"})
    )

    axis_mean_cols = [f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns]
    if not axis_mean_cols:
        axis_mean_cols = [a for a in axis_cols if a in labels_df.columns]

    rows = []
    for cat, grp in merged.groupby(category_col):
        row: dict = {category_col: cat, "n_items": len(grp)}
        for col in axis_mean_cols:
            row[col] = float(grp[col].astype(float).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_items", ascending=False).reset_index(drop=True)


def cluster_politikas(
    labels_df: pd.DataFrame,
    k_range: tuple[int, int] = (3, 8),
    axis_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> dict:
    """
    K-means clustering in 8D label space. Selects k via silhouette score.

    Args:
        labels_df:    Clean labels (one row per item, axis mean columns).
        k_range:      (min_k, max_k) inclusive range to search.
        axis_cols:    Axis columns (without _mean suffix).
        random_state: Seed.

    Returns dict with:
        best_k             int
        silhouette_scores  dict[k -> score]
        labels_df          input DataFrame with `cluster` column added
        cluster_centers    DataFrame (k x axes)
    """
    axis_cols = axis_cols or AXES
    mean_cols = [f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns]
    if not mean_cols:
        mean_cols = [a for a in axis_cols if a in labels_df.columns]

    data = labels_df[mean_cols].astype(float).dropna().to_numpy()
    valid_idx = labels_df[mean_cols].astype(float).dropna().index

    best_k = k_range[0]
    best_score = -1.0
    sil_scores: dict = {}

    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(data):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = km.fit_predict(data)
        if len(set(cluster_labels)) < 2:
            continue
        score = float(silhouette_score(data, cluster_labels))
        sil_scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    km_best = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    cluster_assignments = km_best.fit_predict(data)

    result_df = labels_df.copy()
    result_df["cluster"] = np.nan
    result_df.loc[valid_idx, "cluster"] = cluster_assignments

    centers = pd.DataFrame(km_best.cluster_centers_, columns=mean_cols)
    centers.index.name = "cluster"

    return {
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "labels_df": result_df,
        "cluster_centers": centers,
    }


# ── 2.4.2 — Ideological convergence (filter bubble test) ─────────────────────

def compute_landing_convergence(
    interactions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
    min_votes: int = 100,
    window_size: int = 10,
    user_col: str = "user_id",
    item_id_col: str = "item_id",
    action_col: str = "action",
) -> dict:
    """
    Test whether the ideological diversity of proposals that users are exposed to
    decreases over time — the filter bubble / convergence hypothesis.

    Only analyses users with >= min_votes interactions (default 100).

    For each qualifying user in the cohort, computes a rolling-window variance of
    the 8D vectors of proposals they were exposed to. Decreasing variance at the
    cohort level would indicate the platform is serving an ideologically narrowing feed.

    Returns a dict with:
        retention_funnel       {vote_threshold: n_users}
        cohort_size            int
        convergence_curve      list of {interaction_bin, mean_variance, n_users}
        levene_sig_fraction    fraction of users with Levene p < 0.05
        levene_conv_fraction   fraction of those who are converging (not just changing)
        per_user               list of per-user summary dicts
    """
    axis_cols = axis_cols or AXES
    mean_cols = {a: f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns}
    if not mean_cols:
        mean_cols = {a: a for a in axis_cols if a in labels_df.columns}
    if not mean_cols:
        return {"error": "No axis columns found in labels_df"}

    inter = interactions_df.copy()
    inter.columns = [c.strip().lower() for c in inter.columns]
    uid_col = next((c for c in inter.columns if "user" in c), None)
    pid_col = next((c for c in inter.columns if "politics_id" in c or c == item_id_col), None)
    act_col = next((c for c in inter.columns if c == action_col or "action" in c), None)
    ts_col  = next((c for c in inter.columns if "creat" in c or "time" in c), None)

    if not all([uid_col, pid_col, act_col]):
        return {"error": f"Missing required columns. Found: {inter.columns.tolist()}"}

    votes_per_user = inter.groupby(uid_col).size()
    retention_funnel = {
        t: int((votes_per_user >= t).sum())
        for t in [5, 10, 20, 50, 100, 200, 500]
    }
    total_users = len(votes_per_user)

    cohort_users = set(votes_per_user[votes_per_user >= min_votes].index)
    if not cohort_users:
        return {
            "retention_funnel": retention_funnel,
            "total_users": total_users,
            "cohort_size": 0,
            "error": f"No users with >= {min_votes} votes",
        }

    inter_cohort = inter[inter[uid_col].isin(cohort_users)].copy()
    if ts_col:
        try:
            inter_cohort[ts_col] = pd.to_datetime(inter_cohort[ts_col], errors="coerce")
            inter_cohort = inter_cohort.sort_values([uid_col, ts_col])
        except Exception:
            inter_cohort = inter_cohort.sort_values(uid_col)
    else:
        inter_cohort = inter_cohort.sort_values(uid_col)

    item_scores = labels_df.set_index("item_id")[list(mean_cols.values())].to_dict("index")

    per_user = []
    all_trajectories: list[tuple[int, float]] = []

    for uid, grp in inter_cohort.groupby(uid_col):
        grp = grp.reset_index(drop=True)
        score_seq = []
        for _, row in grp.iterrows():
            iid = row[pid_col]
            scores = item_scores.get(iid)
            if scores is None:
                continue
            vec = np.array([float(scores.get(mc, 0)) for mc in mean_cols.values()], dtype=float)
            score_seq.append(vec)

        if len(score_seq) < window_size * 2:
            continue

        score_arr = np.array(score_seq)
        T = len(score_arr)

        variances = []
        for i in range(window_size - 1, T):
            window = score_arr[max(0, i - window_size + 1): i + 1]
            var = float(window.std(axis=0).mean())
            variances.append(var)
            all_trajectories.append((i, var))

        mid = len(variances) // 2
        first_vars = variances[:mid]
        second_vars = variances[mid:]
        levene_p = 1.0
        converging = False
        if len(first_vars) >= 3 and len(second_vars) >= 3:
            try:
                _, levene_p = scipy_stats.levene(first_vars, second_vars)
                converging = float(np.mean(second_vars)) < float(np.mean(first_vars))
            except Exception:
                pass

        per_user.append({
            "user_id": uid,
            "n_votes": len(grp),
            "mean_variance_first_half": float(np.mean(first_vars)) if first_vars else float("nan"),
            "mean_variance_second_half": float(np.mean(second_vars)) if second_vars else float("nan"),
            "levene_p": float(levene_p),
            "levene_significant": levene_p < 0.05,
            "converging": converging,
        })

    if not per_user:
        return {
            "retention_funnel": retention_funnel,
            "total_users": total_users,
            "cohort_size": len(cohort_users),
            "error": "No users had enough matched interactions after join",
        }

    traj_df = pd.DataFrame(all_trajectories, columns=["interaction_idx", "variance"])
    bin_size = 10
    traj_df["bin"] = (traj_df["interaction_idx"] // bin_size) * bin_size
    curve = (
        traj_df.groupby("bin")
        .agg(mean_variance=("variance", "mean"), n_users=("variance", "count"))
        .reset_index()
        .rename(columns={"bin": "interaction_bin"})
    )
    convergence_curve = curve.to_dict("records")

    levene_sig = [u for u in per_user if u["levene_significant"]]
    levene_conv = [u for u in per_user if u["levene_significant"] and u["converging"]]

    return {
        "retention_funnel": retention_funnel,
        "total_users": total_users,
        "cohort_size": len(per_user),
        "min_votes_threshold": min_votes,
        "convergence_curve": convergence_curve,
        "levene_sig_fraction": len(levene_sig) / max(len(per_user), 1),
        "levene_conv_fraction": len(levene_conv) / max(len(per_user), 1),
        "per_user": per_user,
    }


# ── 2.4.3 — Cross-category contradictions ─────────────────────────────────────

def cross_category_contradictions(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
    category_col: str = "category",
    user_col: str = "user_id",
    politics_id_col: str = "politics_id",
    action_col: str = "action",
    item_id_col: str = "item_id",
    min_votes: int = 5,
) -> pd.DataFrame:
    """
    Find cross-category contradictions: aggregate voting patterns where a cohort's
    position on the same axis reverses between policy categories.

    A contradiction occurs when:
        sign(cohort position on axis A in category X) !=
        sign(cohort position on axis A in category Y)
    where both positions are computed from >= min_votes interactions per user.

    Returns:
        DataFrame with columns: category_x, category_y, axis,
        n_contradictions, fraction_users — sorted by n_contradictions.
    """
    axis_cols = axis_cols or AXES
    mean_cols = [f"{a}_mean" for a in axis_cols if f"{a}_mean" in items_df.columns]
    if not mean_cols:
        mean_cols = [a for a in axis_cols if a in items_df.columns]

    if category_col not in items_df.columns:
        warnings.warn(f"Column `{category_col}` not found in items_df; skipping.")
        return pd.DataFrame()

    inter = interactions_df.copy()
    inter["_weight"] = inter[action_col].map({"AGREE": 1, "DISAGREE": -1}).fillna(0)

    merged = inter.merge(
        items_df[[item_id_col, category_col] + mean_cols],
        left_on=politics_id_col,
        right_on=item_id_col,
        how="inner",
    )

    user_positions = (
        merged.groupby([user_col, category_col])
        .apply(lambda g: pd.Series({
            **{col.replace("_mean", "_pos"): float((g[col].astype(float) * g["_weight"]).sum())
               for col in mean_cols},
            "n_votes": len(g),
        }))
        .reset_index()
    )
    user_positions = user_positions[user_positions["n_votes"] >= min_votes]
    pos_cols = [col.replace("_mean", "_pos") for col in mean_cols]
    n_all_users = len(inter[user_col].unique())

    contradiction_rows = []
    for ax, pos_col in zip(axis_cols[:len(pos_cols)], pos_cols):
        if pos_col not in user_positions.columns:
            continue
        pivot = user_positions.pivot_table(
            index=user_col, columns=category_col, values=pos_col, aggfunc="first",
        )
        sign_mat = np.sign(pivot.values)
        cat_list = list(pivot.columns)
        for i, cat_x in enumerate(cat_list):
            for j, cat_y in enumerate(cat_list):
                if j <= i:
                    continue
                sx = sign_mat[:, i]
                sy = sign_mat[:, j]
                valid_mask = (sx != 0) & (sy != 0) & ~np.isnan(sx) & ~np.isnan(sy)
                count = int((sx[valid_mask] != sy[valid_mask]).sum())
                if count > 0:
                    contradiction_rows.append({
                        "category_x": cat_x,
                        "category_y": cat_y,
                        "axis": ax,
                        "n_contradictions": count,
                        "fraction_users": count / n_all_users,
                    })

    if not contradiction_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(contradiction_rows)
        .sort_values("n_contradictions", ascending=False)
        .reset_index(drop=True)
    )


# ── 2.4.4 — Axis correlation structure ───────────────────────────────────────

def axis_correlation_matrix(
    labels_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute the 8x8 Pearson correlation matrix of mean axis scores.

    Returns:
        DataFrame (8x8) of Pearson correlations, indexed by axis name.
    """
    axis_cols = axis_cols or AXES
    mean_cols = [f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns]
    if not mean_cols:
        mean_cols = [a for a in axis_cols if a in labels_df.columns]

    data = labels_df[mean_cols].astype(float).dropna()
    corr = data.corr(method="pearson")
    corr.columns = [c.replace("_mean", "") for c in corr.columns]
    corr.index = corr.index.str.replace("_mean", "")
    return corr


def axis_vif(
    labels_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each axis.

    VIF > 5 = moderate collinearity; VIF > 10 = severe.
    Axes with high VIF are candidates for removal in a reduced model.

    Returns:
        DataFrame with columns: axis, vif — sorted by vif descending.
    """
    from sklearn.linear_model import LinearRegression

    axis_cols = axis_cols or AXES
    mean_cols = [f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns]
    if not mean_cols:
        mean_cols = [a for a in axis_cols if a in labels_df.columns]

    data = labels_df[mean_cols].astype(float).dropna().to_numpy()

    vif_values = []
    for i in range(data.shape[1]):
        X_others = np.delete(data, i, axis=1)
        y = data[:, i]
        r2 = LinearRegression().fit(X_others, y).score(X_others, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vif_values.append({"axis": mean_cols[i].replace("_mean", ""), "vif": round(float(vif), 2)})

    return pd.DataFrame(vif_values).sort_values("vif", ascending=False).reset_index(drop=True)
