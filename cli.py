#!/usr/bin/env python3
"""
Minimal public CLI wrappers for core analysis tasks.

These wrappers intentionally avoid private config conventions.
Input and output paths are passed explicitly so the tool works
on any dataset, not just Politikea's private corpus.

Usage:
    python cli.py clean --input annotations.parquet --output labels_clean.parquet
    python cli.py validate --labels labels_clean.parquet --items proposals.csv --output-dir results/
    python cli.py dimensionality --labels labels_clean.parquet --output-dir results/
    python cli.py insights --labels labels_clean.parquet --items proposals.csv --output-dir results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_toolkit_to_path() -> None:
    """Ensure the toolkit/ directory (containing the analysis/ package) is on sys.path.

    This makes the CLI fully self-contained: it does not depend on the private
    ideological_research/src/ tree and works from any working directory.
    """
    toolkit_dir = Path(__file__).resolve().parent
    if str(toolkit_dir) not in sys.path:
        sys.path.insert(0, str(toolkit_dir))


_add_toolkit_to_path()

from analysis.cleaning import compute_item_stability, filter_by_confidence, flag_valid_items
from analysis.label_io import load_all_runs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Politikea ideological mapping toolkit — data-agnostic wrappers.\n"
            "See docs/methodology.md for an explanation of each step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── clean ──────────────────────────────────────────────────────────────────
    p_clean = sub.add_parser(
        "clean",
        help="Confidence-filter and aggregate multi-run labels.",
        description=(
            "Aggregates multiple GPT scoring runs per proposal into a single stable\n"
            "label, then flags valid items based on reliability thresholds.\n\n"
            "Input:  consolidated multi-run annotations (parquet)\n"
            "Output: labels_clean.parquet with per-item ICC, sign agreement, validity flag"
        ),
    )
    p_clean.add_argument("--input", required=True, help="Path to consolidated annotations parquet.")
    p_clean.add_argument("--output", required=True, help="Path to write labels_clean parquet.")
    p_clean.add_argument(
        "--global-conf", type=float, default=0.5,
        help="Minimum mean confidence across all axes (default: 0.5).",
    )
    p_clean.add_argument(
        "--axis-conf", type=float, default=0.4,
        help="Minimum per-axis confidence (default: 0.4).",
    )
    p_clean.add_argument(
        "--min-stable-axes", type=int, default=6,
        help="Minimum number of stable axes for item to be valid (default: 6/8).",
    )
    p_clean.add_argument(
        "--min-runs", type=int, default=10,
        help="Minimum number of scoring runs required (default: 10).",
    )
    p_clean.add_argument(
        "--std-threshold", type=float, default=30.0,
        help="Maximum per-axis score std for axis to be considered stable (default: 30).",
    )
    p_clean.add_argument(
        "--sign-agreement-threshold", type=float, default=0.6,
        help="Minimum sign agreement fraction for axis stability (default: 0.6).",
    )

    # ── validate ───────────────────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="Measure text/8D similarity correlation, linguistic encoding (H3), and deduplication (H4).",
        description=(
            "Runs the full validation suite:\n"
            "  H1/H2: Spearman text→8D and 8D→text correlation\n"
            "  H3:    Per-axis linguistic encoding (embedding projection)\n"
            "  H4:    Proposal deduplication via joint text+8D signal\n\n"
            "Input:  labels_clean.parquet + proposals CSV/parquet with a 'text' column\n"
            "Output: validation_report.md + pair-level diagnostics"
        ),
    )
    p_val.add_argument("--labels", required=True, help="Path to labels_clean.parquet.")
    p_val.add_argument("--items", required=True, help="Path to proposals CSV/parquet with a 'text' column.")
    p_val.add_argument(
        "--output-dir", required=True,
        help="Directory for validation report and figures.",
    )
    p_val.add_argument(
        "--sample-pairs", type=int, default=500,
        help="Number of random pairs to embed (default: 500; use 0 for all — slow).",
    )
    p_val.add_argument(
        "--nn-k", type=int, default=50,
        help="Number of nearest neighbours per item (default: 50).",
    )
    p_val.add_argument(
        "--embedding-model",
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformers model for text embeddings.",
    )
    p_val.add_argument(
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for CI (default: 1000).",
    )
    p_val.add_argument(
        "--n-perm", type=int, default=1000,
        help="Permutation resamples for p-value (default: 1000).",
    )
    p_val.add_argument(
        "--text-col", default="text",
        help="Column name containing proposal text (default: 'text').",
    )
    p_val.add_argument(
        "--text-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for text-based deduplication (H4, default: 0.85).",
    )
    p_val.add_argument(
        "--sim-8d-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for 8D ideological deduplication (H4, default: 0.85).",
    )
    p_val.add_argument(
        "--skip-h3", action="store_true",
        help="Skip the per-axis linguistic encoding analysis (H3) — faster for quick checks.",
    )
    p_val.add_argument(
        "--skip-h4", action="store_true",
        help="Skip the proposal deduplication clustering (H4).",
    )

    # ── dimensionality ─────────────────────────────────────────────────────────
    p_dim = sub.add_parser(
        "dimensionality",
        help="PCA, axis collinearity, and 2-axis reconstruction R² analysis.",
        description=(
            "Runs PCA on the 8D label space, measures VIF for each axis,\n"
            "and evaluates how much variance each 2-axis pair reconstructs.\n\n"
            "Input:  labels_clean.parquet\n"
            "Output: dimensionality_report.md + figures (pca_scree, axis_r2_heatmap)"
        ),
    )
    p_dim.add_argument("--labels", required=True, help="Path to labels_clean.parquet.")
    p_dim.add_argument(
        "--output-dir", required=True,
        help="Directory for dimensionality report and figures.",
    )
    p_dim.add_argument(
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for R² stability estimates (default: 1000).",
    )
    p_dim.add_argument(
        "--skip-predictive", action="store_true",
        help="Skip AUC predictive analysis (requires interaction data).",
    )
    p_dim.add_argument(
        "--interactions",
        default=None,
        help="Optional path to interactions CSV (item_id, user_id, vote) for AUC analysis.",
    )

    # ── insights ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser(
        "insights",
        help="Category centroids, axis correlation heatmap, and K-means clustering.",
        description=(
            "Runs the higher-level political structure analyses:\n"
            "  - Category centroids: mean 8D vector per policy category\n"
            "  - Axis correlation matrix + VIF table\n"
            "  - K-means clustering in 8D label space\n\n"
            "Input:  labels_clean.parquet + items CSV/parquet with a 'category' column\n"
            "Output: insights_report.md"
        ),
    )
    p_ins.add_argument("--labels", required=True, help="Path to labels_clean.parquet.")
    p_ins.add_argument("--items", required=True, help="Path to proposals CSV/parquet with a 'category' column.")
    p_ins.add_argument(
        "--output-dir", required=True,
        help="Directory for the insights report.",
    )
    p_ins.add_argument(
        "--category-col", default="category",
        help="Column in items file identifying the policy category (default: 'category').",
    )
    p_ins.add_argument(
        "--k-range", default="3,8",
        help="K-means cluster count range to search, e.g. '3,8' (default: 3,8).",
    )

    return p


def _load_items(path: str) -> "pd.DataFrame":
    """Load a proposals file from CSV or parquet."""
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(p).to_pandas()
    return pd.read_csv(p)


def cmd_clean(args: argparse.Namespace) -> int:
    df = load_all_runs(args.input)
    filtered = filter_by_confidence(df, global_threshold=args.global_conf, axis_threshold=args.axis_conf)
    stability = compute_item_stability(filtered)
    stability = flag_valid_items(
        stability,
        min_stable_axes=args.min_stable_axes,
        min_runs=args.min_runs,
        std_threshold=args.std_threshold,
        sign_agreement_threshold=args.sign_agreement_threshold,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stability.to_parquet(out, index=False)
    n_valid = stability["valid"].sum() if "valid" in stability.columns else len(stability)
    n_total = len(stability)
    print(f"[clean] {n_valid:,} / {n_total:,} items valid ({100 * n_valid / n_total:.1f}%)")
    print(f"[clean] Wrote -> {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    import pandas as pd
    from analysis.label_io import AXES
    from analysis.similarity import (
        spearman_text_vs_8d,
        spearman_8d_vs_text,
        nearest_neighbors_8d,
        embedding_axis_projection,
        lexical_axis_anchors,
        find_proposal_clusters,
        top_clusters_by_engagement,
    )

    labels_df = pd.read_parquet(args.labels)
    items_df  = _load_items(args.items)

    valid_labels = labels_df[labels_df["valid"]] if "valid" in labels_df.columns else labels_df

    axis_mean_cols = [c for c in valid_labels.columns if c.startswith("axis_") and c.endswith("_mean")]
    if not axis_mean_cols:
        axis_mean_cols = [c for c in valid_labels.columns if c.startswith("axis_")]

    text_col = args.text_col if args.text_col in items_df.columns else "text"
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_n = args.sample_pairs if args.sample_pairs > 0 else None
    embedding_models = [args.embedding_model]

    print(f"[validate] Valid items: {len(valid_labels):,}")
    print(f"[validate] Text column: '{text_col}'")
    print(f"[validate] Embedding model: {args.embedding_model}")

    # H1: Text → 8D correlation
    primary_result = spearman_text_vs_8d(
        labels_df=valid_labels,
        items_df=items_df,
        text_col=text_col,
        axis_cols=axis_mean_cols,
        embedding_models=embedding_models,
        nn_k=args.nn_k,
        sample_n=sample_n,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        use_ollama_summaries=False,
    )
    print(f"[validate] Primary ρ = {primary_result['spearman_r']:.4f}, verdict: {primary_result['verdict']}")

    # 8D → Text (diagnostic direction)
    pairs_df = nearest_neighbors_8d(valid_labels, k=args.nn_k, axis_cols=axis_mean_cols)
    secondary_result = spearman_8d_vs_text(
        pairs_df=pairs_df,
        items_df=items_df,
        text_col=text_col,
        axis_cols=axis_mean_cols,
        embedding_models=embedding_models,
        sample_n=sample_n,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
    )

    # H3: Per-axis linguistic encoding
    axis_projections = None
    if not args.skip_h3:
        print("[validate] Running H3: per-axis linguistic encoding...")
        raw_axes = [c.replace("_mean", "") for c in axis_mean_cols]
        axis_projections = embedding_axis_projection(
            labels_df=valid_labels,
            items_df=items_df,
            text_col=text_col,
            axis_cols=raw_axes,
            model_name=args.embedding_model,
        )
        lexical_result = lexical_axis_anchors(
            labels_df=valid_labels,
            items_df=items_df,
            text_col=text_col,
            axis_cols=raw_axes,
        )
        print(f"[validate] H3 complete — {len(axis_projections)} axes analysed")
    else:
        lexical_result = None

    # H4: Proposal deduplication
    cluster_result = None
    if not args.skip_h4:
        print("[validate] Running H4: proposal deduplication clustering...")
        full_pairs_df = primary_result.get("full_pairs_df") or primary_result.get("pairs_df")
        if full_pairs_df is not None and len(full_pairs_df) > 0:
            raw_cluster = find_proposal_clusters(
                pairs_df=full_pairs_df,
                text_threshold=args.text_threshold,
                sim_8d_threshold=args.sim_8d_threshold,
            )
            cluster_result = top_clusters_by_engagement(
                clusters=raw_cluster,
                items_df=items_df,
                text_col=text_col,
                top_n=20,
            )
            print(f"[validate] H4 complete — {len(cluster_result)} clusters found")
        else:
            print("[validate] H4 skipped: no pairs_df in primary result")

    # Write report
    report_path = out_dir / "validation_report.md"
    _write_validation_report(
        primary_result=primary_result,
        secondary_result=secondary_result,
        axis_projections=axis_projections,
        lexical_anchors=lexical_result,
        cluster_result=cluster_result,
        out_path=report_path,
    )
    print(f"[validate] Report -> {report_path}")
    return 0


def _write_validation_report(
    primary_result: dict,
    secondary_result: dict,
    out_path: Path,
    axis_projections: "dict | None" = None,
    lexical_anchors: "dict | None" = None,
    cluster_result: "list[dict] | None" = None,
) -> None:
    """Write a structured Markdown validation report."""
    lines: list[str] = []
    a = lines.append

    a("# Semantic Validity Report")
    a("")
    a("## 1. What This Measures")
    a("")
    a("**Core claim**: text similarity and 8D ideological similarity are each noisy signals")
    a("for identifying related proposals. Their *intersection* — pairs that score high on both —")
    a("is a reliable near-duplicate detector. This enables proposal deduplication at scale")
    a("without human review.")
    a("")

    # Section 2: Global correlations
    a("## 2. Global Correlation")
    a("")
    pr = primary_result
    sr = secondary_result
    a("| Direction | ρ | 95% CI | p | perm-p | N pairs | Verdict |")
    a("|-----------|---|--------|---|--------|---------|---------|")
    a(f"| Text → 8D (primary) | {pr['spearman_r']:.3f} | "
      f"[{pr['rho_ci_low']:.3f}, {pr['rho_ci_high']:.3f}] | "
      f"{pr['p_value']:.4f} | {pr['permutation_p']:.4f} | "
      f"{pr['n_pairs']:,} | **{pr['verdict']}** |")
    a(f"| 8D → Text (diagnostic) | {sr['spearman_r']:.3f} | "
      f"[{sr['rho_ci_low']:.3f}, {sr['rho_ci_high']:.3f}] | "
      f"{sr['p_value']:.4f} | {sr['permutation_p']:.4f} | "
      f"{sr['n_pairs']:,} | **{sr['verdict']}** |")
    a("")

    # Section 3: Pair examples
    if "pairs_df" in pr and pr["pairs_df"] is not None and len(pr["pairs_df"]) > 0:
        p_df = pr["pairs_df"].copy()
        text_hi  = p_df["cosine_similarity_text"] >= p_df["cosine_similarity_text"].quantile(0.80)
        sim8d_lo = p_df["cosine_similarity_8d"] <= p_df["cosine_similarity_8d"].quantile(0.20)
        sim8d_hi = p_df["cosine_similarity_8d"] >= p_df["cosine_similarity_8d"].quantile(0.80)
        text_lo  = p_df["cosine_similarity_text"] <= p_df["cosine_similarity_text"].quantile(0.20)

        discord_hi_text = p_df[text_hi & sim8d_lo].sort_values(
            ["cosine_similarity_text", "cosine_similarity_8d"], ascending=[False, True]
        ).head(5)
        discord_hi_8d = p_df[sim8d_hi & text_lo].sort_values(
            ["cosine_similarity_8d", "cosine_similarity_text"], ascending=[False, True]
        ).head(5)

        a("## 3. Why Each Signal Alone Fails")
        a("")
        a("### 3a. Text similar, ideologically opposite (noise mode 1)")
        a("")
        a("| Text sim | 8D sim | Proposal A | Proposal B |")
        a("|----------|--------|------------|------------|")
        for _, row in discord_hi_text.iterrows():
            ta = str(row.get("text_a", ""))[:90].replace("|", "/")
            tb = str(row.get("text_b", ""))[:90].replace("|", "/")
            a(f"| {row.get('cosine_similarity_text', float('nan')):.3f} "
              f"| {row.get('cosine_similarity_8d', float('nan')):.3f} "
              f"| {ta} | {tb} |")
        a("")
        a("### 3b. Ideologically aligned, differently phrased (noise mode 2)")
        a("")
        a("| 8D sim | Text sim | Proposal A | Proposal B |")
        a("|--------|----------|------------|------------|")
        for _, row in discord_hi_8d.iterrows():
            ta = str(row.get("text_a", ""))[:90].replace("|", "/")
            tb = str(row.get("text_b", ""))[:90].replace("|", "/")
            a(f"| {row.get('cosine_similarity_8d', float('nan')):.3f} "
              f"| {row.get('cosine_similarity_text', float('nan')):.3f} "
              f"| {ta} | {tb} |")
        a("")

    # Section 4: H3 — Axis-level linguistic encoding
    if axis_projections:
        a("## 4. Per-Axis Linguistic Encoding (H3)")
        a("")
        a("For each axis, a linguistic direction vector is built as the mean-embedding difference")
        a("between high-scoring (top 25%) and low-scoring (bottom 25%) items. Spearman ρ measures")
        a("how well that geometric direction predicts the actual axis score.")
        a("")
        a("| # | Axis | ρ | 95% CI | p | n | Signal |")
        a("|---|------|---|--------|---|---|--------|")
        short = lambda x: x.replace("axis_", "").replace("_", " ")
        for idx, (axis, r) in enumerate(
            sorted(axis_projections.items(), key=lambda x: -x[1]["rho"]), 1
        ):
            ci = f"[{r['rho_ci_low']:.3f}, {r['rho_ci_high']:.3f}]"
            a(f"| {idx} | {short(axis)} | {r['rho']:.3f} | {ci} | "
              f"{r['p']:.4f} | {r['n_items']} | {r['interpretation']} |")
        a("")
        a("*Table: Per-axis linguistic encoding strength (H3).*")
        a("")
        strong = [short(ax) for ax, r in axis_projections.items() if r["rho"] >= 0.25]
        latent = [short(ax) for ax, r in axis_projections.items() if r["rho"] < 0.1]
        if strong:
            a(f"**Axes with vocabulary signal** ({len(strong)}/{len(axis_projections)}): "
              + ", ".join(f"`{w}`" for w in strong))
            a("")
        if latent:
            a(f"**Ideologically latent axes** (not expressed in vocabulary, "
              f"{len(latent)}/{len(axis_projections)}): "
              + ", ".join(f"`{w}`" for w in latent))
            a("")

    # Section 5: H2 — Vocabulary signatures
    if lexical_anchors:
        a("## 5. Vocabulary Signatures Per Axis (H2)")
        a("")
        short = lambda x: x.replace("axis_", "").replace("_", " ")
        for axis, data in lexical_anchors.items():
            rho_str = ""
            if axis_projections and axis in axis_projections:
                rho_str = f" (H3 ρ={axis_projections[axis]['rho']:.3f})"
            a(f"**{short(axis)}**{rho_str}")
            a(f"  + pole: {', '.join(f'`{w}`' for w in data['pos_words'][:6])}")
            a(f"  − pole: {', '.join(f'`{w}`' for w in data['neg_words'][:6])}")
            a("")

    # Section 6: H4 — Proposal deduplication
    if cluster_result:
        a("## 6. Proposal Deduplication (H4)")
        a("")
        total_clusters = len(cluster_result)
        total_items    = sum(c["cluster_size"] for c in cluster_result)
        a(f"**{total_clusters} clusters** containing {total_items} proposals in total.")
        a("")
        a("| Rank | Cluster size | Text sim | 8D sim | Anchor proposal |")
        a("|------|-------------|----------|--------|-----------------|")
        for rank, cluster in enumerate(cluster_result, 1):
            anchor_a = cluster["anchor_a_text"][:80].replace("|", "/").replace("\n", " ")
            a(f"| {rank} | {cluster['cluster_size']} | "
              f"{cluster['anchor_text_sim']:.3f} | {cluster['anchor_8d_sim']:.3f} | {anchor_a}… |")
        a("")

    a("_Report generated by `cli.py validate`_")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_dimensionality(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    from analysis.label_io import AXES
    from analysis.dimensionality import (
        pca_analysis,
        reconstruction_r2_all_subsets,
    )

    labels_df = pd.read_parquet(args.labels)
    valid_labels = labels_df[labels_df["valid"]] if "valid" in labels_df.columns else labels_df

    axis_mean_cols = [c for c in valid_labels.columns if c.startswith("axis_") and c.endswith("_mean")]
    if not axis_mean_cols:
        axis_mean_cols = [c for c in valid_labels.columns if c.startswith("axis_")]

    X = valid_labels[axis_mean_cols].dropna()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dimensionality] Items with complete 8D labels: {len(X):,}")

    pca_result = pca_analysis(X, n_components=min(8, len(X.columns)))
    r2_result  = reconstruction_r2_all_subsets(X, axis_cols=list(X.columns), n_boot=args.n_boot)

    report_path = out_dir / "dimensionality_report.md"
    with open(report_path, "w") as f:
        f.write("# Dimensionality Report\n\n")
        f.write(f"Items analysed: {len(X):,}\n\n")
        scree = pca_result.get("scree_df")
        if scree is not None:
            f.write("## PCA Variance Explained\n\n")
            f.write("| Component | Explained Variance | Cumulative |\n")
            f.write("|-----------|-------------------|------------|\n")
            for i, row in scree.iterrows():
                f.write(
                    f"| PC{int(i) + 1} "
                    f"| {row.get('explained_variance_ratio', 0):.3f} "
                    f"| {row.get('cumulative_variance', 0):.3f} |\n"
                )
        if r2_result:
            top = sorted(r2_result, key=lambda x: x.get("mean_r2", 0), reverse=True)[:10]
            f.write("\n## Top 10 Two-Axis Pairs by Reconstruction R²\n\n")
            f.write("| Rank | Pair | Mean R² |\n")
            f.write("|------|------|---------|\n")
            for i, row in enumerate(top, 1):
                f.write(f"| {i} | {row.get('pair', '')} | {row.get('mean_r2', 0):.4f} |\n")
        f.write("\n_Report generated by `cli.py dimensionality`_\n")

    print(f"[dimensionality] Report -> {report_path}")
    return 0


def cmd_insights(args: argparse.Namespace) -> int:
    import pandas as pd
    from analysis.insights import (
        category_centroids,
        axis_correlation_matrix,
        axis_vif,
        cluster_politikas,
    )

    labels_df = pd.read_parquet(args.labels)
    items_df  = _load_items(args.items)

    valid_labels = labels_df[labels_df["valid"]] if "valid" in labels_df.columns else labels_df

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[insights] Valid items: {len(valid_labels):,}")

    # Category centroids
    centroids = category_centroids(valid_labels, items_df, category_col=args.category_col)
    print(f"[insights] {len(centroids)} categories found")

    # Axis correlation and VIF
    corr_matrix = axis_correlation_matrix(valid_labels)
    vif_df      = axis_vif(valid_labels)

    # K-means clustering
    k_min, k_max = [int(x) for x in args.k_range.split(",")]
    cluster_result = cluster_politikas(valid_labels, k_range=(k_min, k_max))
    print(f"[insights] Best k = {cluster_result['best_k']}")

    # Write report
    report_path = out_dir / "insights_report.md"
    short = lambda x: x.replace("axis_", "").replace("_", " ")

    with open(report_path, "w") as f:
        f.write("# Insights Report\n\n")

        f.write("## 1. Category Centroids\n\n")
        f.write("Mean 8D ideological vector per policy category.\n\n")
        axis_cols = [c for c in centroids.columns if c.endswith("_mean")]
        header = "| Category | n |" + "".join(f" {short(c)} |" for c in axis_cols)
        f.write(header + "\n")
        f.write("|" + "---|" * (len(axis_cols) + 2) + "\n")
        for _, row in centroids.iterrows():
            vals = "".join(f" {row[c]:+.1f} |" for c in axis_cols)
            f.write(f"| {row.get('category', '—')} | {int(row['n_items'])} |{vals}\n")
        f.write("\n")

        f.write("## 2. Axis Correlation Matrix\n\n")
        f.write("Pearson correlations between the 8 axis mean scores.\n\n")
        if not corr_matrix.empty:
            cols = list(corr_matrix.columns)
            f.write("| |" + "".join(f" {short(c)} |" for c in cols) + "\n")
            f.write("|---|" + "---|" * len(cols) + "\n")
            for row_name in corr_matrix.index:
                row_vals = "".join(f" {corr_matrix.loc[row_name, c]:+.2f} |" for c in cols)
                f.write(f"| {short(str(row_name))} |{row_vals}\n")
        f.write("\n")

        f.write("## 3. Axis VIF (Collinearity)\n\n")
        f.write("VIF > 5 indicates moderate collinearity; > 10 is severe.\n\n")
        f.write("| Axis | VIF |\n|------|-----|\n")
        for _, row in vif_df.iterrows():
            f.write(f"| {short(str(row['axis']))} | {row['vif']:.2f} |\n")
        f.write("\n")

        f.write("## 4. K-Means Clustering\n\n")
        f.write(f"Best k = **{cluster_result['best_k']}** (selected by silhouette score)\n\n")
        sil = cluster_result["silhouette_scores"]
        f.write("| k | Silhouette |\n|---|------------|\n")
        for k, score in sorted(sil.items()):
            marker = " ← best" if k == cluster_result["best_k"] else ""
            f.write(f"| {k} | {score:.4f}{marker} |\n")
        f.write("\n")
        centers = cluster_result["cluster_centers"]
        axis_cols_c = list(centers.columns)
        f.write("### Cluster Centers\n\n")
        f.write("| Cluster |" + "".join(f" {short(c)} |" for c in axis_cols_c) + "\n")
        f.write("|---|" + "---|" * len(axis_cols_c) + "\n")
        for i, row in centers.iterrows():
            vals = "".join(f" {row[c]:+.1f} |" for c in axis_cols_c)
            f.write(f"| {i} |{vals}\n")
        f.write("\n_Report generated by `cli.py insights`_\n")

    print(f"[insights] Report -> {report_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "clean":
        return cmd_clean(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "dimensionality":
        return cmd_dimensionality(args)
    if args.cmd == "insights":
        return cmd_insights(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
