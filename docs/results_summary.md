# Results Summary

This page describes the key metrics produced by each stage of the analysis pipeline and how to interpret them. For the full results from the Politikea Phase 1 study, see [PAPER.pdf](../PAPER.pdf) (arXiv submission pending).

---

## Cleaning (`cli.py clean`)

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| Valid item rate | Fraction of proposals passing all reliability gates | ≥ 70% suggests the prompt and model are working well |
| ICC(2,1) per axis | Intraclass correlation across repeated runs | ≥ 0.85 is excellent; < 0.70 warrants investigation |
| Sign agreement per axis | Fraction of runs agreeing on the score's polarity | ≥ 0.90 for well-defined proposals |
| Per-axis std | Standard deviation across runs for a single item | Lower is more reliable; threshold default is 30 |

## Validation (`cli.py validate`)

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| Global text→8D ρ | Spearman correlation between text similarity and 8D similarity | Positive but moderate — text and ideology are related but not the same |
| Within-category ρ | Same correlation restricted to same-category pairs | Lower than global — within a topic, ideological axes discriminate beyond vocabulary |
| Per-axis linguistic encoding (H3) | How well text embeddings predict each axis score | High ρ axes have vocabulary signal; low ρ axes are ideologically latent |
| Proposal deduplication (H4) | Clusters of proposals that are similar in both text AND 8D space | Joint filter eliminates both noise modes |

## Dimensionality (`cli.py dimensionality`)

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| PCA variance explained | How many principal components capture the variance | If 2–3 PCs explain > 80%, the axes have significant collinearity |
| VIF per axis | Variance Inflation Factor — collinearity measure | VIF > 10 = high redundancy with other axes; VIF < 2 = unique signal |
| 2-axis reconstruction R² | How well each pair of axes predicts all 8 | Higher R² = that pair captures more of the full ideological space |

## Insights (`cli.py insights`)

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| Category centroids | Mean 8D vector per policy category | Categories should have distinct profiles |
| Axis correlation matrix | Pearson correlations between axes | High correlations suggest redundant axes |
| K-means silhouette | Cluster quality in 8D space | Higher is better; typical range 0.1–0.4 for political data |

<img src="figures/category_radar.png" width="805" alt="Category centroid profiles">

*Category centroid profiles for the top policy categories. Each line connects a category's mean score across the 8 axes, revealing distinct ideological fingerprints per policy area.*

## Cross-Model Triangulation

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| Directional agreement | Fraction of items where primary and audit model agree on score polarity | ≥ 0.85 to pass the publication gate |
| Mean absolute error (MAE) | Average magnitude difference between primary and audit scores | ≤ 15 to pass the gate |
| Per-axis MAE | MAE broken down by axis | Identifies which axes are most sensitive to model choice |

---

## Interpreting Your Results

Your numbers will differ from the paper's Phase 1 results — that is expected and correct. The toolkit runs the same methodology on your data. Differences are informative: they tell you how your data, language, and model configuration interact with the 8-axis framework.

If your results diverge significantly from Phase 1 benchmarks, see `docs/methodology.md` for threshold sensitivity and `CONTRIBUTING.md` for how to share your findings.
