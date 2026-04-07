# Key Findings

This page summarizes the structure of the paper's findings for non-authors. For the full analysis, see [PAPER.pdf](../PAPER.pdf) (arXiv submission pending).

---

## What Was Built

An 8-axis ideological scoring framework for political proposals, applied via LLM structured output. Each proposal receives a continuous score (-100 to +100) on 8 independent axes. Multiple scoring runs per proposal enable reliability estimation via ICC(2,1).

The analysis pipeline includes:
- **Cleaning**: confidence filtering, multi-run aggregation, validity gating
- **Validation**: text-ideological similarity analysis, per-axis linguistic encoding, deduplication
- **Dimensionality**: PCA, axis collinearity (VIF), 2-axis reconstruction ranking
- **Triangulation**: cross-model calibration bias detection

## What Was Found

### Within-model reliability is high

Repeated scoring runs of the same proposal produce consistent results. ICC(2,1) values across all 8 axes are well above standard reliability thresholds. The majority of proposals pass the validity gate. See `docs/results_summary.md` for how to interpret these metrics on your own data.

### The 8 axes are not fully independent

PCA reveals that a small number of principal components capture most of the variance. Several axes, particularly the economic ones, are highly correlated. This does not mean the correlated axes are redundant — they capture distinct concepts that happen to co-occur in the data.

<img src="figures/axis_correlation.png" width="575" alt="Axis correlation matrix">

*Pearson correlations between the 8 axes. The economic cluster (aequitas, market, inequality, risk) shows high collinearity; the authority cluster (imperium, technocracy, universalism) carries independent signal.*

### Text similarity and ideological similarity are different signals

Proposals with similar wording can have opposite ideological scores, and proposals with similar scores can use very different language. Neither signal alone is reliable for identifying related proposals. Their intersection — pairs scoring high on both — is a reliable near-duplicate detector.

### Cross-model agreement fails in structurally different ways

When a second model (audit model) re-scores the same proposals, the results diverge beyond acceptable thresholds. Different models fail for different reasons:
- Some models show **acquiescence bias**: systematic positive offset across all axes
- Other models show **task overload**: coarse quantization, round-number scores, high instability

<img src="figures/triangulation_axis_means.png" width="633" alt="Gemini triangulation — acquiescence bias">

*Per-axis comparison: GPT (reference) vs. Gemini 2.5 Flash. The audit model's scores are systematically shifted positive across all axes — a correctable offset bias.*

<img src="figures/triangulation_axis_means_llama.png" width="633" alt="Llama triangulation — task overload">

*Per-axis comparison: GPT (reference) vs. llama3.1:8b. The audit model shows high noise and coarse quantization — a structurally different, less correctable failure mode.*

This is the paper's central methodological contribution: the triangulation gate doesn't just reject — it *diagnoses* the failure mode, telling practitioners whether calibration is worth attempting.

### Prompt engineering matters for open-weight models

Different prompt variants produce dramatically different reliability levels with smaller open-weight models. Language matching (translating the prompt to match the proposal language) is the most impactful single change. Few-shot examples, counterintuitively, can hurt agreement. See `docs/prompt-engineering.md` for details.

---

## What Remains Open

- Human annotation validation (no human labels in Phase 1)
- Generalizability beyond Spanish political discourse
- Whether the optimal axis set differs across political cultures
- Effective calibration protocols for closing the cross-model gap

These are active research directions. See `CONTRIBUTING.md` for how to contribute.
