# Politikea Ideology Mapping Toolkit

AI tools for political sensemaking are being built right now, by governments, platforms, and civic organizations. Most of the methodology behind them is invisible. We think that needs to change, not because any single team has the right answer, but because the field moves faster and more responsibly when the work is in the open.

This toolkit is a **methodology replication kit**. It contains the labeling prompt, the analysis pipeline, and example data so you can run the same analysis on your own political proposals. You cannot replicate Politikea's exact numbers — those come from a private corpus of user-generated proposals on the Politikea platform. What you *can* do is apply the same methodology to your data and tell us what you find.

The more people stress-test this, the better it gets for everyone.

**Paper**: [link pending — arxiv cs.CY submission in progress]

---

## What This Is

Politikea is a civic platform that scores political proposals across 8 independent ideological axes using AI. Instead of a left-right slider, each proposal gets a continuous score from −100 to +100 on each axis. A user's political position emerges from the proposals they agree and disagree with.

This toolkit contains:

- **The labeling prompt** (`prompts/label_8axis_v1.txt`) — the exact prompt used to score proposals with GPT-4 class models. This is the operational definition of the 8 axes. Understanding any score requires reading this.
- **The analysis pipeline** (`analysis/`) — clean, validate, and analyze your scored proposals
- **Example proposals** (`data/examples/`) — 100 hand-crafted synthetic politikas covering all policy categories and a wide ideological range
- **End-to-end notebook** (`notebooks/01_end_to_end_demo.ipynb`) — walk through the full pipeline
- **Documentation** (`docs/`) — axis definitions, methodology, and result summaries

---

## The 8 Axes

| Axis | −100 pole | +100 pole |
|------|-----------|-----------|
| `aequitas_libertas` | Full redistribution, equity of outcome | Individual freedom, minimal redistribution |
| `imperium_anarkhia` | Centralized state authority | Decentralization, self-organization |
| `universalism_particularism` | Universal rules and rights for all | Group-specific or context-specific policies |
| `market_collective_allocation` | Market allocation of resources | Collective/democratic allocation |
| `inequality_acceptance_correction` | Inequality is acceptable and natural | Inequality is structural and must be corrected |
| `individual_socialized_risk` | Individuals absorb their own risk | Risk pooled collectively |
| `progressivism_preservation` | Active reform, embrace of change | Stability, tradition, continuity |
| `technocracy_populism` | Expert-led governance | Popular-will primacy |

Scores are continuous (float, −100 to +100). A score of 0 means balanced or ambiguous. See `prompts/label_8axis_v1.txt` for the full scoring guidance and `docs/axes.md` for the rationale behind each axis.

---

## Quick Start

### Prerequisites

```bash
pip install pandas pyarrow scikit-learn sentence-transformers scipy matplotlib seaborn
```

### 1. Score your proposals

Use the prompt in `prompts/label_8axis_v1.txt` with any GPT-4 class model. Call the model N=10+ times per proposal and save the raw JSON responses. The multi-run design is essential: it gives you ICC reliability estimates and sign agreement per axis.

```python
from openai import OpenAI

prompt = open("prompts/label_8axis_v1.txt").read()
# ... see notebooks/01_end_to_end_demo.ipynb for a full example
```

### 2. Clean and validate labels

```bash
python cli.py clean \
    --input path/to/annotations.parquet \
    --output path/to/labels_clean.parquet

python cli.py validate \
    --labels path/to/labels_clean.parquet \
    --items path/to/proposals.parquet \
    --output-dir results/

python cli.py dimensionality \
    --labels path/to/labels_clean.parquet \
    --output-dir results/ \
    --skip-predictive
```

### 3. Explore results

Open `notebooks/01_end_to_end_demo.ipynb` for a step-by-step walkthrough with the example dataset.

---

## Repository Structure

```
politikea-ideology-mapping/
├── README.md                        ← this file
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
│
├── data/
│   └── examples/
│       ├── README.md                ← column spec and category list
│       └── politikas_100.csv        ← example proposals
│
├── analysis/                        ← analysis modules
│   ├── cli.py                       ← clean, validate, dimensionality subcommands
│   └── analysis/
│       ├── cleaning.py
│       ├── similarity.py
│       ├── dimensionality.py
│       └── stats_utils.py
│
├── prompts/
│   └── label_8axis_v1.txt           ← the labeling prompt; this IS the axis definition
│
├── notebooks/
│   └── 01_end_to_end_demo.ipynb     ← full pipeline walkthrough
│
├── docs/
│   ├── axes.md                      ← axis documentation and design rationale
│   └── methodology.md               ← full methodology description
│
└── scripts/
    └── run_pipeline.sh              ← convenience shell script
```

---

## What the Methodology Shows (Phase 1 Summary)

These are Politikea's numbers from 2,977 Spanish political proposals. Your results will differ based on your data. This is intentional — the point is for you to run the analysis and see what you find.

| Metric | Politikea Phase 1 |
|--------|-------------------|
| Proposals scored | 2,977 |
| Valid items | 2,442 (82.0%) |
| ICC(2,1) range | 0.885 – 0.976 |
| 3 PCs explain | ≥ 80% of variance |
| Best 2-axis pair | aequitas + progressivism (R² = 0.626) |
| Politikea product pair (aequitas + imperium) | R² = 0.595 (rank 11 of 28) |
| Global text→8D ρ | 0.198 |
| Cross-model audit | Both Gemini 2.5 Flash and llama3.1:8b fail the gate — for structurally different reasons |

The cross-model audit result is the main methodological contribution: AI models carry implicit calibration biases that are consistent within a model and only detectable through cross-model comparison. See the paper for details.

---

## What We Are Not Claiming

- We are not claiming the 8 axes are the correct or complete political space
- We are not claiming GPT-5.2 labels are ground-truth validated (no human annotation layer in Phase 1)
- We are not claiming these results generalize beyond Spanish political discourse in this corpus

The axis set is a starting point. Challenge it with your data. We want to know what you find.

---

## Contributing

See `CONTRIBUTING.md`. The highest-value contributions are:

1. **Replications on different languages or political systems** — do the same axes make sense in other contexts?
2. **Alternative axis sets** — is there a better combination?
3. **Cross-model calibration** — can you close the gap between GPT and open-weight models?
4. **More example proposals** — contributions of additional synthetic proposals in other languages are welcome

---

## License

[License TBD — will be set before public release. Expected: Apache 2.0 or MIT for code, CC BY 4.0 for prompt and documentation.]

User data and proprietary platform components are not included and will not be released.

---

## Cite

```
Vergara et al. (2026). "Mapping Political Proposals in 8 Dimensions:
Reliability, Calibration Bias, and What AI Models Cannot Agree On."
Politikea Research Technical Report Phase 1. [arxiv link pending]
```
