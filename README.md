# Politikea Ideology Mapping Toolkit

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![arXiv](https://img.shields.io/badge/arXiv-cs.CY-b31b1b.svg)](#cite)

AI tools for political sensemaking are being built right now, by governments, platforms, and civic organizations. Most of the methodology behind them is invisible. We think that needs to change, not because any single team has the right answer, but because the field moves faster and more responsibly when the work is in the open.

This toolkit is a **methodology replication kit**. It contains the labeling prompt, the analysis pipeline, and example data so you can run the same analysis on your own political proposals. You cannot replicate Politikea's exact numbers — those come from a private corpus of user-generated proposals on the Politikea platform. What you *can* do is apply the same methodology to your data and tell us what you find.

The more people stress-test this, the better it gets for everyone.

---

## Quick Start

```bash
git clone https://github.com/politikea/ideology-mapping.git
cd ideology-mapping
pip install -e .

# Optional: install the openai SDK if you want to score your own proposals
pip install -e ".[scoring]"

# Run the full pipeline on pre-scored example data
python cli.py clean --input data/examples/annotations_example.parquet --output output/labels_clean.parquet
python cli.py validate --labels output/labels_clean.parquet --items data/examples/politikas_100.csv --output-dir output/
python cli.py dimensionality --labels output/labels_clean.parquet --output-dir output/ --skip-predictive
python cli.py insights --labels output/labels_clean.parquet --items data/examples/politikas_100.csv --output-dir output/
```

Pre-computed outputs are in `data/examples/reference_outputs/` if you want to browse results without running the pipeline.

To score your own proposals, see `prompts/README.md` or use the batch scorer:

```bash
python scripts/score_proposals.py --input my_proposals.csv --output annotations.parquet
```

<details>
<summary><strong>Using conda?</strong></summary>

```bash
conda create -n ideology-mapping python=3.11 -y
conda activate ideology-mapping
pip install -e .

# For notebook exploration:
pip install -e ".[dev]"
python -m ipykernel install --user --name ideology-mapping
```

All dependencies are managed via `pyproject.toml`. The conda env just provides Python and isolation — `pip install -e .` handles the rest.

</details>

---

## Reading Order

| Goal | Start here |
|------|------------|
| Understand the research | [PAPER.pdf](PAPER.pdf) |
| See results visually | [notebooks/01_end_to_end_demo.ipynb](notebooks/01_end_to_end_demo.ipynb) |
| Replicate on your own data | [docs/replication-guide.md](docs/replication-guide.md) |
| Deep dive: axis definitions | [docs/axes.md](docs/axes.md) |
| Deep dive: methodology | [docs/methodology.md](docs/methodology.md) |

---

## What This Is

Politikea is a civic platform that maps users across two ideological axes (equity vs. freedom; authority vs. decentralisation) based on their votes on concrete policy proposals. This toolkit extends the product framework to **8 independent axes** as a research exercise — stress-testing whether two dimensions are sufficient, and what signal is lost or gained at different dimensionalities. Each proposal gets a continuous score from -100 to +100 on each axis.

This toolkit contains:

- **The labeling prompt** (`prompts/label_8axis_v1.txt`) — the exact prompt used to score proposals with GPT-4 class models. This is the operational definition of the 8 axes. Understanding any score requires reading this.
- **The JSON output schema** (`prompts/label_8axis_v1_schema.json`) — for API structured output enforcement
- **A scoring module** (`analysis/labeler.py`) — score proposals via any OpenAI-compatible API with caching and retries
- **The analysis pipeline** (`analysis/`) — clean, validate, and analyze your scored proposals
- **Cross-model triangulation** (`analysis/triangulation.py`, `cli.py triangulate`) — detect model-specific calibration biases
- **Example data with pre-computed outputs** (`data/examples/`) — 100 synthetic proposals, pre-scored annotations, and pipeline reports ready to explore without an API key
- **End-to-end notebook** (`notebooks/01_end_to_end_demo.ipynb`) — walk through the full pipeline
- **Documentation** (`docs/`) — axis definitions, methodology, [replication guide](docs/replication-guide.md), and [prompt engineering lessons](docs/prompt-engineering.md)

---

## The 8 Axes

| Axis | -100 pole | +100 pole |
|------|-----------|-----------|
| `aequitas_libertas` | Full redistribution, equity of outcome | Individual freedom, minimal redistribution |
| `imperium_anarkhia` | Centralized state authority | Decentralization, self-organization |
| `universalism_particularism` | Universal rules and rights for all | Group-specific or context-specific policies |
| `market_collective_allocation` | Market allocation of resources | Collective/democratic allocation |
| `inequality_acceptance_correction` | Inequality is acceptable and natural | Inequality is structural and must be corrected |
| `individual_socialized_risk` | Individuals absorb their own risk | Risk pooled collectively |
| `progressivism_preservation` | Active reform, embrace of change | Stability, tradition, continuity |
| `technocracy_populism` | Expert-led governance | Popular-will primacy |

Scores are continuous (float, -100 to +100). A score of 0 means balanced or ambiguous. See `prompts/label_8axis_v1.txt` for the full scoring guidance and `docs/axes.md` for the rationale behind each axis.

---

## Research

**Paper**: *"Mapping Political Proposals in 8 Dimensions: Reliability, Calibration Bias, and What AI Models Cannot Agree On"* — [PAPER.pdf](PAPER.pdf) (arXiv submission pending)

The paper presents the full methodology and findings from Politikea's Phase 1 study. It covers:

- **8-axis ideological framework**: design rationale, axis independence analysis, and how the axes were selected
- **Within-model reliability**: ICC(2,1) estimates from repeated scoring runs, confidence filtering, and validity gating
- **Cross-model triangulation**: what happens when you audit GPT labels with a second model (Gemini, llama) — and why single-model annotation is methodologically insufficient
- **Prompt engineering for open-weight models**: which prompt variants help and which hurt agreement with the reference model
- **Dimensionality analysis**: PCA structure, axis collinearity, and which 2-axis pairs best reconstruct the full 8D space
- **Semantic validation**: the relationship between text similarity and ideological similarity, and why each signal alone is unreliable

This toolkit enables you to replicate the methodology on your own data. The paper describes what was found; the toolkit lets you run the same analysis and compare.

---

## Pipeline Overview

### 1. Score proposals

Use the prompt in `prompts/label_8axis_v1.txt` with any frontier model (Claude, GPT-4o, or equivalent). Call the model N=10+ times per proposal at temperature=0.2. The multi-run design is essential: it gives you ICC reliability estimates and sign agreement per axis.

```python
import anthropic, json

client = anthropic.Anthropic()
prompt_template = open("prompts/label_8axis_v1.txt").read()
prompt = prompt_template.replace("{{TEXT}}", proposal_text)

responses = []
for _ in range(13):
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    responses.append(json.loads(resp.content[0].text))
```

See `prompts/README.md` for full details.

### 2. Clean and validate labels

```bash
python cli.py clean --input annotations.parquet --output labels_clean.parquet
python cli.py validate --labels labels_clean.parquet --items proposals.csv --output-dir results/
python cli.py dimensionality --labels labels_clean.parquet --output-dir results/ --skip-predictive
python cli.py insights --labels labels_clean.parquet --items proposals.csv --output-dir results/
```

### 3. Cross-model triangulation (optional)

```bash
python cli.py triangulate \
    --reference labels_clean.parquet \
    --audit audit_model_labels.parquet \
    --output-dir results/
```

### 4. Explore results

Open `notebooks/01_end_to_end_demo.ipynb` for a step-by-step walkthrough with the example dataset.

---

## Repository Structure

```
ideology-mapping/
├── README.md                        <- this file
├── PAPER.pdf                        <- the research paper (arXiv submission pending)
├── LICENSE                          <- Apache 2.0
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── cli.py                           <- CLI: clean, validate, dimensionality, insights, triangulate
│
├── analysis/                        <- analysis library
│   ├── cleaning.py                  <- confidence filtering, ICC, validity flags
│   ├── dimensionality.py            <- PCA, VIF, 2-axis reconstruction R²
│   ├── insights.py                  <- category centroids, correlation, clustering
│   ├── label_io.py                  <- axis definitions, parquet I/O
│   ├── labeler.py                   <- score proposals via any OpenAI-compatible API
│   ├── similarity.py                <- text<>8D Spearman, embedding projection, dedup
│   ├── stats_utils.py               <- bootstrap, permutation, ICC helpers
│   ├── triangulation.py             <- cross-model agreement and gating
│   ├── visualize.py                 <- publication-quality charts (heatmap, biplot, radar)
│   └── category_mapping.py          <- category ID -> name lookup
│
├── prompts/
│   ├── README.md                    <- usage guide: how to call the prompt
│   ├── label_8axis_v1.txt           <- the labeling prompt (this IS the axis definition)
│   └── label_8axis_v1_schema.json   <- JSON Schema for structured output enforcement
│
├── data/
│   └── examples/
│       ├── README.md                <- column spec and category list
│       ├── politikas_100.csv        <- 100 synthetic example proposals
│       ├── annotations_example.parquet <- pre-scored multi-run annotations (synthetic)
│       ├── labels_clean.parquet     <- cleaned/aggregated labels (from annotations_example)
│       ├── reference_outputs/       <- pre-computed pipeline reports
│       └── triangulation_demo/      <- mock audit labels for triangulation examples
│
├── notebooks/
│   └── 01_end_to_end_demo.ipynb     <- full pipeline walkthrough
│
├── docs/
│   ├── axes.md                      <- axis documentation and design rationale
│   ├── methodology.md               <- full methodology description
│   ├── results_summary.md           <- key metrics and how to interpret them
│   ├── prompt-engineering.md        <- prompt ablation lessons for practitioners
│   ├── findings.md                  <- paper findings summary with embedded figures
│   ├── calibration-protocol.md      <- cross-model calibration guide
│   ├── replication-guide.md         <- step-by-step from raw data to results
│   └── figures/                     <- key paper visualizations for GitHub browsing
│
└── scripts/
    ├── score_proposals.py           <- batch scoring CLI
    ├── generate_example_labels.py   <- regenerate synthetic example annotations + labels
    ├── generate_triangulation_mocks.py <- regenerate triangulation demo data
    └── run_pipeline.sh              <- convenience shell script
```

---

## What We Are Not Claiming

- We are not claiming the 8 axes are the correct or complete political space
- We are not claiming GPT labels are ground-truth validated (no human annotation layer in Phase 1)
- We are not claiming these results generalize beyond Spanish political discourse in the Phase 1 corpus

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

Apache License 2.0. See [LICENSE](LICENSE) for the full text.

User data and proprietary platform components are not included and will not be released.

---

## Cite

```
Vergara, J. (2026). "Mapping Political Proposals in 8 Dimensions:
Reliability, Calibration Bias, and What AI Models Cannot Agree On."
Politikea Research Technical Report Phase 1. See PAPER.pdf (arXiv submission pending).
```
