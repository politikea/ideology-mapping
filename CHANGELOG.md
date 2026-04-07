# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-01

### Added
- `analysis/labeler.py` — score proposals via any OpenAI-compatible API with caching and retries
- `scripts/score_proposals.py` — batch scoring CLI script
- `cli.py triangulate` subcommand — cross-model label comparison with gate decision and report
- `analysis/visualize.py` — publication-quality visualization functions (heatmap, biplot, radar, distributions)
- `prompts/label_8axis_v1_schema.json` — JSON Schema for structured output enforcement
- `prompts/README.md` — usage guide for the labeling prompt
- `docs/results_summary.md` — key metrics and how to interpret them
- `docs/prompt-engineering.md` — prompt ablation lessons for open-weight model practitioners
- `data/examples/triangulation_demo/` — synthetic mock data for triangulation demonstrations
- `CHANGELOG.md`
- README badges (license, Python version, arXiv)
- Research section in README linking to the paper

### Fixed
- Prompt `{{JSON_SCHEMA}}` placeholder replaced with inline schema and example response
- README repository structure diagram corrected (cli.py at root, correct analysis/ listing)
- LICENSE section in README updated from "TBD" to Apache 2.0
- CONTRIBUTING.md license reference updated

### Changed
- README reorganized with quickstart, pipeline overview, and research context sections
- `pip install -e .` now documented as the primary install method

## [0.1.0] - 2026-03-13

### Added
- Initial public release
- 8-axis labeling prompt (`prompts/label_8axis_v1.txt`)
- Analysis pipeline: cleaning, validation, dimensionality, insights
- Cross-model triangulation module (`analysis/triangulation.py`)
- CLI with `clean`, `validate`, `dimensionality`, `insights` subcommands
- 100 synthetic example proposals (`data/examples/politikas_100.csv`)
- End-to-end demo notebook
- Documentation: axes, methodology
- Apache 2.0 license
