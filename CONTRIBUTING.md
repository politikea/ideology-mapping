# Contributing

Thank you for taking the time to engage with this work. The whole point of releasing it is to find out what we got wrong.

---

## What We Most Want From Contributors

### 1. Replications on different languages or political systems

The methodology was developed on Spanish political proposals from a single civic platform. We do not know if:
- The 8 axes make sense in other political cultures
- The axis weights (and which axes are most independent) hold in other languages
- The ICC reliability holds with models fine-tuned for other languages

If you run this on your own data and get different findings, that is a contribution.

### 2. Alternative axis sets

The 8 axes in `prompts/label_8axis_v1.txt` are a design choice, not a universal truth. If you think a different axis set better captures political disagreement in your context, we want to know:
- Which axes you replaced and why
- Whether the replacement axes achieve comparable ICC reliability
- How the PCA structure changes

### 3. Cross-model calibration protocols

Both Gemini 2.5 Flash and llama3.1:8b fail the triangulation gate against GPT-5.2 (see the paper, Section 4.2). We believe:
- Gemini's failure (positivity bias) is correctable via per-axis offset normalization or few-shot polarity anchors
- llama's failure (task overload) is correctable via task decomposition (Variant F from the prompt ablation) or a larger model

If you build and validate a calibration protocol that closes the gap, that is a high-value contribution.

### 4. More example proposals

The `data/examples/politikas_100.csv` file contains 100 hand-crafted synthetic proposals across all 12 categories. Contributions of additional synthetic proposals in other languages or covering underrepresented categories are welcome.

---

## How to Contribute

1. Fork the repository
2. Create a branch: `git checkout -b your-contribution-name`
3. Make your changes
4. Open a pull request with a clear description of what you changed and why

For larger changes (new axis sets, language ports, calibration protocols), please open an issue first to discuss the approach.

---

## What We Are Not Looking For

- Changes to the core 8-axis prompt without empirical validation (ICC testing on ≥100 proposals)
- Raw user data from any platform
- Claims about individual users' political positions

---

## Code Standards

- Python 3.9+
- `ruff` for linting (`pip install ruff && ruff check .`)
- No private credentials, API keys, or platform-specific configuration in any file

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0 or MIT — to be finalized before public release).
