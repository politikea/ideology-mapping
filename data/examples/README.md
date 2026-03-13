# Example Dataset: `politikas_100.csv`

This file contains example political proposals (*politikas*) for use with the toolkit.
It is the recommended starting point for testing and demonstration.

## Status

This file contains 100 hand-crafted synthetic proposals covering all policy categories and a wide ideological range. Each proposal is manually crafted to:

- Cover all 12 policy categories used in Politikea Phase 1
- Span the full ideological range on multiple axes
- Include pairs that are similar in vocabulary but opposite in stance (to demonstrate the joint-filter finding from the validation analysis)
- Include pairs that are ideologically aligned but differently phrased

## Column Specification

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `item_id` | string | Yes | Unique stable identifier (e.g. `ex_001`). Used for joins, caching, and deduplication. |
| `text` | string | Yes | Proposal text. Spanish; 20–200 words recommended. Should be a complete, self-contained policy proposal. |
| `category` | string | Yes | One of the 12 policy categories below. Used for within-category validation and category centroid analysis. |
| `source` | string | Yes | `"synthetic"` for manually crafted examples, or public attribution string for real proposals. |

## Categories

The 12 policy categories recognized by the toolkit:

| Category (es) | Description |
|---------------|-------------|
| Economía e Impuestos | Fiscal policy, taxation, redistribution |
| Servicios Públicos | Healthcare, social services, public infrastructure |
| Trabajo y Mercado Laboral | Employment, labor rights, wages |
| Infraestructura y Vivienda | Housing policy, urban planning, transport |
| Instituciones y Gobierno | Electoral reform, governance, anti-corruption |
| Seguridad y Defensa | Police, military, public safety |
| Derechos y Libertades | Civil rights, personal freedoms, privacy |
| Ciencia, Energía y Medioambiente | Climate, energy, research and development |
| Cultura y Educación Cívica | Education, culture, civic participation |
| Identidad y Cohesión Social | Social cohesion, migration, national identity |
| Relaciones Internacionales | Foreign policy, international treaties |
| Otros | Proposals that do not fit the above categories |

## Usage

```bash
# Score these proposals with GPT-4o (or any compatible model)
# See prompts/label_8axis_v1.txt for the exact prompt

# Then run the toolkit pipeline:
python cli.py clean \
    --input data/examples/annotations.parquet \
    --output data/examples/labels_clean.parquet

python cli.py validate \
    --labels data/examples/labels_clean.parquet \
    --items data/examples/politikas_100.csv \
    --output-dir results/

python cli.py dimensionality \
    --labels data/examples/labels_clean.parquet \
    --output-dir results/ \
    --skip-predictive
```

See `notebooks/01_end_to_end_demo.ipynb` for a step-by-step walkthrough.
