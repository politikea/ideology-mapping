"""
category_mapping.py — Shared category_id -> category name mapping helpers.

This keeps analysis-time category enrichment explicit and local so downstream
scripts do not depend on external folders.
"""
from __future__ import annotations


# Maps the 12 UUID-based category identifiers to their canonical English labels.
# A category column with plain string names (e.g. "Economy & Taxation") does not
# require this mapping — it is only used when items have a UUID category_id field.
CATEGORY_ID_TO_NAME = {
    "1cd52ba1-e2d4-4d51-9fc7-d22439a4cafd": "Infraestructura y Vivienda",
    "213d622e-12e2-48f1-a73b-a863708dc7bc": "Instituciones y Gobierno",
    "271543cf-20f1-41d6-9979-612f99d5b13e": "Ciencia, Energía y Medioambiente",
    "3549ba89-4da8-4c9a-8522-0874609a94bb": "Economía e Impuestos",
    "88e13d4f-4aa8-49d3-a459-fae4c794b2d9": "Cultura y Educación Cívica",
    "8b6d4e72-140a-4767-a12a-e1a33b16d4ae": "Derechos y Libertades",
    "8dc311b0-093b-4514-80ff-51ab541cbf66": "Relaciones Internacionales",
    "9e3daffd-5e5b-4d25-b93b-86c77e3a7c9b": "Trabajo y Mercado Laboral",
    "b7e31c46-2849-4fc1-b18c-19613ca09e0c": "Seguridad y Defensa",
    "cc8ac5f2-8f9c-497f-9902-2d8cd2d18798": "Identidad y Cohesión Social",
    "cd19c3ff-8b7b-484e-a504-9ac9e6d183d9": "Servicios Públicos",
    "f316e6fb-4e60-4afb-a37f-72cb255c484f": "Otros",
}

# Canonical analysis-time taxonomy (12 labels)
CATEGORY_LABELS = [
    "Instituciones y Gobierno",
    "Economía e Impuestos",
    "Servicios Públicos",
    "Infraestructura y Vivienda",
    "Ciencia, Energía y Medioambiente",
    "Cultura y Educación Cívica",
    "Derechos y Libertades",
    "Trabajo y Mercado Laboral",
    "Seguridad y Defensa",
    "Identidad y Cohesión Social",
    "Otros",
    "Relaciones Internacionales",
]


def map_category_id_to_name(category_id: object) -> str | None:
    """Map UUID-like category_id to a normalized category label."""
    if category_id is None:
        return None
    key = str(category_id).strip()
    if not key:
        return None
    return CATEGORY_ID_TO_NAME.get(key)
