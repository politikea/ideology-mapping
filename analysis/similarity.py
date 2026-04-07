"""
similarity.py — Semantic validity tools.

Validates that politikas with similar 8D label vectors are genuinely
similar in content, by comparing:
  - cosine similarity in 8D label space
  - cosine similarity of text embeddings (sentence-transformers)

Then reports Spearman correlation between the two.  An optional Ollama
spot-check provides a qualitative sanity layer on the top-K pairs.

All functions are pure (no file I/O, no CLI).
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats as scipy_stats

from .label_io import AXES

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cosine_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _pair_corr_with_ci_perm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    n_boot: int = 1000,
    n_perm: int = 1000,
) -> dict:
    rho, pval = scipy_stats.spearmanr(x, y)
    rng = np.random.default_rng(random_state)
    n = len(x)

    boot_rhos = []
    boot_iter = range(int(n_boot))
    if tqdm is not None and n_boot >= 200:
        boot_iter = tqdm(boot_iter, total=int(n_boot), desc="bootstrap_rho", leave=False)
    for _ in boot_iter:
        idx = rng.integers(0, n, size=n)
        r_b, _ = scipy_stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(float(r_b) if np.isfinite(r_b) else np.nan)
    rho_ci = (
        float(np.nanpercentile(boot_rhos, 2.5)),
        float(np.nanpercentile(boot_rhos, 97.5)),
    )

    greater = 0
    observed = abs(float(rho))
    perm_iter = range(int(n_perm))
    if tqdm is not None and n_perm >= 200:
        perm_iter = tqdm(perm_iter, total=int(n_perm), desc="perm_test", leave=False)
    for _ in perm_iter:
        y_perm = rng.permutation(y)
        r_p, _ = scipy_stats.spearmanr(x, y_perm)
        if abs(float(r_p)) >= observed:
            greater += 1
    perm_p = float((greater + 1) / (n_perm + 1))
    return {
        "spearman_r": float(rho),
        "p_value": float(pval),
        "permutation_p": perm_p,
        "rho_ci_low": rho_ci[0],
        "rho_ci_high": rho_ci[1],
    }


# ── 8D nearest neighbours ─────────────────────────────────────────────────────

def nearest_neighbors_8d(
    df: pd.DataFrame,
    k: int = 50,
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Find the top-K most similar pairs in 8D label space (cosine similarity).

    Args:
        df:        DataFrame with item_id + 8 axis columns (one row per item).
        k:         Number of nearest neighbours per item to retrieve.
        axis_cols: Which columns to use as the 8D vector (defaults to AXES).

    Returns:
        DataFrame with columns:
            item_id_a, item_id_b, cosine_similarity_8d
        Sorted descending by cosine_similarity_8d; duplicate pairs removed.
    """
    axis_cols = axis_cols or AXES
    df = df.dropna(subset=axis_cols).copy()

    X = df[axis_cols].astype(float).to_numpy()

    # Normalise rows for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = X / norms

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric="cosine", algorithm="brute")
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    item_ids = df["item_id"].to_numpy()
    pairs = []
    seen: set = set()
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists[1:], idxs[1:]):  # skip self (index 0)
            a, b = item_ids[i], item_ids[j]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "item_id_a": a,
                "item_id_b": b,
                "cosine_similarity_8d": float(1.0 - dist),
            })

    return (
        pd.DataFrame(pairs)
        .sort_values("cosine_similarity_8d", ascending=False)
        .reset_index(drop=True)
    )


def nearest_neighbors_text(
    items_df: pd.DataFrame,
    *,
    text_col: str = "text_norm",
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
    k: int = 50,
    mean_center: bool = False,
) -> pd.DataFrame:
    """
    Find top-K nearest neighbor pairs in text-embedding space.

    Args:
        mean_center: Subtract corpus mean before computing similarity.
                     Removes shared "Spanish political domain" component,
                     exposing within-domain variance.

    Returns columns: item_id_a, item_id_b, cosine_similarity_text
    """
    work = items_df[["item_id", text_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["item_id_a", "item_id_b", "cosine_similarity_text"])

    item_ids = work["item_id"].tolist()
    texts = work[text_col].astype(str).tolist()
    embeddings = embed_texts(texts, model_name=embedding_model, mean_center=mean_center)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = embeddings / norms
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(work)), metric="cosine", algorithm="brute")
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    pairs = []
    seen: set = set()
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists[1:], idxs[1:]):
            a, b = item_ids[i], item_ids[j]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "item_id_a": a,
                    "item_id_b": b,
                    "cosine_similarity_text": float(1.0 - dist),
                }
            )
    return (
        pd.DataFrame(pairs)
        .sort_values("cosine_similarity_text", ascending=False)
        .reset_index(drop=True)
    )


def summarize_texts_ollama(
    texts: list[str],
    model: str = "llama2:7b",
    timeout_s: int = 45,
    max_input_chars: int = 1200,
    cache_dir: "Path | None" = None,
    mode: str = "summary",
) -> tuple[list[str | None], dict]:
    """
    Produce compact policy text representations with local Ollama.

    Args:
        mode: "summary" (default) — one neutral sentence describing the proposal.
              "stance" — structured stance descriptor: "A favor de X, en contra de Y,
              posición: [positiva/negativa/neutral] respecto a [eje]". Designed to
              encode political stance explicitly for embedding, increasing ideological
              signal over topic signal.

    If Ollama refuses to summarize an item, returns None for that item (not raw text).
    Callers should filter None entries before embedding.
    """
    try:
        import requests
    except ImportError:
        warnings.warn("requests not available; using raw text instead of Ollama summaries.")
        out = [str(t)[:max_input_chars] for t in texts]
        return out, {
            "n_total": len(out),
            "n_cache_hit": 0,
            "n_generated": 0,
            "n_fallback_raw": len(out),
            "n_non_cache_written": 0,
            "prompt_version": "summary_es_v3",
            "model": model,
        }

    conn: "sqlite3.Connection | None" = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "ollama_summaries.sqlite3"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_cache (
                cache_key TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                max_input_chars INTEGER NOT NULL,
                raw_hash TEXT NOT NULL,
                generated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()

    prompt_version = "summary_es_v4" if mode == "summary" else "stance_es_v1"

    _REFUSAL_PREFIXES = (
        "no puedo", "lo siento", "i'm sorry", "i cannot", "as an ai",
        "como modelo", "como ia", "not able to", "cannot provide",
        "no es posible", "no me es posible", "no puedo cumplir",
    )

    def _is_refusal(text: str) -> bool:
        return text.lower().strip().startswith(_REFUSAL_PREFIXES)

    def _is_valid_summary(text: str, max_words: int = 24) -> bool:
        t = text.strip()
        if not t:
            return False
        if "\n" in t:
            return False
        # Ask model for one concise sentence, but validate deterministically here.
        n_words = len(t.split())
        return 4 <= n_words <= max_words

    n_cache_hit = 0
    n_generated = 0
    n_fallback_raw = 0
    n_non_cache_written = 0
    n_skipped_refusal = 0
    out: list[str | None] = []
    text_iter = texts
    if tqdm is not None and len(texts) >= 50:
        text_iter = tqdm(texts, total=len(texts), desc="ollama_summaries", leave=False)
    for t in text_iter:
        raw = str(t)[:max_input_chars]
        cache_key = hashlib.sha256(
            f"{prompt_version}::{model}::{max_input_chars}::{raw}".encode("utf-8")
        ).hexdigest()
        if conn is not None:
            row = conn.execute(
                "SELECT summary FROM summary_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                cached_summary = str(row[0])
                if _is_refusal(cached_summary):
                    out.append(None)
                    n_skipped_refusal += 1
                    n_cache_hit += 1
                else:
                    out.append(cached_summary)
                    n_cache_hit += 1
                continue

        if mode == "stance":
            prompt = (
                "Analiza la siguiente propuesta política y describe su postura ideológica "
                "en UNA frase en ESPAÑOL (máximo 28 palabras). "
                "Indica: qué defiende, qué rechaza, y si es favorable o contraria a la intervención del Estado. "
                "No opines. Sin comillas.\n\n"
                f"Propuesta:\n{raw}"
            )
        else:
            prompt = (
                "Resume la propuesta politica en UNA frase neutral en ESPAÑOL (maximo 24 palabras). "
                "Debe indicar accion de politica publica y objetivo. No opines, no uses comillas.\n\n"
                f"Propuesta:\n{raw}"
            )
        try:
            final_summary: str | None = None
            source = "skipped_refusal"
            prompt_current = prompt
            for attempt in range(2):
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt_current, "stream": False},
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                candidate = str(resp.json().get("response", "")).strip().strip('"').strip("'")
                if _is_refusal(candidate):
                    # Model refuses to summarize — skip item, no fallback
                    break
                if _is_valid_summary(candidate, max_words=24):
                    final_summary = candidate
                    source = "ollama"
                    break
                # Second attempt: corrective prompt
                if mode == "stance":
                    prompt_current = (
                        "Tu respuesta anterior no cumplió el formato.\n"
                        "Responde SOLO una frase en ESPAÑOL, máximo 28 palabras, sin saltos de línea, sin comillas. "
                        "Describe la postura: qué defiende, qué rechaza.\n\n"
                        f"Propuesta:\n{raw}"
                    )
                else:
                    prompt_current = (
                        "Tu respuesta anterior no cumplio formato.\n"
                        "Responde SOLO una frase en ESPAÑOL, neutral, maximo 24 palabras, "
                        "sin saltos de linea y sin comillas.\n\n"
                        f"Propuesta:\n{raw}"
                    )

            if source == "skipped_refusal":
                n_skipped_refusal += 1
            elif source != "ollama":
                # Format check failed both attempts — fall back to raw
                final_summary = raw
                source = "fallback_raw"
                n_fallback_raw += 1
            n_generated += 1
            out.append(final_summary)
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO summary_cache(
                        cache_key, summary, source, model, prompt_version,
                        max_input_chars, raw_hash, generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        final_summary if final_summary is not None else source,
                        source,
                        model,
                        prompt_version,
                        int(max_input_chars),
                        hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        int(time.time()),
                    ),
                )
                conn.commit()
                n_non_cache_written += 1
        except Exception:
            out.append(None)
            n_skipped_refusal += 1
            n_generated += 1
    if conn is not None:
        conn.close()
    return out, {
        "n_total": len(texts),
        "n_cache_hit": n_cache_hit,
        "n_generated": n_generated,
        "n_fallback_raw": n_fallback_raw,
        "n_skipped_refusal": n_skipped_refusal,
        "n_non_cache_written": n_non_cache_written,
        "prompt_version": prompt_version,
        "model": model,
        "cache_backend": "sqlite",
    }


def attach_8d_similarity(
    pairs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    axis_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Attach cosine_similarity_8d to pair rows using label vectors."""
    axis_cols = axis_cols or AXES
    cols = ["item_id"] + axis_cols
    work = labels_df[cols].dropna(subset=axis_cols).copy()
    lookup = {
        row["item_id"]: row[axis_cols].astype(float).to_numpy()
        for _, row in work.iterrows()
    }
    rows = []
    for _, row in pairs_df.iterrows():
        a, b = row["item_id_a"], row["item_id_b"]
        if a not in lookup or b not in lookup:
            continue
        rec = dict(row)
        rec["cosine_similarity_8d"] = _cosine_from_vectors(lookup[a], lookup[b])
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Text embeddings ───────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 64,
    mean_center: bool = False,
) -> np.ndarray:
    """
    Encode a list of strings using sentence-transformers.

    Args:
        texts:       List of politika text strings.
        model_name:  Hugging Face sentence-transformer model identifier.
        batch_size:  Encoding batch size (tune for available memory).
        mean_center: If True, subtract the corpus mean vector before returning.
                     This removes the shared "Spanish political text" component,
                     leaving only variance across proposals. Equivalent to
                     projecting onto the subspace orthogonal to the corpus mean.
                     Recommended when comparing within a single domain.

    Returns:
        Float32 array of shape (len(texts), embedding_dim).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for text embedding validation.\n"
            "Install with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if mean_center:
        corpus_mean = embeddings.mean(axis=0, keepdims=True)
        embeddings = embeddings - corpus_mean
        # Re-normalize so cosine similarity remains in [-1, 1]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)

    return embeddings


def cosine_similarity_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise cosine similarity.

    If b is None, computes the self-similarity matrix of a.
    Returns an (N, M) matrix.
    """
    if b is None:
        b = a
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


# ── Spearman correlation: 8D similarity vs text similarity ───────────────────

def spearman_8d_vs_text(
    pairs_df: pd.DataFrame,
    items_df: pd.DataFrame,
    text_col: str = "text_norm",
    axis_cols: Optional[list[str]] = None,
    embedding_models: Optional[list[str]] = None,
    sample_n: Optional[int] = None,
    random_state: int = 42,
    n_boot: int = 1000,
    n_perm: int = 1000,
    pass_rho_threshold: float = 0.5,
    pass_p_threshold: float = 0.05,
    marginal_rho_threshold: float = 0.3,
    marginal_p_threshold: float = 0.10,
) -> dict:
    """
    Compute Spearman correlation between 8D label similarity and
    text embedding similarity for a set of item pairs.

    Args:
        pairs_df:         Output of nearest_neighbors_8d().
        items_df:         Clean items DataFrame with item_id + text_col.
        text_col:         Column in items_df containing the text.
        axis_cols:        Axis columns to use for 8D vector.
        embedding_models: Sentence-transformer models for robustness checks.
        sample_n:         If set, randomly sample this many pairs to embed.
        random_state:     Seed for sampling.

    Returns:
        Dict with keys: spearman_r, p_value, n_pairs, verdict, rho_ci, permutation_p
    """
    axis_cols = axis_cols or AXES

    if sample_n is not None and len(pairs_df) > sample_n:
        pairs_df = pairs_df.sample(n=sample_n, random_state=random_state)

    # Build lookup from item_id → text
    text_lookup = items_df.set_index("item_id")[text_col].to_dict()

    valid_pairs = []
    for _, row in pairs_df.iterrows():
        a, b = row["item_id_a"], row["item_id_b"]
        if a in text_lookup and b in text_lookup:
            valid_pairs.append({
                "item_id_a": a,
                "item_id_b": b,
                "cosine_similarity_8d": row["cosine_similarity_8d"],
                "text_a": text_lookup[a],
                "text_b": text_lookup[b],
            })

    if len(valid_pairs) < 5:
        return {"spearman_r": np.nan, "p_value": np.nan, "n_pairs": 0, "verdict": "INSUFFICIENT_DATA"}

    valid_df = pd.DataFrame(valid_pairs)

    if not embedding_models:
        embedding_models = ["paraphrase-multilingual-mpnet-base-v2"]

    # Embed all unique texts efficiently
    unique_items = list(set(valid_df["item_id_a"].tolist() + valid_df["item_id_b"].tolist()))
    unique_texts = [text_lookup[i] for i in unique_items]
    model_results: list[dict] = []
    valid_df["cosine_similarity_text"] = np.nan

    for model_name in embedding_models:
        print(f"[similarity] Embedding {len(unique_texts)} unique texts with {model_name}...")
        embeddings = embed_texts(unique_texts, model_name=model_name)
        embed_lookup = dict(zip(unique_items, embeddings))

        # Compute text cosine similarity for each pair
        text_sims = []
        for _, row in valid_df.iterrows():
            ea = embed_lookup[row["item_id_a"]]
            eb = embed_lookup[row["item_id_b"]]
            cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            text_sims.append(cos)
        tmp_df = valid_df.copy()
        tmp_df["cosine_similarity_text"] = text_sims

        x = tmp_df["cosine_similarity_8d"].to_numpy(dtype=float)
        y = tmp_df["cosine_similarity_text"].to_numpy(dtype=float)
        stats = _pair_corr_with_ci_perm(
            x,
            y,
            random_state=random_state,
            n_boot=n_boot,
            n_perm=n_perm,
        )

        model_results.append(
            {
                "embedding_model": model_name,
                "spearman_r": stats["spearman_r"],
                "p_value": stats["p_value"],
                "permutation_p": stats["permutation_p"],
                "rho_ci_low": stats["rho_ci_low"],
                "rho_ci_high": stats["rho_ci_high"],
                "pairs_df": tmp_df,
            }
        )

    # Use first model as canonical table for reporting and evaluate robustness
    primary = model_results[0]
    valid_df = primary["pairs_df"]
    rho = primary["spearman_r"]
    pval = primary["p_value"]
    perm_p = primary["permutation_p"]
    rho_ci = (primary["rho_ci_low"], primary["rho_ci_high"])

    robustness = {
        "n_models": len(model_results),
        "model_results": model_results,
        "rho_range": [
            float(np.nanmin([m["spearman_r"] for m in model_results])),
            float(np.nanmax([m["spearman_r"] for m in model_results])),
        ],
    }

    # Verdict thresholds
    if pval < pass_p_threshold and perm_p < pass_p_threshold and rho > pass_rho_threshold:
        verdict = "PASS"
    elif pval < marginal_p_threshold and rho > marginal_rho_threshold:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        "spearman_r": float(rho),
        "p_value": float(pval),
        "permutation_p": float(perm_p),
        "rho_ci_low": float(rho_ci[0]),
        "rho_ci_high": float(rho_ci[1]),
        "n_pairs": int(len(valid_df)),
        "verdict": verdict,
        "pairs_df": valid_df,  # included for report generation
        "robustness": robustness,
    }


def within_category_spearman(
    labels_df: pd.DataFrame,
    items_df: pd.DataFrame,
    *,
    text_col: str = "text_norm",
    axis_cols: Optional[list[str]] = None,
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
    category_col: str = "category",
    nn_k: int = 30,
    min_category_size: int = 20,
    n_boot: int = 500,
    n_perm: int = 500,
    random_state: int = 42,
    mean_center: bool = False,
) -> dict:
    """
    Run the text→8D nearest-neighbor correlation separately within each category,
    then aggregate (median and per-category breakdown).

    Motivation: cross-category nearest neighbors conflate topic similarity with
    ideological similarity (a housing proposal and a housing proposal are closer
    than a housing vs. immigration proposal regardless of ideology). Restricting
    to within-category pairs removes this topic-confound and tests whether text
    similarity is informative *within the same policy domain*.

    Returns:
        {
          "median_rho": float,
          "categories": [{"category": str, "rho": float, "p": float, "n_pairs": int}, ...],
          "pooled_rho": float,  # Spearman on all within-category pairs concatenated
          "pooled_p": float,
          "n_pairs_total": int,
        }
    """
    axis_cols = axis_cols or AXES

    # Merge category column into labels
    cat_source = None
    if category_col in items_df.columns:
        cat_source = items_df[["item_id", category_col]].copy()
    elif "category" in items_df.columns and category_col != "category":
        cat_source = items_df[["item_id", "category"]].rename(columns={"category": category_col})
    elif "category_id" in items_df.columns:
        from analysis.category_mapping import map_category_id_to_name
        tmp = items_df[["item_id", "category_id"]].copy()
        tmp[category_col] = tmp["category_id"].apply(map_category_id_to_name)
        cat_source = tmp[["item_id", category_col]]

    if cat_source is None:
        print("[similarity] within_category_spearman: no category column found, skipping")
        return {}

    merged = items_df.merge(cat_source, on="item_id", how="left") if category_col not in items_df.columns else items_df.copy()
    merged[category_col] = merged[category_col].fillna("uncategorized")

    categories = [c for c, g in merged.groupby(category_col) if len(g) >= min_category_size]
    if not categories:
        print(f"[similarity] No categories with >= {min_category_size} items")
        return {}

    print(f"[similarity] Within-category analysis: {len(categories)} categories "
          f"(min size {min_category_size})")

    all_x, all_y = [], []
    per_cat = []

    for cat in sorted(categories):
        cat_items = merged[merged[category_col] == cat].copy()
        cat_labels = labels_df[labels_df["item_id"].isin(cat_items["item_id"])].copy()
        if len(cat_items) < min_category_size or len(cat_labels) < min_category_size:
            continue

        pairs = nearest_neighbors_text(
            cat_items,
            text_col=text_col,
            embedding_model=embedding_model,
            k=min(nn_k, len(cat_items) - 1),
            mean_center=mean_center,
        )
        if pairs.empty:
            continue
        pairs = attach_8d_similarity(pairs, cat_labels, axis_cols=axis_cols)
        x = pairs["cosine_similarity_text"].to_numpy(dtype=float)
        y = pairs["cosine_similarity_8d"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            continue

        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        per_cat.append({"category": cat, "rho": float(rho), "p": float(p), "n_pairs": len(x)})
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    if not per_cat:
        return {}

    all_x_arr = np.array(all_x)
    all_y_arr = np.array(all_y)
    from scipy.stats import spearmanr
    pooled_rho, pooled_p = spearmanr(all_x_arr, all_y_arr)

    per_cat_sorted = sorted(per_cat, key=lambda r: -r["rho"])
    return {
        "median_rho": float(np.median([r["rho"] for r in per_cat])),
        "pooled_rho": float(pooled_rho),
        "pooled_p": float(pooled_p),
        "n_pairs_total": len(all_x),
        "categories": per_cat_sorted,
    }


def spearman_text_vs_8d(
    labels_df: pd.DataFrame,
    items_df: pd.DataFrame,
    *,
    text_col: str = "text_norm",
    axis_cols: Optional[list[str]] = None,
    embedding_models: Optional[list[str]] = None,
    nn_k: int = 50,
    sample_n: Optional[int] = None,
    random_state: int = 42,
    n_boot: int = 1000,
    n_perm: int = 1000,
    use_ollama_summaries: bool = False,
    summary_model: str = "llama2:7b",
    summary_mode: str = "summary",
    summary_cache_dir: "Path | None" = None,
    mean_center_embeddings: bool = False,
    pass_rho_threshold: float = 0.5,
    pass_p_threshold: float = 0.05,
    marginal_rho_threshold: float = 0.3,
    marginal_p_threshold: float = 0.10,
    local_alignment_threshold: float = 0.70,
    local_quantile: float = 0.90,
) -> dict:
    """
    Reverse-direction validation:
    do text-nearest politikas also have close 8D ideological vectors?
    """
    axis_cols = axis_cols or AXES
    if not embedding_models:
        embedding_models = ["paraphrase-multilingual-mpnet-base-v2"]

    items_work = items_df.copy()
    repr_col = text_col
    if use_ollama_summaries:
        print(f"[similarity] Generating Ollama {summary_mode}s with model={summary_model}...")
        base_texts = items_work[text_col].fillna("").astype(str).tolist()
        summaries_raw, summary_stats = summarize_texts_ollama(
            base_texts,
            model=summary_model,
            cache_dir=summary_cache_dir,
            mode=summary_mode,
        )
        items_work["_summary_text"] = summaries_raw
        # Drop items where llama refused to summarize — None means skipped
        n_before = len(items_work)
        items_work = items_work[items_work["_summary_text"].notna()].copy()
        n_dropped = n_before - len(items_work)
        if n_dropped:
            print(f"[similarity] Dropped {n_dropped} items with refused/None summaries "
                  f"({n_dropped/n_before:.1%})")
        repr_col = "_summary_text"
    else:
        summary_stats = None

    if mean_center_embeddings:
        print("[similarity] Mean-centering embeddings (removes shared domain component)")

    model_results = []
    for model_name in embedding_models:
        pairs_df = nearest_neighbors_text(
            items_work,
            text_col=repr_col,
            embedding_model=model_name,
            k=nn_k,
            mean_center=mean_center_embeddings,
        )
        text_lookup = items_work.set_index("item_id")[repr_col].to_dict()
        if not pairs_df.empty:
            pairs_df["text_a"] = pairs_df["item_id_a"].map(text_lookup)
            pairs_df["text_b"] = pairs_df["item_id_b"].map(text_lookup)
        # Attach 8D similarity to ALL pairs before sampling (needed for H4 clusters)
        pairs_df = attach_8d_similarity(pairs_df, labels_df, axis_cols=axis_cols)
        full_pairs_df = pairs_df.copy()  # preserve full set for deduplication analysis
        if sample_n is not None and len(pairs_df) > sample_n:
            pairs_df = pairs_df.sample(n=sample_n, random_state=random_state)
        if len(pairs_df) < 5:
            continue
        x = pairs_df["cosine_similarity_text"].to_numpy(dtype=float)
        y = pairs_df["cosine_similarity_8d"].to_numpy(dtype=float)
        stats = _pair_corr_with_ci_perm(
            x,
            y,
            random_state=random_state,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        model_results.append(
            {
                "embedding_model": model_name,
                **stats,
                "pairs_df": pairs_df,
                "full_pairs_df": full_pairs_df,
            }
        )

    if not model_results:
        return {"spearman_r": np.nan, "p_value": np.nan, "n_pairs": 0, "verdict": "INSUFFICIENT_DATA"}

    primary = model_results[0]
    rho = float(primary["spearman_r"])
    pval = float(primary["p_value"])
    perm_p = float(primary["permutation_p"])
    pairs_primary = primary["pairs_df"]

    # Local-alignment metric: among top text-sim pairs, how many are also high in 8D?
    q = float(np.nanquantile(pairs_primary["cosine_similarity_text"].to_numpy(dtype=float), local_quantile))
    top_text = pairs_primary[pairs_primary["cosine_similarity_text"] >= q].copy()
    if len(top_text) == 0:
        local_alignment_rate = float("nan")
    else:
        local_alignment_rate = float(
            np.mean(top_text["cosine_similarity_8d"].astype(float).to_numpy() >= local_alignment_threshold)
        )

    if pval < pass_p_threshold and perm_p < pass_p_threshold and rho > pass_rho_threshold:
        verdict = "PASS"
    elif pval < marginal_p_threshold and rho > marginal_rho_threshold:
        verdict = "MARGINAL"
    elif np.isfinite(local_alignment_rate) and local_alignment_rate >= 0.60:
        verdict = "MIXED_LOCAL_ALIGNMENT"
    else:
        verdict = "FAIL"

    return {
        "spearman_r": rho,
        "p_value": pval,
        "permutation_p": perm_p,
        "rho_ci_low": float(primary["rho_ci_low"]),
        "rho_ci_high": float(primary["rho_ci_high"]),
        "n_pairs": int(len(pairs_primary)),
        "verdict": verdict,
        "pairs_df": pairs_primary,
        "full_pairs_df": primary.get("full_pairs_df", pairs_primary),
        "local_alignment_rate": local_alignment_rate,
        "local_alignment_threshold": local_alignment_threshold,
        "local_quantile": local_quantile,
        "robustness": {
            "n_models": len(model_results),
            "model_results": model_results,
            "rho_range": [
                float(np.nanmin([m["spearman_r"] for m in model_results])),
                float(np.nanmax([m["spearman_r"] for m in model_results])),
            ],
        },
        "text_representation": "ollama_summary" if use_ollama_summaries else "raw_text",
        "summary_model": summary_model if use_ollama_summaries else None,
        "summary_stats": summary_stats,
    }


# ── Ollama spot-check ─────────────────────────────────────────────────────────

def ollama_spot_check(
    pairs_df: pd.DataFrame,
    items_df: pd.DataFrame,
    text_col: str = "text_norm",
    model: str = "llama3",
    top_n: int = 10,
    timeout_s: int = 30,
) -> pd.DataFrame:
    """
    Qualitative spot-check: ask a local Ollama model whether the top-N
    most similar pairs are ideologically similar.

    This is NOT a statistical test — use it as a sanity check only.
    Gracefully returns an empty DataFrame if Ollama is unavailable.

    Args:
        pairs_df:  Output of nearest_neighbors_8d() (sorted desc by sim).
        items_df:  Clean items DataFrame.
        text_col:  Text column in items_df.
        model:     Ollama model name (must be pulled locally).
        top_n:     Number of top pairs to check.
        timeout_s: HTTP timeout in seconds.

    Returns:
        DataFrame with columns: item_id_a, item_id_b, cosine_similarity_8d,
        text_a, text_b, ollama_verdict, ollama_reasoning
    """
    try:
        import requests
    except ImportError:
        warnings.warn("requests not available; skipping Ollama spot-check.")
        return pd.DataFrame()

    top_pairs = pairs_df.head(top_n).copy()
    text_lookup = items_df.set_index("item_id")[text_col].to_dict()

    results = []
    for _, row in top_pairs.iterrows():
        a, b = row["item_id_a"], row["item_id_b"]
        text_a = text_lookup.get(a, "")
        text_b = text_lookup.get(b, "")

        prompt = (
            "You are a political analyst. Read these two political proposals "
            "and answer: are they ideologically similar?\n\n"
            f"Proposal A: {text_a[:500]}\n\n"
            f"Proposal B: {text_b[:500]}\n\n"
            "Answer with YES or NO, then one sentence of reasoning."
        )

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            output = resp.json().get("response", "").strip()
            verdict = "YES" if output.upper().startswith("YES") else "NO"
        except Exception as e:
            warnings.warn(f"Ollama request failed: {e}")
            output = ""
            verdict = "UNAVAILABLE"

        results.append({
            "item_id_a": a,
            "item_id_b": b,
            "cosine_similarity_8d": row["cosine_similarity_8d"],
            "text_a": text_a[:200],
            "text_b": text_b[:200],
            "ollama_verdict": verdict,
            "ollama_reasoning": output[:300],
        })

    return pd.DataFrame(results)


# ── H2: Lexical axis anchors ──────────────────────────────────────────────────

def lexical_axis_anchors(
    items_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    text_col: str = "text",
    axis_cols: Optional[list[str]] = None,
    n_top: int = 12,
    quantile: float = 0.25,
) -> dict[str, dict]:
    """
    For each axis, identify the words most associated with high (+) and low (-)
    scoring items using TF-IDF difference between the top and bottom quantile.

    Returns a dict: axis -> {"pos_words": [...], "neg_words": [...], "n_pos": int, "n_neg": int}

    This answers: "does language carry stance signal for this axis?"
    High discrimination = language encodes ideology on that axis.
    Low discrimination = axis captures latent dimension not expressed in vocabulary.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {}

    axis_cols = axis_cols or AXES
    mean_cols = {a: f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns}
    if not mean_cols:
        return {}

    text_lookup = items_df.set_index("item_id")[text_col].to_dict()
    results: dict[str, dict] = {}

    for axis, mean_col in mean_cols.items():
        merged = labels_df[["item_id", mean_col]].dropna(subset=[mean_col]).copy()
        merged["text"] = merged["item_id"].map(text_lookup)
        merged = merged.dropna(subset=["text"])
        if len(merged) < 20:
            continue

        scores = merged[mean_col].astype(float)
        low_thresh  = scores.quantile(quantile)
        high_thresh = scores.quantile(1 - quantile)

        pos_texts = merged[scores >= high_thresh]["text"].tolist()
        neg_texts = merged[scores <= low_thresh]["text"].tolist()

        if len(pos_texts) < 5 or len(neg_texts) < 5:
            continue

        all_texts = pos_texts + neg_texts
        labels_bin = [1] * len(pos_texts) + [0] * len(neg_texts)

        try:
            vec = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            X = vec.fit_transform(all_texts)
            feature_names = vec.get_feature_names_out()

            # mean TF-IDF per class
            X_arr = X.toarray()
            pos_mask = np.array(labels_bin) == 1
            mean_pos = X_arr[pos_mask].mean(axis=0)
            mean_neg = X_arr[~pos_mask].mean(axis=0)
            diff = mean_pos - mean_neg

            top_pos_idx = np.argsort(diff)[-n_top:][::-1]
            top_neg_idx = np.argsort(diff)[:n_top]

            results[axis] = {
                "pos_words": [feature_names[i] for i in top_pos_idx],
                "neg_words": [feature_names[i] for i in top_neg_idx],
                "n_pos": len(pos_texts),
                "n_neg": len(neg_texts),
                "mean_diff_top": float(diff[top_pos_idx[0]]) if len(top_pos_idx) else 0.0,
            }
        except Exception:
            continue

    return results


# ── H3: Embedding axis projection ────────────────────────────────────────────

def embedding_axis_projection(
    items_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    text_col: str = "text",
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
    axis_cols: Optional[list[str]] = None,
    quantile: float = 0.25,
    n_boot: int = 500,
) -> dict[str, dict]:
    """
    For each ideological axis, compute a "linguistic direction vector" as the
    difference between mean embeddings of high-scoring and low-scoring items.
    Then project all items onto this direction and compute Spearman rho vs
    the actual 8D axis score.

    This answers: "does the embedding space have a direction that encodes
    ideological axis X?" — i.e. does vocabulary geometry mirror ideology geometry?

    A high rho (>0.3) means language geometrically encodes that axis.
    A low rho means the axis captures something beyond surface vocabulary.

    Returns: axis -> {
        "rho": float, "p": float,
        "rho_ci_low": float, "rho_ci_high": float,
        "n_items": int,
        "interpretation": str
    }
    """
    try:
        from sentence_transformers import SentenceTransformer
        from scipy.stats import spearmanr
    except ImportError:
        return {}

    axis_cols = axis_cols or AXES
    mean_cols = {a: f"{a}_mean" for a in axis_cols if f"{a}_mean" in labels_df.columns}
    if not mean_cols:
        return {}

    # Build item text + score lookup
    text_lookup = items_df.set_index("item_id")[text_col].to_dict()
    merged_base = labels_df[["item_id"] + list(mean_cols.values())].copy()
    merged_base["text"] = merged_base["item_id"].map(text_lookup)
    merged_base = merged_base.dropna(subset=["text"])

    if len(merged_base) < 30:
        return {}

    print(f"[similarity] Computing axis-direction projections for {len(mean_cols)} axes "
          f"using {embedding_model}...")

    # Embed all items once
    all_ids   = merged_base["item_id"].tolist()
    all_texts = merged_base["text"].astype(str).tolist()

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        all_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embed_lookup = dict(zip(all_ids, embeddings))

    results: dict[str, dict] = {}

    for axis, mean_col in mean_cols.items():
        col_data = merged_base[["item_id", mean_col]].dropna(subset=[mean_col]).copy()
        if len(col_data) < 20:
            continue

        scores = col_data[mean_col].astype(float)
        low_thresh  = scores.quantile(quantile)
        high_thresh = scores.quantile(1 - quantile)

        pos_ids = col_data[scores >= high_thresh]["item_id"].tolist()
        neg_ids = col_data[scores <= low_thresh]["item_id"].tolist()

        if len(pos_ids) < 5 or len(neg_ids) < 5:
            continue

        # Direction vector: mean(pos embeddings) - mean(neg embeddings)
        pos_embs = np.stack([embed_lookup[i] for i in pos_ids if i in embed_lookup])
        neg_embs = np.stack([embed_lookup[i] for i in neg_ids if i in embed_lookup])
        direction = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction /= norm

        # Project all items onto this direction
        all_embs   = np.stack([embed_lookup[i] for i in col_data["item_id"] if i in embed_lookup])
        valid_ids  = [i for i in col_data["item_id"] if i in embed_lookup]
        projections = all_embs @ direction  # scalar per item

        score_map = col_data.set_index("item_id")[mean_col].to_dict()
        axis_scores = np.array([score_map[i] for i in valid_ids])

        rho, p = spearmanr(projections, axis_scores)

        # Bootstrap CI for rho
        n = len(projections)
        boot_rhos = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            r, _ = spearmanr(projections[idx], axis_scores[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        ci_low  = float(np.percentile(boot_rhos, 2.5))  if boot_rhos else np.nan
        ci_high = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else np.nan

        if rho >= 0.4:
            interp = "strong — language geometrically encodes this axis"
        elif rho >= 0.25:
            interp = "moderate — partial linguistic encoding"
        elif rho >= 0.1:
            interp = "weak — minimal surface-vocabulary signal"
        else:
            interp = "none — axis is latent, not expressed in vocabulary"

        results[axis] = {
            "rho": float(rho),
            "p": float(p),
            "rho_ci_low": ci_low,
            "rho_ci_high": ci_high,
            "n_items": n,
            "interpretation": interp,
        }

    return results


# ── H4: Combined signal — proposal deduplication ─────────────────────────────

def find_proposal_clusters(
    nn_8d: pd.DataFrame,
    nn_text: pd.DataFrame,
    text_threshold: float = 0.70,
    sim_8d_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Find groups of proposals that are BOTH linguistically similar (text ≥ threshold)
    AND ideologically aligned (8D ≥ threshold).

    The joint threshold is intentionally strict: it eliminates cases where
    topic overlap fools the text embedding, and cases where similar ideology
    uses completely different vocabulary. Only pairs that clear both bars are
    considered reliable duplicates.

    Uses union-find to chain pairs into clusters (if A~B and B~C, they are
    one cluster even if A and C were never directly compared).

    Args:
        nn_8d:            Output of nearest_neighbors_8d (item_id_a, item_id_b, cosine_similarity_8d).
        nn_text:          Output of nearest_neighbors_text (item_id_a, item_id_b, cosine_similarity_text).
        text_threshold:   Minimum text cosine similarity.
        sim_8d_threshold: Minimum 8D cosine similarity.

    Returns:
        DataFrame with columns:
            item_id, cluster_id, cluster_size
        Sorted by cluster_size descending.
    """
    # Normalise pair orientation so (a,b) always has a ≤ b
    def _normalise(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
        df = df.copy()
        swap = df["item_id_a"] > df["item_id_b"]
        df.loc[swap, ["item_id_a", "item_id_b"]] = (
            df.loc[swap, ["item_id_b", "item_id_a"]].values
        )
        return df[["item_id_a", "item_id_b", sim_col]].drop_duplicates(
            subset=["item_id_a", "item_id_b"]
        )

    t8  = _normalise(nn_8d[nn_8d["cosine_similarity_8d"]   >= sim_8d_threshold], "cosine_similarity_8d")
    txt = _normalise(nn_text[nn_text["cosine_similarity_text"] >= text_threshold],  "cosine_similarity_text")

    # Inner join: keep only pairs that pass BOTH thresholds
    both = t8.merge(txt, on=["item_id_a", "item_id_b"], how="inner")
    if both.empty:
        return pd.DataFrame(columns=["item_id", "cluster_id", "cluster_size"])

    # Union-Find
    parent: dict = {}

    def _find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)  # path compression
            x = parent.get(x, x)
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    for _, row in both.iterrows():
        _union(row["item_id_a"], row["item_id_b"])

    # Collect all item IDs touched by any high-confidence pair
    all_ids = set(both["item_id_a"]) | set(both["item_id_b"])
    rows = []
    for item_id in all_ids:
        root = _find(item_id)
        rows.append({"item_id": item_id, "cluster_id": root})

    cluster_df = pd.DataFrame(rows)
    sizes = cluster_df.groupby("cluster_id").size().rename("cluster_size").reset_index()
    cluster_df = cluster_df.merge(sizes, on="cluster_id")
    return (
        cluster_df
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def top_clusters_by_engagement(
    cluster_df: pd.DataFrame,
    interactions_df: "pd.DataFrame | None",
    items_df: pd.DataFrame,
    nn_both: "pd.DataFrame | None" = None,
    text_col: str = "text_norm",
    top_n: int = 10,
    min_cluster_size: int = 2,
) -> list[dict]:
    """
    Rank proposal clusters by user engagement (unique voters across all items
    in the cluster), then annotate each with its anchor pair (highest combined
    text+8D score).

    Args:
        cluster_df:       Output of find_proposal_clusters.
        interactions_df:  Raw interactions (user_id, politics_id, action).
                          If None, clusters are ranked by size alone.
        items_df:         Items with item_id + text_col.
        nn_both:          Inner-join pair table with both cosine_similarity_8d
                          and cosine_similarity_text (output from the merge
                          inside find_proposal_clusters — optional, used to
                          show anchor pair scores).
        text_col:         Column to pull proposal text from.
        top_n:            How many clusters to return.
        min_cluster_size: Minimum items in a cluster.

    Returns:
        List of dicts, one per cluster, sorted by n_voters descending:
            cluster_id, cluster_size, n_voters, items (list of item_ids),
            anchor_a_text, anchor_b_text, anchor_text_sim, anchor_8d_sim
    """
    if cluster_df.empty:
        return []

    cluster_df = cluster_df[cluster_df["cluster_size"] >= min_cluster_size].copy()
    text_map = dict(zip(items_df["item_id"], items_df[text_col].astype(str)))

    results = []
    for cid, grp in cluster_df.groupby("cluster_id"):
        item_ids = list(grp["item_id"].unique())

        # Engagement: unique users who voted on any item in this cluster
        n_voters = 0
        if interactions_df is not None and not interactions_df.empty:
            politics_col = next(
                (c for c in ("politics_id", "politika_id", "item_id") if c in interactions_df.columns),
                None,
            )
            if politics_col:
                n_voters = int(
                    interactions_df[interactions_df[politics_col].isin(item_ids)]["user_id"]
                    .nunique()
                )
        else:
            n_voters = len(item_ids)  # fallback: cluster size as proxy

        # Anchor pair: from nn_both, pick the pair with highest sum of both sims
        anchor_a, anchor_b, anchor_tsim, anchor_8dsim = "", "", 0.0, 0.0
        if nn_both is not None and not nn_both.empty:
            cluster_pairs = nn_both[
                (nn_both["item_id_a"].isin(item_ids)) & (nn_both["item_id_b"].isin(item_ids))
            ]
            if not cluster_pairs.empty:
                best = cluster_pairs.assign(
                    combined=cluster_pairs["cosine_similarity_text"] + cluster_pairs["cosine_similarity_8d"]
                ).nlargest(1, "combined").iloc[0]
                anchor_a     = text_map.get(best["item_id_a"], "")[:120]
                anchor_b     = text_map.get(best["item_id_b"], "")[:120]
                anchor_tsim  = float(best["cosine_similarity_text"])
                anchor_8dsim = float(best["cosine_similarity_8d"])

        results.append({
            "cluster_id":    cid,
            "cluster_size":  len(item_ids),
            "n_voters":      n_voters,
            "items":         item_ids,
            "anchor_a_text": anchor_a,
            "anchor_b_text": anchor_b,
            "anchor_text_sim": anchor_tsim,
            "anchor_8d_sim":   anchor_8dsim,
        })

    results.sort(key=lambda x: x["n_voters"], reverse=True)
    return results[:top_n]

