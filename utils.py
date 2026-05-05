# ============================================================
# utils.py — Core logic SIKAP
# Model: paraphrase-multilingual-MiniLM-L12-v2 (118MB)
# Ringan untuk Streamlit Cloud free tier (800MB RAM limit)
# ============================================================

import os
import pickle
import numpy as np
import faiss
import streamlit as st

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Kenapa model ini (bukan BGE-M3)?
#   BGE-M3      : ~1.1 GB RAM → OOM di Streamlit Cloud free tier
#   MiniLM-L12  : ~118 MB RAM → aman, support 50+ bahasa + Indonesia ✓
# PENTING: FAISS index HARUS direbuild dulu dengan rebuild_index.py!


# ─────────────────────────────────────────────
# 1. LOAD MODEL EMBEDDING (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat model embedding...")
def load_embedding_model():
    """
    Load sentence-transformers model yang ringan.
    Di-cache Streamlit → load sekali, reuse terus.
    """
    # Set HF_TOKEN jika ada di secrets (percepat download HuggingFace)
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
        if hf_token:
            os.environ["HF_TOKEN"]              = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    except Exception:
        pass  # HF_TOKEN opsional

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model


# ─────────────────────────────────────────────
# 2. LOAD FAISS INDEX (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat FAISS index...")
def load_faiss_index(index_path: str = "sikap_bge.index"):
    """
    Load FAISS index dari file .index.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"File '{index_path}' tidak ditemukan. "
            "Pastikan file ada di root folder project dan sudah di-push ke GitHub."
        )
    index = faiss.read_index(index_path)
    return index


# ─────────────────────────────────────────────
# 3. LOAD METADATA (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat metadata klasifikasi...")
def load_metadata(meta_path: str = "metadata.pkl"):
    """
    Load metadata.pkl — mapping index FAISS ke data klasifikasi.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"File '{meta_path}' tidak ditemukan. "
            "Pastikan file ada di root folder project dan sudah di-push ke GitHub."
        )
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Normalkan ke list of dict
    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict(orient="records")
    elif isinstance(metadata, dict):
        keys = list(metadata.keys())
        n = len(metadata[keys[0]])
        metadata = [{k: metadata[k][i] for k in keys} for i in range(n)]

    return metadata


# ─────────────────────────────────────────────
# 4. EMBED QUERY
# ─────────────────────────────────────────────

def embed_query(model, text: str) -> np.ndarray:
    """
    Ubah teks query menjadi vector embedding.
    normalize_embeddings=True agar cocok dengan inner product similarity.
    Return: numpy array shape (1, dim), dtype float32
    """
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return vec  # shape: (1, dim)


# ─────────────────────────────────────────────
# 5. SEARCH FAISS CANDIDATES
# ─────────────────────────────────────────────

def search_candidates(
    index,
    query_vec: np.ndarray,
    metadata: list,
    top_k: int = 30,
) -> list:
    """
    Cari top_k kandidat paling mirip dari FAISS index.
    Return: list of dict kandidat + similarity_score
    """
    scores, indices = index.search(query_vec, top_k)
    scores  = scores[0]
    indices = indices[0]

    candidates = []
    for rank, (idx, raw_score) in enumerate(zip(indices, scores)):
        if idx < 0 or idx >= len(metadata):
            continue

        row = dict(metadata[idx])

        # Inner product dengan normalized vectors = cosine similarity (0–1)
        # L2 distance: konversi ke similarity
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            sim = float(np.clip(raw_score, 0.0, 1.0))
        else:
            sim = float(1.0 / (1.0 + max(raw_score, 0.0)))

        row["_similarity"] = sim
        row["_rank"]       = rank
        candidates.append(row)

    return candidates


# ─────────────────────────────────────────────
# 6. PARSE LEVEL KODE
# ─────────────────────────────────────────────

def get_kode_level(kode: str) -> int:
    """
    Hitung level hierarki kode.
    "500"       → 1 (primer)
    "500.2"     → 2 (sekunder)
    "500.2.1"   → 3 (tersier)
    "500.2.1.4" → 4 (kuartier)
    """
    if not kode or not isinstance(kode, str):
        return 0
    return len(str(kode).strip().split("."))


# ─────────────────────────────────────────────
# 7. SCORING SYSTEM
# ─────────────────────────────────────────────

def score_candidates(candidates: list) -> list:
    """
    FINAL_SCORE = (embedding_similarity × 0.70)
                + (domain_match         × 0.20)
                + (activity_match       × 0.10)

    domain_match & activity_match ditentukan dari mayoritas top-5.
    """
    if not candidates:
        return []

    from collections import Counter

    def dominant(field: str) -> str:
        vals = [
            str(c.get(field, "")).strip().lower()
            for c in candidates[:5] if c.get(field)
        ]
        return Counter(vals).most_common(1)[0][0] if vals else ""

    dom_dominant = dominant("domain")
    act_dominant = dominant("activity")

    scored = []
    for c in candidates:
        sim = c.get("_similarity", 0.0)
        dom = str(c.get("domain",   "")).strip().lower()
        act = str(c.get("activity", "")).strip().lower()

        domain_match   = 1.0 if (dom and dom == dom_dominant) else 0.0
        activity_match = 1.0 if (act and act == act_dominant) else 0.0

        final_score = (sim * 0.70) + (domain_match * 0.20) + (activity_match * 0.10)

        row = dict(c)
        row["_domain_match"]   = domain_match
        row["_activity_match"] = activity_match
        row["_final_score"]    = round(final_score, 4)
        scored.append(row)

    scored.sort(key=lambda x: x["_final_score"], reverse=True)
    return scored


# ─────────────────────────────────────────────
# 8. HIERARCHICAL FILTERING & RANKING
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD_QUARTIER = 0.40
SIMILARITY_THRESHOLD_TERTIER  = 0.25

def get_recommendations(scored_candidates: list, top_n: int = 3) -> list:
    """
    Pilih rekomendasi:
    1. Prioritas kuartier (level 4) dengan similarity ≥ threshold
    2. Fallback tersier (level 3) jika kuartier kurang
    3. Best-effort jika masih kurang
    """
    quartier  = []
    tertier   = []
    seen_kode = set()

    for c in scored_candidates:
        kode  = str(c.get("kode", "")).strip()
        level = get_kode_level(kode)
        sim   = c.get("_similarity", 0.0)

        if kode in seen_kode:
            continue

        if level == 4 and sim >= SIMILARITY_THRESHOLD_QUARTIER:
            quartier.append(c)
            seen_kode.add(kode)
        elif level == 3 and sim >= SIMILARITY_THRESHOLD_TERTIER:
            tertier.append(c)
            seen_kode.add(kode)

    result = quartier[:top_n]

    if len(result) < top_n:
        needed = top_n - len(result)
        existing_parents = {".".join(k["kode"].split(".")[:3]) for k in result}
        extra = [t for t in tertier if t["kode"] not in existing_parents]
        result.extend(extra[:needed])

    # Fallback best-effort
    if len(result) < top_n:
        used = {r["kode"] for r in result}
        fallback = [
            c for c in scored_candidates
            if c.get("kode") not in used and get_kode_level(c.get("kode", "")) >= 3
        ]
        result.extend(fallback[: top_n - len(result)])

    return result[:top_n]


# ─────────────────────────────────────────────
# 9. PIPELINE UTAMA
# ─────────────────────────────────────────────

def classify(
    inti_surat: str,
    model,
    index,
    metadata: list,
    top_n: int = 3,
) -> list:
    """
    Full pipeline: embed → search → score → recommend
    """
    query_vec       = embed_query(model, inti_surat)
    candidates      = search_candidates(index, query_vec, metadata, top_k=40)
    scored          = score_candidates(candidates)
    recommendations = get_recommendations(scored, top_n=top_n)
    return recommendations
