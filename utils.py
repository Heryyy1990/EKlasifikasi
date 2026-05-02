# ============================================================
# utils.py — Core logic SIKAP
# Berisi: load model, embed, search FAISS, scoring, ranking
# ============================================================

import os
import pickle
import numpy as np
import faiss
import streamlit as st

# ─────────────────────────────────────────────
# 1. LOAD BGE-M3 MODEL (cached — load sekali)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat model embedding BGE-M3...")
def load_embedding_model():
    """
    Load BGE-M3 menggunakan FlagEmbedding.
    use_fp16=True → hemat ~50% memori, cocok untuk Streamlit Cloud gratis.
    HF_TOKEN dibaca dari Streamlit Secrets untuk menghindari rate limit HuggingFace.
    """
    import os
    # Set HF_TOKEN jika ada di secrets (hindari rate limit HuggingFace)
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    except Exception:
        pass  # HF_TOKEN opsional, lanjut tanpa token

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,        # hemat memori
        device="cpu",         # Streamlit Cloud tidak punya GPU
    )
    return model


# ─────────────────────────────────────────────
# 2. LOAD FAISS INDEX (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat FAISS index...")
def load_faiss_index(index_path: str = "sikap_bge.index"):
    """
    Load FAISS index dari file .index.
    File ini berisi seluruh embedding dataset.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"File '{index_path}' tidak ditemukan. "
            "Pastikan file ada di root folder project."
        )
    index = faiss.read_index(index_path)
    return index


# ─────────────────────────────────────────────
# 3. LOAD METADATA (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat metadata klasifikasi...")
def load_metadata(meta_path: str = "metadata.pkl"):
    """
    Load metadata.pkl yang berisi mapping index → data klasifikasi.
    Ekspektasi struktur: list of dict dengan key:
    kode, uraian, penjelasan, konteks, domain, activity, embedding_text
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"File '{meta_path}' tidak ditemukan. "
            "Pastikan file ada di root folder project."
        )
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Normalkan ke list of dict jika berbentuk lain
    if hasattr(metadata, "to_dict"):
        # Jika pandas DataFrame
        metadata = metadata.to_dict(orient="records")
    elif isinstance(metadata, dict):
        # Jika dict of lists (format DataFrame)
        keys = list(metadata.keys())
        n = len(metadata[keys[0]])
        metadata = [{k: metadata[k][i] for k in keys} for i in range(n)]

    return metadata


# ─────────────────────────────────────────────
# 4. EMBED QUERY
# ─────────────────────────────────────────────

def embed_query(model, text: str) -> np.ndarray:
    """
    Ubah teks query menjadi vector embedding menggunakan BGE-M3.
    Return: numpy array shape (1, dim), dtype float32
    """
    output = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec = np.array(output["dense_vecs"], dtype=np.float32)
    # Normalisasi L2 agar cocok dengan inner product similarity
    faiss.normalize_L2(vec)
    return vec


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
    Return: list of dict kandidat dengan field asli + similarity_score
    """
    # FAISS search
    scores, indices = index.search(query_vec, top_k)
    scores = scores[0]    # (top_k,)
    indices = indices[0]  # (top_k,)

    candidates = []
    for rank, (idx, raw_score) in enumerate(zip(indices, scores)):
        if idx < 0 or idx >= len(metadata):
            continue  # skip invalid index

        row = dict(metadata[idx])  # copy agar aman

        # Konversi score ke rentang 0–1
        # Jika index pakai inner product (cosine), score sudah 0–1
        # Jika index pakai L2 distance, konversi: sim = 1 / (1 + dist)
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            sim = float(np.clip(raw_score, 0.0, 1.0))
        else:
            sim = float(1.0 / (1.0 + max(raw_score, 0.0)))

        row["_similarity"] = sim
        row["_rank"] = rank
        candidates.append(row)

    return candidates


# ─────────────────────────────────────────────
# 6. PARSE LEVEL KODE
# ─────────────────────────────────────────────

def get_kode_level(kode: str) -> int:
    """
    Tentukan level hierarki berdasarkan jumlah segmen kode.
    Contoh:
      "500"         → level 1 (primer)
      "500.2"       → level 2 (sekunder)
      "500.2.1"     → level 3 (tersier)
      "500.2.1.4"   → level 4 (kuartier)
    """
    if not kode or not isinstance(kode, str):
        return 0
    parts = str(kode).strip().split(".")
    return len(parts)


# ─────────────────────────────────────────────
# 7. SCORING SYSTEM
# ─────────────────────────────────────────────

def score_candidates(candidates: list) -> list:
    """
    Hitung FINAL_SCORE untuk setiap kandidat.

    Formula:
    FINAL_SCORE = (embedding_similarity × 0.70)
                + (domain_match × 0.20)
                + (activity_match × 0.10)

    domain_match dan activity_match ditentukan berdasarkan
    konsensus domain/activity dari top-5 kandidat teratas
    (mayoritas voting sederhana).
    """
    if not candidates:
        return []

    # Tentukan domain & activity dominan dari top-5
    top5 = candidates[:5]

    def dominant(field: str) -> str:
        values = [str(c.get(field, "")).strip().lower() for c in top5 if c.get(field)]
        if not values:
            return ""
        from collections import Counter
        return Counter(values).most_common(1)[0][0]

    dom_dominant = dominant("domain")
    act_dominant = dominant("activity")

    scored = []
    for c in candidates:
        sim   = c.get("_similarity", 0.0)
        dom   = str(c.get("domain",   "")).strip().lower()
        act   = str(c.get("activity", "")).strip().lower()

        domain_match   = 1.0 if (dom and dom == dom_dominant) else 0.0
        activity_match = 1.0 if (act and act == act_dominant) else 0.0

        final_score = (sim * 0.70) + (domain_match * 0.20) + (activity_match * 0.10)

        row = dict(c)
        row["_domain_match"]   = domain_match
        row["_activity_match"] = activity_match
        row["_final_score"]    = round(final_score, 4)
        scored.append(row)

    # Urutkan dari skor tertinggi
    scored.sort(key=lambda x: x["_final_score"], reverse=True)
    return scored


# ─────────────────────────────────────────────
# 8. HIERARCHICAL FILTERING & RANKING
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD_QUARTIER = 0.45   # min similarity untuk kuartier diterima
SIMILARITY_THRESHOLD_TERTIER  = 0.30   # min similarity untuk tersier diterima

def get_recommendations(scored_candidates: list, top_n: int = 3) -> list:
    """
    Pilih rekomendasi terbaik dengan strategi hierarchical:
    1. Utamakan kuartier (level 4) dengan similarity ≥ threshold
    2. Jika kurang, lengkapi dengan tersier (level 3)
    3. Hindari kode duplikat / kode induk yang sudah ada

    Return: list of dict hasil rekomendasi, maks top_n item.
    """
    quartier  = []   # level 4
    tertier   = []   # level 3
    seen_kode = set()

    for c in scored_candidates:
        kode  = str(c.get("kode", "")).strip()
        level = get_kode_level(kode)
        sim   = c.get("_similarity", 0.0)
        score = c.get("_final_score", 0.0)

        if kode in seen_kode:
            continue

        if level == 4 and sim >= SIMILARITY_THRESHOLD_QUARTIER:
            quartier.append(c)
            seen_kode.add(kode)
        elif level == 3 and sim >= SIMILARITY_THRESHOLD_TERTIER:
            tertier.append(c)
            seen_kode.add(kode)

    # Gabung: kuartier dulu, baru tersier jika kurang
    result = quartier[:top_n]
    if len(result) < top_n:
        needed = top_n - len(result)
        # Filter tersier yang bukan parent dari kuartier yang sudah masuk
        existing_parents = {".".join(k["kode"].split(".")[:3]) for k in result}
        extra_tertier = [
            t for t in tertier
            if t["kode"] not in existing_parents
        ]
        result.extend(extra_tertier[:needed])

    # Jika masih kurang, ambil sisa scored tanpa batasan level (best effort)
    if len(result) < top_n:
        used = {r["kode"] for r in result}
        fallback = [
            c for c in scored_candidates
            if c.get("kode") not in used
            and get_kode_level(c.get("kode", "")) >= 3
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
    Jalankan full pipeline klasifikasi:
    embed → search → score → recommend

    Return: list of dict rekomendasi
    """
    query_vec   = embed_query(model, inti_surat)
    candidates  = search_candidates(index, query_vec, metadata, top_k=40)
    scored      = score_candidates(candidates)
    recommendations = get_recommendations(scored, top_n=top_n)
    return recommendations
