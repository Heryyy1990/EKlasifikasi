cat > /home/claude/sikap-app/utils.py << 'ENDOFFILE'
# ============================================================
# utils.py — Core logic SIKAP
# Model: paraphrase-multilingual-MiniLM-L12-v2 (118MB)
# Pipeline: embed query_expansion → FAISS search → score → rank
# ============================================================

import os
import pickle
import numpy as np
import faiss
import streamlit as st

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ─────────────────────────────────────────────
# 1. LOAD MODEL EMBEDDING (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat model embedding...")
def load_embedding_model():
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
        if hf_token:
            os.environ["HF_TOKEN"]              = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model


# ─────────────────────────────────────────────
# 2. LOAD FAISS INDEX (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat FAISS index...")
def load_faiss_index(index_path: str = "sikap_bge.index"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"File '{index_path}' tidak ditemukan. "
            "Pastikan sudah di-push ke GitHub."
        )
    return faiss.read_index(index_path)


# ─────────────────────────────────────────────
# 3. LOAD METADATA (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat metadata klasifikasi...")
def load_metadata(meta_path: str = "metadata.pkl"):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"File '{meta_path}' tidak ditemukan. "
            "Pastikan sudah di-push ke GitHub."
        )
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict(orient="records")
    elif isinstance(metadata, dict):
        keys = list(metadata.keys())
        n    = len(metadata[keys[0]])
        metadata = [{k: metadata[k][i] for k in keys} for i in range(n)]

    return metadata


# ─────────────────────────────────────────────
# 4. EMBED TEKS
# ─────────────────────────────────────────────

def embed_text(model, text: str) -> np.ndarray:
    """
    Embed satu teks → numpy array (1, dim) float32, normalized.
    """
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return vec


# ─────────────────────────────────────────────
# 5. SEARCH FAISS
# ─────────────────────────────────────────────

def search_candidates(index, query_vec: np.ndarray, metadata: list, top_k: int = 40) -> list:
    scores, indices = index.search(query_vec, top_k)
    scores  = scores[0]
    indices = indices[0]

    candidates = []
    for rank, (idx, raw_score) in enumerate(zip(indices, scores)):
        if idx < 0 or idx >= len(metadata):
            continue
        row = dict(metadata[idx])
        sim = float(np.clip(raw_score, 0.0, 1.0)) \
              if index.metric_type == faiss.METRIC_INNER_PRODUCT \
              else float(1.0 / (1.0 + max(raw_score, 0.0)))
        row["_similarity"] = sim
        row["_rank"]       = rank
        candidates.append(row)

    return candidates


# ─────────────────────────────────────────────
# 6. KEYWORD OVERLAP BONUS
# ─────────────────────────────────────────────

def keyword_overlap_score(query: str, doc: dict) -> float:
    """
    Hitung overlap kata antara query dan teks dokumen (uraian + domain + activity).
    Memberikan bonus untuk kecocokan kata kunci spesifik.
    Return: 0.0 – 1.0
    """
    import re
    stop_words = {
        "dan", "atau", "yang", "untuk", "dari", "ke", "di", "pada", "dengan",
        "adalah", "ini", "itu", "oleh", "dalam", "juga", "tidak", "sebagai",
        "akan", "telah", "sudah", "dapat", "serta", "hal", "sesuai"
    }

    def tokenize(text: str) -> set:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {t for t in tokens if t not in stop_words}

    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    doc_text = " ".join(filter(None, [
        str(doc.get("uraian",      "")),
        str(doc.get("penjelasan",  "")),
        str(doc.get("konteks",     "")),
        str(doc.get("domain",      "")),
        str(doc.get("activity",    "")),
    ]))
    doc_tokens = tokenize(doc_text)

    if not doc_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens)
    # Normalize terhadap ukuran query (recall-oriented)
    return min(overlap / len(query_tokens), 1.0)


# ─────────────────────────────────────────────
# 7. SCORING SYSTEM
# ─────────────────────────────────────────────

def score_candidates(candidates: list, query: str) -> list:
    """
    FINAL_SCORE = (embedding_similarity  × 0.55)
                + (keyword_overlap       × 0.25)
                + (domain_match          × 0.12)
                + (activity_match        × 0.08)

    keyword_overlap: bonus kecocokan kata kunci eksplisit antara query dan dokumen.
    domain/activity_match: konsensus dari top-5 kandidat.
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

        kw_score       = keyword_overlap_score(query, c)
        domain_match   = 1.0 if (dom and dom == dom_dominant) else 0.0
        activity_match = 1.0 if (act and act == act_dominant) else 0.0

        final_score = (
            sim            * 0.55 +
            kw_score       * 0.25 +
            domain_match   * 0.12 +
            activity_match * 0.08
        )

        row = dict(c)
        row["_kw_score"]       = round(kw_score,   4)
        row["_domain_match"]   = domain_match
        row["_activity_match"] = activity_match
        row["_final_score"]    = round(final_score, 4)
        scored.append(row)

    scored.sort(key=lambda x: x["_final_score"], reverse=True)
    return scored


# ─────────────────────────────────────────────
# 8. PARSE LEVEL KODE
# ─────────────────────────────────────────────

def get_kode_level(kode: str) -> int:
    if not kode or not isinstance(kode, str):
        return 0
    return len(str(kode).strip().split("."))


# ─────────────────────────────────────────────
# 9. HIERARCHICAL FILTERING
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD_QUARTIER = 0.35
SIMILARITY_THRESHOLD_TERTIER  = 0.20

def get_recommendations(scored_candidates: list, top_n: int = 3) -> list:
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

    # Best-effort fallback
    if len(result) < top_n:
        used = {r["kode"] for r in result}
        fallback = [
            c for c in scored_candidates
            if c.get("kode") not in used and get_kode_level(c.get("kode", "")) >= 3
        ]
        result.extend(fallback[: top_n - len(result)])

    return result[:top_n]


# ─────────────────────────────────────────────
# 10. PIPELINE UTAMA
# ─────────────────────────────────────────────

def classify(inti_surat: str, model, index, metadata: list, top_n: int = 3) -> list:
    """
    Full pipeline: embed query_expansion → search → score (dengan keyword overlap) → rank
    Parameter inti_surat di sini sebenarnya berisi search_query (hasil query expansion Gemini).
    """
    query_vec       = embed_text(model, inti_surat)
    candidates      = search_candidates(index, query_vec, metadata, top_k=50)
    scored          = score_candidates(candidates, query=inti_surat)
    recommendations = get_recommendations(scored, top_n=top_n)
    return recommendations
ENDOFFILE
echo "utils.py written"
