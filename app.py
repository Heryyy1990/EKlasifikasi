import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import re
import json
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SIKAP – Sistem Identifikasi Klasifikasi Arsip Pintar",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 900px; }

/* ── Header ── */
.sikap-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.sikap-logo {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.sikap-tagline {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 0.4rem;
    letter-spacing: 0.5px;
}

/* ── Input Card ── */
.input-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.card-label {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.6rem;
}

/* ── Streamlit textarea & input overrides ── */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.15) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(14, 165, 233, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.5rem 0 0.8rem;
}
.result-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
}
.result-card.rank-1 {
    border-color: rgba(56, 189, 248, 0.5);
    background: rgba(14, 165, 233, 0.08);
}
.result-card.rank-1::before {
    content: '★ REKOMENDASI UTAMA';
    position: absolute;
    top: 0; right: 0;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 12px;
    border-radius: 0 14px 0 10px;
}
.result-rank {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.result-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 1px;
}
.result-uraian {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0.2rem 0;
}
.result-meta {
    display: flex;
    gap: 0.7rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 2px 10px;
    border-radius: 20px;
    display: inline-block;
}
.badge-level {
    background: rgba(129, 140, 248, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(129, 140, 248, 0.3);
}
.badge-score {
    background: rgba(52, 211, 153, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(52, 211, 153, 0.25);
}
.badge-domain {
    background: rgba(251, 191, 36, 0.12);
    color: #fcd34d;
    border: 1px solid rgba(251, 191, 36, 0.2);
}
.result-detail {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ── Intent box ── */
.intent-box {
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
}
.intent-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.3rem;
}
.intent-text {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* ── Divider ── */
.sikap-divider {
    border: none;
    border-top: 1px solid rgba(56, 189, 248, 0.1);
    margin: 1.5rem 0;
}

/* ── Alert / warning ── */
.stAlert { border-radius: 10px !important; }

/* ── No result ── */
.no-result {
    text-align: center;
    padding: 2rem;
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CSV_PATH        = "klasifikasi_sikap_final_v2.csv"
META_KUARTIER   = "meta_kuartier.pkl"
META_TERSIER    = "meta_tersier.pkl"
INDEX_KUARTIER  = "faiss_kuartier.index"
INDEX_TERSIER   = "faiss_tersier.index"

EMBEDDING_MODEL = "models/text-embedding-002"
GEMINI_MODEL    = "gemini-2.5-flash"
EMBED_DIM       = 768
CONFIDENCE_THR  = 0.72   # skor cosine similarity ambang batas

# ─────────────────────────────────────────────
# HELPERS – file existence check
# ─────────────────────────────────────────────
def check_required_files():
    missing = []
    for f in [CSV_PATH, META_KUARTIER, META_TERSIER, INDEX_KUARTIER, INDEX_TERSIER]:
        if not os.path.exists(f):
            missing.append(f)
    return missing

# ─────────────────────────────────────────────
# LOAD RESOURCES  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    with open(META_KUARTIER, "rb") as f:
        meta_kuartier = pickle.load(f)
    with open(META_TERSIER, "rb") as f:
        meta_tersier = pickle.load(f)

    idx_kuartier = faiss.read_index(INDEX_KUARTIER)
    idx_tersier  = faiss.read_index(INDEX_TERSIER)

    return df, meta_kuartier, meta_tersier, idx_kuartier, idx_tersier

# ─────────────────────────────────────────────
# GEMINI CLIENT  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_genai_client():
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def get_embedding(client, text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    # L2 normalize untuk cosine similarity via inner product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

# ─────────────────────────────────────────────
# GEMINI – EXTRACT INTENT & PRIMER/SEKUNDER
# ─────────────────────────────────────────────
def _safe_parse_json(raw: str) -> dict:
    """
    Coba berbagai cara parse JSON dari output Gemini yang tidak selalu rapi.
    Urutan upaya:
      1. Parse langsung
      2. Strip markdown fences (```json … ```)
      3. Cari blok { … } pertama dengan regex
      4. Regex per-field sebagai last resort
    """
    text = raw.strip()

    # ── Upaya 1: langsung ──
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ── Upaya 2: hapus markdown fences ──
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Upaya 3: ambil blok JSON pertama { … } ──
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # ── Upaya 4: regex per-field (fallback kasar) ──
    def _extract(pattern, default=""):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip().strip('"').strip("'") if m else default

    intent_summary = _extract(r'"intent_summary"\s*:\s*"([^"]*)"')
    kode_primer    = _extract(r'"kode_primer"\s*:\s*"([^"]*)"')
    kode_sekunder  = _extract(r'"kode_sekunder"\s*:\s*"([^"]*)"')
    domain         = _extract(r'"domain"\s*:\s*"([^"]*)"')

    kw_match = re.search(r'"keywords"\s*:\s*\[([^\]]*)\]', text)
    if kw_match:
        keywords = [k.strip().strip('"').strip("'") for k in kw_match.group(1).split(",") if k.strip()]
    else:
        keywords = []

    if kode_primer:
        return {
            "intent_summary": intent_summary or "Tidak dapat dianalisis.",
            "kode_primer":    kode_primer,
            "kode_sekunder":  kode_sekunder,
            "keywords":       keywords,
            "domain":         domain or "umum",
        }

    # ── Upaya 5: benar-benar gagal → kembalikan default aman ──
    return {
        "intent_summary": "Analisis otomatis tidak berhasil – menggunakan pencarian penuh.",
        "kode_primer":    "",
        "kode_sekunder":  "",
        "keywords":       [],
        "domain":         "umum",
    }


def extract_intent(client, user_text: str, df: pd.DataFrame) -> dict:
    """
    Minta Gemini menganalisis input dan menentukan:
    - ringkasan intent
    - kode_primer (2–3 digit, mis: 000 / 500 / 900)
    - kode_sekunder (mis: 510 / 921)
    - keywords
    Kembalikan dict. Tidak akan raise exception.
    """
    # Bangun daftar primer unik sebagai referensi
    primer_list = (
        df[df["level"] == "primer"][["kode", "uraian"]]
        .drop_duplicates()
        .head(40)
        .to_dict(orient="records")
    )
    primer_str = "\n".join(
        f'  - {r["kode"]} : {r["uraian"]}' for r in primer_list
    )

    prompt = f"""Kamu adalah ahli klasifikasi arsip pemerintahan Indonesia.
Analisis teks surat berikut dan tentukan klasifikasinya.

TEKS SURAT:
\"\"\"{user_text}\"\"\"

DAFTAR KODE PRIMER:
{primer_str}

INSTRUKSI PENTING:
- Balas HANYA dengan satu objek JSON valid.
- JANGAN tulis apapun sebelum atau sesudah JSON.
- JANGAN gunakan markdown, backtick, atau komentar.
- Gunakan tanda kutip ganda untuk semua string.

FORMAT WAJIB:
{{"intent_summary":"<ringkasan 1-2 kalimat>","kode_primer":"<kode 2-3 digit>","kode_sekunder":"<kode 3 digit atau kosong>","keywords":["<kw1>","<kw2>","<kw3>"],"domain":"<domain>"}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=400,
            ),
        )
        raw = response.text or ""
    except Exception as e:
        # Jika API call gagal total, kembalikan default
        return {
            "intent_summary": f"API error: {e}",
            "kode_primer":    "",
            "kode_sekunder":  "",
            "keywords":       [],
            "domain":         "umum",
        }

    return _safe_parse_json(raw)

# ─────────────────────────────────────────────
# FILTER CANDIDATES BY PRIMER / SEKUNDER
# ─────────────────────────────────────────────
def filter_candidates(meta: list, kode_primer: str, kode_sekunder: str) -> list[int]:
    """Kembalikan indeks dalam meta yang cocok dengan primer/sekunder."""
    indices = []
    for i, m in enumerate(meta):
        kode = str(m.get("kode", ""))
        if kode_sekunder:
            if kode.startswith(kode_sekunder):
                indices.append(i)
        elif kode_primer:
            if kode.startswith(kode_primer[:1]):  # digit pertama primer
                indices.append(i)
    # fallback: kembalikan semua jika kosong
    return indices if indices else list(range(len(meta)))

# ─────────────────────────────────────────────
# VECTOR SEARCH ON SUBSET
# ─────────────────────────────────────────────
def vector_search_subset(
    faiss_index,
    meta: list,
    query_vec: np.ndarray,
    candidate_indices: list[int],
    top_k: int = 5,
) -> list[dict]:
    """
    Strategi: ambil top-k*4 dari index global, lalu filter hanya
    yang masuk candidate_indices. Ini menghindari rebuild index parsial
    (hemat memori di Streamlit Cloud gratis).
    """
    k_search = min(top_k * 6, faiss_index.ntotal)
    q = query_vec.reshape(1, -1)
    scores, ids = faiss_index.search(q, k_search)

    candidate_set = set(candidate_indices)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if idx in candidate_set:
            results.append({**meta[idx], "score": float(score)})
        if len(results) >= top_k:
            break

    # fallback: jika hasil kurang dari top_k, relaksasi filter
    if len(results) < top_k:
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if idx not in candidate_set:
                results.append({**meta[idx], "score": float(score)})
            if len(results) >= top_k:
                break

    return results

# ─────────────────────────────────────────────
# MAIN CLASSIFICATION PIPELINE
# ─────────────────────────────────────────────
def classify(client, user_text: str, df, meta_kuartier, meta_tersier, idx_kuartier, idx_tersier):
    # 1. Extract intent via Gemini
    intent = extract_intent(client, user_text, df)

    kode_primer   = intent.get("kode_primer", "").strip()
    kode_sekunder = intent.get("kode_sekunder", "").strip()

    # 2. Embed query
    query_vec = get_embedding(client, user_text)

    # 3. Filter kandidat kuartier
    cand_kuartier = filter_candidates(meta_kuartier, kode_primer, kode_sekunder)

    # 4. Vector search kuartier
    top_kuartier = vector_search_subset(idx_kuartier, meta_kuartier, query_vec, cand_kuartier, top_k=3)

    # 5. Confidence check
    top_score = top_kuartier[0]["score"] if top_kuartier else 0.0
    use_tersier_fallback = top_score < CONFIDENCE_THR

    top_tersier = []
    if use_tersier_fallback:
        cand_tersier = filter_candidates(meta_tersier, kode_primer, kode_sekunder)
        top_tersier  = vector_search_subset(idx_tersier, meta_tersier, query_vec, cand_tersier, top_k=2)

    return {
        "intent": intent,
        "kuartier": top_kuartier,
        "tersier": top_tersier,
        "confidence_ok": not use_tersier_fallback,
    }

# ─────────────────────────────────────────────
# RENDER RESULT CARD
# ─────────────────────────────────────────────
def render_result_card(item: dict, rank: int):
    rank_cls  = "rank-1" if rank == 1 else ""
    score_pct = f"{item['score'] * 100:.1f}%"
    level     = item.get("level", "-").capitalize()
    domain    = item.get("domain", item.get("keywords", ""))
    if isinstance(domain, list):
        domain = ", ".join(domain[:3])
    uraian    = item.get("uraian", "-")
    kode      = item.get("kode", "-")
    penjelasan = item.get("penjelasan", item.get("konteks", ""))

    detail_html = f'<div class="result-detail">{penjelasan[:220]}{"…" if len(penjelasan)>220 else ""}</div>' if penjelasan else ""

    st.markdown(f"""
<div class="result-card {rank_cls}">
  <div class="result-rank">#{rank}</div>
  <div class="result-code">{kode}</div>
  <div class="result-uraian">{uraian}</div>
  <div class="result-meta">
    <span class="badge badge-level">📁 {level}</span>
    <span class="badge badge-score">✓ {score_pct}</span>
    {'<span class="badge badge-domain">🏷 ' + domain + '</span>' if domain else ''}
  </div>
  {detail_html}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
<div class="sikap-header">
  <div class="sikap-logo">SIKAP</div>
  <div class="sikap-tagline">Sistem Identifikasi Klasifikasi Arsip Pintar &nbsp;·&nbsp; Powered by Gemini + FAISS</div>
</div>
""", unsafe_allow_html=True)

    # ── File check ──
    missing = check_required_files()
    if missing:
        st.error(
            f"⚠️ File berikut tidak ditemukan di repositori:\n\n"
            + "\n".join(f"- `{f}`" for f in missing)
            + "\n\nPastikan semua file sudah di-push ke GitHub sebelum deploy."
        )
        st.stop()

    # ── Load resources ──
    with st.spinner("Memuat model & indeks…"):
        df, meta_kuartier, meta_tersier, idx_kuartier, idx_tersier = load_resources()

    client = get_genai_client()
    if client is None:
        st.error(
            "🔑 **GOOGLE_API_KEY** belum dikonfigurasi.\n\n"
            "Tambahkan di **Settings → Secrets** Streamlit Cloud:\n"
            "```\nGOOGLE_API_KEY = \"AIza...\"\n```"
        )
        st.stop()

    # ── Input ──
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📝 Uraian / Perihal / Ringkasan Surat</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        placeholder="Contoh: Surat undangan mengikuti Bimbingan Teknis Pengelolaan Keuangan Daerah Tahun 2025…",
        height=130,
        label_visibility="collapsed",
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.button("🔍 Cari Kode Klasifikasi", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Processing ──
    if submitted:
        if not user_input.strip():
            st.warning("Masukkan uraian surat terlebih dahulu.")
            st.stop()

        with st.spinner("Menganalisis & mencari kode klasifikasi…"):
            try:
                result = classify(
                    client, user_input.strip(),
                    df, meta_kuartier, meta_tersier, idx_kuartier, idx_tersier
                )
            except json.JSONDecodeError:
                st.error("Gemini mengembalikan respons tidak valid. Coba ulangi.")
                st.stop()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.stop()

        intent = result["intent"]

        # ── Intent summary ──
        st.markdown(f"""
<div class="intent-box">
  <div class="intent-label">🤖 Analisis Gemini</div>
  <div class="intent-text">
    {intent.get('intent_summary', '-')}<br>
    <strong style="color:#94a3b8">Primer:</strong> {intent.get('kode_primer','-')} &nbsp;|&nbsp;
    <strong style="color:#94a3b8">Sekunder:</strong> {intent.get('kode_sekunder','-') or '—'} &nbsp;|&nbsp;
    <strong style="color:#94a3b8">Domain:</strong> {intent.get('domain','-')} &nbsp;|&nbsp;
    <strong style="color:#94a3b8">Kata kunci:</strong> {', '.join(intent.get('keywords', []))}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Kuartier results ──
        kuartier = result["kuartier"]
        if kuartier:
            st.markdown('<div class="result-section-title">🎯 Top 3 Kode Kuartier (Paling Spesifik)</div>', unsafe_allow_html=True)
            for i, item in enumerate(kuartier, 1):
                render_result_card(item, i)
        else:
            st.markdown('<div class="no-result">Tidak ditemukan kode kuartier yang cocok.</div>', unsafe_allow_html=True)

        # ── Tersier fallback ──
        tersier = result["tersier"]
        if tersier:
            st.markdown('<hr class="sikap-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-section-title">🔁 Alternatif Kode Tersier (Confidence Rendah – Fallback)</div>',
                unsafe_allow_html=True,
            )
            for i, item in enumerate(tersier, 1):
                render_result_card(item, i)

        if result["confidence_ok"]:
            st.success("✅ Confidence tinggi — hasil kuartier diyakini akurat.")
        else:
            st.info("ℹ️ Confidence rendah — disertakan kode tersier sebagai alternatif.")

    # ── Footer ──
    st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; margin-top:3rem; padding-bottom:1rem;">
  SIKAP &nbsp;·&nbsp; Klasifikasi Arsip Pemerintahan &nbsp;·&nbsp; Gemini 2.5 Flash + FAISS
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
