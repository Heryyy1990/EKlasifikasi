# ============================================================
# app.py — SIKAP: Sistem Klasifikasi Arsip Terpadu
# Frontend Streamlit + orchestration pipeline
# ============================================================

import streamlit as st
import google.generativeai as genai

from prompts import EXTRACT_INTI_PROMPT
from utils import (
    load_embedding_model,
    load_faiss_index,
    load_metadata,
    classify,
    get_kode_level,
)

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SIKAP — Klasifikasi Arsip",
    page_icon="🗂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — UI bersih & profesional
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Font & background */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Judul utama */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a3c5e;
    margin-bottom: 0;
}
.sub-title {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* Card inti surat */
.inti-card {
    background: #eef6ff;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
}
.inti-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2563eb;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.inti-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a5f;
}

/* Card rekomendasi */
.rec-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.rec-card:hover {
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}
.rec-rank {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}
.rec-kode {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a3c5e;
    letter-spacing: 0.03em;
}
.rec-uraian {
    font-size: 0.92rem;
    color: #374151;
    margin-top: 0.3rem;
}
.rec-badge-q  { background:#dcfce7; color:#166534; }
.rec-badge-t  { background:#fef9c3; color:#854d0e; }
.rec-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 4px;
    padding: 1px 7px;
    margin-top: 0.4rem;
    margin-right: 5px;
}
.score-bar-wrap {
    background: #f3f4f6;
    border-radius: 50px;
    height: 6px;
    margin-top: 0.6rem;
}
.score-bar-fill {
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    border-radius: 50px;
    height: 6px;
}
.score-label {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 0.25rem;
}

/* Divider */
.divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INISIALISASI GEMINI API
# ─────────────────────────────────────────────

def init_gemini():
    """Baca API key dari Streamlit Secrets dan inisialisasi Gemini."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "⚠️ **GOOGLE_API_KEY tidak ditemukan.**\n\n"
            "Tambahkan di Streamlit Cloud: **Settings → Secrets**\n"
            "```\nGOOGLE_API_KEY = \"AIza...\"\n```"
        )
        st.stop()
    genai.configure(api_key=api_key)


# ─────────────────────────────────────────────
# EKSTRAK INTI SURAT VIA GEMINI
# ─────────────────────────────────────────────

def extract_inti_surat(text: str) -> str:
    """
    Panggil Gemini Flash untuk mengekstrak inti surat.
    Satu API call per klik Proses.
    """
    prompt  = EXTRACT_INTI_PROMPT.format(text=text.strip())
    model   = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,         # deterministic & konsisten
            max_output_tokens=30,    # inti surat pendek, hemat quota
        ),
    )
    inti = response.text.strip().lower()
    # Bersihkan tanda baca sisa
    inti = inti.strip(".,;:\"'")
    return inti


# ─────────────────────────────────────────────
# HELPER UI
# ─────────────────────────────────────────────

LEVEL_LABEL = {1: "Primer", 2: "Sekunder", 3: "Tersier", 4: "Kuartier"}

def render_recommendation_card(rank: int, rec: dict):
    """Render satu kartu rekomendasi."""
    kode    = str(rec.get("kode",   "—"))
    uraian  = str(rec.get("uraian", "—"))
    domain  = str(rec.get("domain", ""))
    sim     = rec.get("_similarity",  0.0)
    score   = rec.get("_final_score", 0.0)
    level   = get_kode_level(kode)
    level_name = LEVEL_LABEL.get(level, f"Level {level}")

    badge_class = "rec-badge-q" if level == 4 else "rec-badge-t"
    score_pct   = int(score * 100)
    sim_pct     = int(sim   * 100)

    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-rank">Rekomendasi #{rank}</div>
        <div class="rec-kode">{kode}</div>
        <div class="rec-uraian">{uraian}</div>
        <div>
            <span class="rec-badge {badge_class}">{level_name}</span>
            {"<span class='rec-badge' style='background:#ede9fe;color:#5b21b6;'>" + domain + "</span>" if domain else ""}
        </div>
        <div class="score-bar-wrap">
            <div class="score-bar-fill" style="width:{score_pct}%;"></div>
        </div>
        <div class="score-label">
            Skor akhir: <strong>{score_pct}%</strong>
            &nbsp;|&nbsp; Similarity: <strong>{sim_pct}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRE-LOAD RESOURCES (akan di-cache otomatis)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_resources():
    """Load semua resource sekaligus agar lebih efisien."""
    embed_model = load_embedding_model()
    faiss_index = load_faiss_index("sikap_bge.index")
    metadata    = load_metadata("metadata.pkl")
    return embed_model, faiss_index, metadata


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

def main():
    # Inisialisasi Gemini
    init_gemini()

    # Header
    st.markdown('<div class="main-title">🗂️ SIKAP</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Sistem Klasifikasi Arsip Terpadu '
        '— Rekomendasi kode arsip otomatis berbasis AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Muat resource (akan tampil spinner sekali saat cold start)
    with st.spinner("Memuat sistem klasifikasi..."):
        try:
            embed_model, faiss_index, metadata = load_all_resources()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.info(
                "Pastikan file **sikap_bge.index** dan **metadata.pkl** "
                "ada di root folder project dan sudah di-push ke GitHub."
            )
            st.stop()

    # Input area
    st.markdown("#### 📝 Masukkan isi surat / perihal / uraian dokumen")
    user_input = st.text_area(
        label="Isi surat",
        placeholder=(
            "Contoh:\n"
            "Permintaan pengadaan sistem arsip digital untuk mendukung "
            "pengelolaan dokumen di lingkungan kantor..."
        ),
        height=140,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        proses_btn = st.button("🔍 Proses", type="primary", use_container_width=True)
    with col2:
        st.markdown(
            "<small style='color:#9ca3af; line-height:2.5rem;'>"
            "Proses menggunakan Gemini + BGE-M3 + FAISS</small>",
            unsafe_allow_html=True,
        )

    # Proses saat tombol ditekan
    if proses_btn:
        if not user_input or len(user_input.strip()) < 5:
            st.warning("⚠️ Silakan masukkan isi surat terlebih dahulu (minimal 5 karakter).")
            st.stop()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Tahap 1: Ekstrak Inti Surat ──
        with st.spinner("🤖 Mengekstrak inti surat dengan Gemini..."):
            try:
                inti = extract_inti_surat(user_input)
            except Exception as e:
                st.error(f"❌ Gagal memanggil Gemini API: {e}")
                st.stop()

        st.markdown(f"""
        <div class="inti-card">
            <div class="inti-label">✦ Inti Surat Terdeteksi</div>
            <div class="inti-text">"{inti}"</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tahap 2–4: Embed → Search → Score → Recommend ──
        with st.spinner("🔎 Mencari klasifikasi yang relevan..."):
            try:
                recommendations = classify(
                    inti_surat  = inti,
                    model       = embed_model,
                    index       = faiss_index,
                    metadata    = metadata,
                    top_n       = 3,
                )
            except Exception as e:
                st.error(f"❌ Gagal menjalankan klasifikasi: {e}")
                st.stop()

        # ── Output ──
        if not recommendations:
            st.warning(
                "⚠️ Tidak ditemukan klasifikasi yang relevan. "
                "Coba tulis uraian yang lebih spesifik."
            )
        else:
            st.markdown("#### 🏆 Rekomendasi Kode Klasifikasi Arsip")
            for i, rec in enumerate(recommendations, start=1):
                render_recommendation_card(i, rec)

            # Catatan kaki
            st.markdown(
                "<small style='color:#9ca3af;'>"
                "Kuartier (4 level) diprioritaskan. "
                "Tersier (3 level) digunakan sebagai fallback jika similarity rendah."
                "</small>",
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        "<center><small style='color:#d1d5db;'>"
        "SIKAP v1.0 · Powered by Gemini Flash + BGE-M3 + FAISS"
        "</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
