import streamlit as st
from google import genai
from utils import load_system, extract_intent, search_classification
from prompts import EXTRACTION_PROMPT

# Konfigurasi Halaman
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

st.title("🗂️ SIKAP")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar")
st.write("Masukkan perihal/uraian surat yang panjang. AI akan memahami inti surat dan mencarikan 3 rekomendasi klasifikasi paling presisi (sampai level Kuartier/Tersier).")

# 1. Load Sistem (Model, FAISS, Metadata)
with st.spinner("Menyiapkan Sistem dan Database Klasifikasi..."):
    model, index, df, kode_dict = load_system()

# 2. Inisialisasi API Key Google Gemini dari Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("API Key Google Gemini belum di-set di Streamlit Secrets!")
    st.stop()

# 3. Input Pengguna
user_input = st.text_area("Uraian Surat:", placeholder="Contoh: Mohon bantuan penerbitan surat pengadaan sistem arsip digital tahun 2026...", height=120)

# 4. Tombol Eksekusi
if st.button("Cari Kode Klasifikasi", type="primary"):
    if not user_input.strip():
        st.warning("Silakan ketik uraian surat terlebih dahulu.")
    else:
        try:
            # Tahap 1: Ekstraksi Gemini
            with st.spinner("🤖 Menggunakan Gemini untuk merangkum inti surat..."):
                inti_surat = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            # Tampilkan hasil ektraksi agar user tahu
            st.success(f"**Inti Pencarian:** {inti_surat}")
            
            # Tahap 2: Semantic Search dengan Vektor
            with st.spinner("🔍 Mencocokkan dengan database klasifikasi (FAISS)..."):
                rekomendasi = search_classification(model, index, df, kode_dict, inti_surat)
                
            # Tahap 3: Tampilkan Hasil
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                
                for idx, rec in enumerate(rekomendasi):
                    # Desain Card Sederhana
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.markdown(f"**Tingkat Akurasi (Vektor):** `{rec['score']:.4f}`")
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan pada level Tersier/Kuartier.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada sistem: {e}")
