import streamlit as st
from google import genai
from utils import load_system, extract_intent, search_classification
from prompts import EXTRACTION_PROMPT

st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

st.title("🗂️ SIKAP")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar")
st.write("Masukkan perihal/uraian surat yang panjang. AI akan memahami inti surat dan mencarikan 3 rekomendasi klasifikasi paling presisi.")

with st.spinner("Menyiapkan Sistem dan Database Klasifikasi..."):
    model, index, df, kode_dict = load_system()

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("API Key Google Gemini belum di-set di Streamlit Secrets!")
    st.stop()

user_input = st.text_area("Uraian Surat:", placeholder="Contoh: Mohon bantuan penerbitan surat pengadaan sistem arsip digital tahun 2026...", height=120)

if st.button("Cari Kode Klasifikasi", type="primary"):
    if not user_input.strip():
        st.warning("Silakan ketik uraian surat terlebih dahulu.")
    else:
        try:
            with st.spinner("🤖 Menggunakan Gemini untuk menganalisis konteks surat..."):
                intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            st.success(f"**Vektor Kata Kunci:** {intent_json.get('intent_query', 'N/A')}")
            st.info(f"**Prediksi Kategori:** {intent_json.get('domain_prediksi', 'N/A')} ➔ {intent_json.get('activity_prediksi', 'N/A')}")
            
            with st.spinner("🔍 Mencocokkan hierarki FAISS dan Metadata..."):
                rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
                
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                
                for idx, rec in enumerate(rekomendasi):
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.markdown(f"**Tingkat Akurasi Final:** `{rec['score']:.4f}`")
                        st.caption(f"*(Skor Vektor Asli: {rec['faiss_score']:.4f})*")
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan pada level Tersier/Kuartier.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada sistem: {e}")
