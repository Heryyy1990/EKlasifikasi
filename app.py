import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

EXTRACTION_PROMPT = """Anda adalah Arsiparis Senior Pemerintah Daerah.
Tugas Anda menganalisis perihal surat dan mengekstraknya ke dalam format JSON untuk sistem Vector Database.

ATURAN MUTLAK:
1. "intent_query": Hasilkan 3-6 kata kunci PENCARIAN. Abaikan kata (surat, permohonan). Jika menyangkut gedung/fasilitas instansi, gunakan frasa "barang milik daerah" atau "aset".
2. "domain_prediksi": Prediksi domain utama (Pilih salah satu: umum, pemerintahan, politik, keamanan, kesejahteraan rakyat, perekonomian, pekerjaan umum, pengawasan, kepegawaian, keuangan).
3. "activity_prediksi": Prediksi sub-kategorinya (misal: perlengkapan, pertanahan, ketatausahaan).

Input Surat: "{input_text}"

HANYA KELUARKAN JSON VALID TANPA MARKDOWN.
Contoh format:
{{
  "intent_query": "sertifikasi legalitas tanah barang milik daerah",
  "domain_prediksi": "umum",
  "activity_prediksi": "perlengkapan"
}}
"""

# ==========================================
# 2. FUNGSI SISTEM (MESIN SIKAP)
# ==========================================
@st.cache_resource
def load_system():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    index = faiss.read_index('vector_sikap_minilm.faiss')
    df = pd.read_pickle('metadata_sikap.pkl')
    kode_dict = dict(zip(df['kode'].astype(str), df['uraian']))
    return model, index, df, kode_dict

def extract_intent(client, input_text, prompt_template):
    prompt = prompt_template.format(input_text=input_text)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    clean_text = response.text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except Exception:
        return {
            "intent_query": clean_text[:100], 
            "domain_prediksi": "", 
            "activity_prediksi": ""
        }

def build_hierarchy_string(kode, kode_dict):
    parts = str(kode).split('.')
    hierarchy = []
    current_kode = ""
    for i, part in enumerate(parts):
        if i == 0:
            current_kode = part
        else:
            current_kode = f"{current_kode}.{part}"
        nama = kode_dict.get(current_kode, "Unknown")
        hierarchy.append(f"{current_kode} ({nama})")
    return " -> ".join(hierarchy)

def search_classification(model, index, df, kode_dict, intent_json, top_k=30):
    query_text = intent_json.get('intent_query', '')
    domain_prediksi = str(intent_json.get('domain_prediksi', '')).lower()
    activity_prediksi = str(intent_json.get('activity_prediksi', '')).lower()
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        level = len(kode.split('.'))
        
        if level < 3:
            continue
            
        faiss_score = float(distances[0][i])
        domain_row = str(row.get('domain', '')).lower()
        activity_row = str(row.get('activity', '')).lower()
        
        domain_match = 1.0 if domain_prediksi and domain_prediksi in domain_row else 0.0
        activity_match = 1.0 if activity_prediksi and activity_prediksi in activity_row else 0.0
        
        final_score = (faiss_score * 0.7) + (domain_match * 0.2) + (activity_match * 0.1)
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': final_score,
            'faiss_score': faiss_score,
            'hierarchy': hierarchy_str
        })
        
    level_4_results = [r for r in results if r['level'] >= 4]
    level_3_results = [r for r in results if r['level'] == 3]
    
    level_4_results = sorted(level_4_results, key=lambda x: x['score'], reverse=True)
    final_results = level_4_results[:3]
    
    if len(final_results) < 3:
        needed = 3 - len(final_results)
        level_3_results = sorted(level_3_results, key=lambda x: x['score'], reverse=True)
        final_results.extend(level_3_results[:needed])
        
    final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    return final_results[:3]

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🗂️ SIKAP")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar")
st.write("Masukkan perihal/uraian surat. AI akan mencari rekomendasi klasifikasi paling presisi.")

with st.spinner("Menyiapkan Sistem..."):
    model, index, df, kode_dict = load_system()

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("API Key Google Gemini belum di-set di Streamlit Secrets!")
    st.stop()

user_input = st.text_area("Uraian Surat:", placeholder="Ketik perihal surat di sini...", height=120)

if st.button("Cari Kode Klasifikasi", type="primary"):
    if not user_input.strip():
        st.warning("Silakan ketik uraian surat terlebih dahulu.")
    else:
        try:
            with st.spinner("🤖 Menganalisis konteks surat..."):
                intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            st.success(f"**Vektor Kata Kunci:** {intent_json.get('intent_query', 'N/A')}")
            st.info(f"**Prediksi Kategori:** {intent_json.get('domain_prediksi', 'N/A')} -> {intent_json.get('activity_prediksi', 'N/A')}")
            
            with st.spinner("🔍 Mencocokkan data..."):
                rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
                
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                for idx, rec in enumerate(rekomendasi):
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.markdown(f"**Tingkat Akurasi Final:** `{rec['score']:.4f}`")
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan pada level Tersier/Kuartier.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
