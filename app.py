import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT PENALARAN MURNI
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

# PROMPT INI MEMAKSA GEMINI BERPIKIR SECARA FUNGSIONAL PEMERINTAHAN
EXTRACTION_PROMPT = """Anda adalah analis sistem klasifikasi arsip pemerintah.
Tugas Anda adalah membedah fungsi organisasi dari uraian surat yang diberikan.

LOGIKA BERPIKIR:
1. Identifikasi SUBSTANSI masalah (apa yang sedang diurus?).
2. Identifikasi FUNGSI organisasi (siapa yang mengurus? apakah pengawasan, perencanaan, atau operasional?).
3. Tentukan KODE PREFIX (Sekunder) yang paling logis sesuai standar klasifikasi 000-900 (misal: 700.1 untuk pengawasan internal, 500.17 untuk pertanahan, dll).

Output HARUS dalam format JSON:
{{
  "analisis_masalah": "inti urusan birokrasi",
  "prefix_lock": "KODE PREFIX HASIL NALAR ANDA"
}}

Input: "{input_text}"
HANYA KELUARKAN JSON VALID TANPA MARKDOWN.
"""

# ==========================================
# 2. FUNGSI MESIN (SMART FILTERING)
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
    except:
        return {"analisis_masalah": input_text, "prefix_lock": ""}

def build_hierarchy_string(kode, kode_dict):
    parts = str(kode).split('.')
    hierarchy = []
    current_kode = ""
    for i, part in enumerate(parts):
        if i == 0: current_kode = part
        else: current_kode = f"{current_kode}.{part}"
        nama = kode_dict.get(current_kode, "Unknown")
        hierarchy.append(f"{current_kode} ({nama})")
    return " -> ".join(hierarchy)

def search_classification(model, index, df, kode_dict, intent_json, top_k=200):
    query_text = intent_json.get('analisis_masalah', '')
    prefix_lock = str(intent_json.get('prefix_lock', ''))
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        level = len(kode.split('.'))
        
        # 1. HANYA AMBIL KODE TERSIER/KUARTIER
        if level < 3: continue
            
        # 2. FILTER KETAT BERDASARKAN HASIL NALAR GEMINI
        if prefix_lock and not kode.startswith(prefix_lock):
            continue
            
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': float(distances[0][i]),
            'hierarchy': build_hierarchy_string(kode, kode_dict)
        })
        
    # Urutkan berdasarkan skor tertinggi
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results[:3]

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🗂️ SIKAP - Logic Engine")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar (Muna Barat)")

with st.spinner("Mengaktifkan Nalar AI..."):
    model, index, df, kode_dict = load_system()

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("API Key Bermasalah!")
    st.stop()

user_input = st.text_area("Input Perihal Surat:", placeholder="Misal: Laporan hasil evaluasi inspektorat...", height=100)

if st.button("Bedah Klasifikasi", type="primary"):
    with st.spinner("🤖 AI sedang menalar rumpun klasifikasi..."):
        intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
    
    lock = intent_json.get('prefix_lock', 'Bebas')
    st.info(f"**Analisis Masalah:** {intent_json.get('analisis_masalah')} | **Prefix Terkunci:** 🔒 `{lock}`")
    
    with st.spinner("🔍 Mencari kode detail dalam rumpun tersebut..."):
        rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
        
    if rekomendasi:
        st.markdown("---")
        for idx, rec in enumerate(rekomendasi):
            with st.container():
                st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                st.write(f"**Uraian:** {rec['uraian']}")
                st.info(f"**Jejak Hierarki:**\n{rec['hierarchy']}")
                st.markdown("---")
    else:
        st.warning("Tidak ditemukan kode detail dalam rumpun tersebut. AI mungkin salah menentukan prefix, silakan coba lagi.")
