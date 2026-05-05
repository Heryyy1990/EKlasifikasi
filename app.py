import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT PENALARAN ARSIP
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

# PROMPT TANPA KAMUS MANUAL - MEMAKSA GEMINI BERPIKIR LOGIS
EXTRACTION_PROMPT = """Anda adalah pakar sistem klasifikasi arsip pemerintah (tata naskah dinas).
User akan memberikan uraian surat yang panjang dan ambigu. 

Tugas Anda adalah berpikir secara hierarkis:
1. Analisis subjek utama surat tersebut.
2. Tentukan KODE PRIMER (000-900) yang paling relevan berdasarkan fungsi pemerintahan.
3. Tentukan KODE SEKUNDER (Prefix dua tingkat, misal: 500.17 atau 000.2).

Panduan Berpikir:
- 000: Umum/Perlengkapan/Rumah Tangga/Kearsipan.
- 100: Pemerintahan/Otonomi.
- 400: Kesejahteraan Rakyat (Pendidikan/Kesehatan).
- 500: Perekonomian (Pertanian/Pertanahan/Hutan).
- 900: Keuangan.

Output harus JSON:
{{
  "inti_masalah": "ringkasan 3 kata tanpa kata permohonan/surat",
  "kode_lock": "KODE SEKUNDER HASIL ANALISIS ANDA"
}}

Input: "{input_text}"
HANYA JSON.
"""

# ==========================================
# 2. FUNGSI MESIN (LOCKED RETRIEVAL)
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
    clean_text = response.text.strip().replace("```json", "").replace("
```", "")
    try:
        return json.loads(clean_text)
    except:
        return {"inti_masalah": input_text, "kode_lock": ""}

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

def search_classification(model, index, df, kode_dict, intent_json, top_k=150):
    query_text = intent_json.get('inti_masalah', '')
    kode_lock = intent_json.get('kode_lock', '')
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        level = len(kode.split('.'))
        
        # 1. WAJIB LEVEL 3 (Tersier) ATAU 4 (Kuartier)
        if level < 3: continue
            
        # 2. FILTER KETAT: Kode harus diawali dengan hasil analisa Gemini (kode_lock)
        if kode_lock and not kode.startswith(kode_lock):
            continue
            
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': float(distances[0][i]),
            'hierarchy': build_hierarchy_string(kode, kode_dict)
        })
        
    # Urutkan berdasarkan level terdalam (Kuartier dulu baru Tersier)
    results = sorted(results, key=lambda x: (x['level'], x['score']), reverse=True)
    return results[:3]

# ==========================================
# 3. UI
# ==========================================
st.title("🗂️ SIKAP - Intelligent Mode")
st.write("Sistem Klasifikasi Otomatis dengan Penalaran Hierarki AI")

with st.spinner("Loading..."):
    model, index, df, kode_dict = load_system()

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.stop()

user_input = st.text_area("Input Uraian Surat:", height=100)

if st.button("Analisis Klasifikasi", type="primary"):
    with st.spinner("🤖 AI sedang membedah struktur klasifikasi..."):
        intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
    
    lock = intent_json.get('kode_lock', '')
    st.info(f"**Analisis Masalah:** {intent_json.get('inti_masalah')} | **Kode Terkunci:** `{lock}`")
    
    with st.spinner("🔍 Mencari kode tersier/kuartier..."):
        rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
        
    if rekomendasi:
        st.markdown("---")
        for idx, rec in enumerate(rekomendasi):
            st.subheader(f"{idx+1}. Kode: {rec['kode']}")
            st.write(f"**Uraian:** {rec['uraian']}")
            st.caption(f"**Hierarki:** {rec['hierarchy']}")
            st.markdown("---")
    else:
        st.warning("Gagal menemukan kode yang cocok dalam rumpun tersebut.")
