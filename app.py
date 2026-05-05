import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT ANALISIS KONTEKS
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

EXTRACTION_PROMPT = """Anda adalah analis urusan pemerintahan.
Tugas Anda menentukan "Kategori Besar" urusan dari sebuah surat.

KATEGORI YANG TERSEDIA:
- Perlengkapan / Aset (Jika terkait gedung, tanah instansi, kendaraan, barang)
- Pertanahan (Jika terkait hak atas tanah, sertifikat tanah masyarakat/umum)
- Kepegawaian (Jika terkait cuti, mutasi, diklat, perjalanan dinas)
- Kearsipan / Tata Naskah (Jika terkait surat menyurat)
- Keuangan (Jika terkait anggaran, pajak, gaji)
- Umum (Jika urusan rutin lainnya)

Output harus JSON:
{{
  "inti_masalah": "tindakan birokrasi singkat",
  "kategori_besar": "Pilih salah satu kategori di atas"
}}

Input: "{input_text}"
HANYA JSON.
"""

# ==========================================
# 2. FUNGSI MESIN (SMART MAPPING)
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
        return {"inti_masalah": input_text, "kategori_besar": "Umum"}

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
    query_text = intent_json.get('inti_masalah', '')
    kategori = intent_json.get('kategori_besar', '').lower()
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        uraian = str(row['uraian']).lower()
        level = len(kode.split('.'))
        
        # Ambil hierarki lengkap untuk pengecekan rumpun
        hierarchy_str = build_hierarchy_string(kode, kode_dict).lower()
        
        if level < 3: continue
            
        # Pengecekan Rumpun (Smart Filter)
        # Jika kategori adalah "Pertanahan", pastikan kata "pertanahan" atau "tanah" ada di hierarkinya
        is_match = False
        if "umum" in kategori: is_match = True # Umum boleh masuk mana saja
        elif "pertanahan" in kategori and ("pertanahan" in hierarchy_str or "tanah" in hierarchy_str): is_match = True
        elif "perlengkapan" in kategori and ("perlengkapan" in hierarchy_str or "aset" in hierarchy_str or "barang milik" in hierarchy_str): is_match = True
        elif "kepegawaian" in kategori and ("kepegawaian" in hierarchy_str or "pegawai" in hierarchy_str): is_match = True
        elif "keuangan" in kategori and ("keuangan" in hierarchy_str or "anggaran" in hierarchy_str): is_match = True
        elif "kearsipan" in kategori and ("kearsipan" in hierarchy_str or "naskah" in hierarchy_str): is_match = True
        
        if not is_match: continue
            
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': float(distances[0][i]),
            'hierarchy': build_hierarchy_string(kode, kode_dict)
        })
        
    # Urutkan berdasarkan skor kemiripan tertinggi
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results[:3]

# ==========================================
# 3. UI
# ==========================================
st.title("🗂️ SIKAP - Smart Context Mode")

with st.spinner("Loading..."):
    model, index, df, kode_dict = load_system()

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.stop()

user_input = st.text_area("Input Uraian Surat:", height=100)

if st.button("Cari Klasifikasi", type="primary"):
    with st.spinner("🤖 Menganalisis urusan..."):
        intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
    
    st.info(f"**Inti Masalah:** {intent_json.get('inti_masalah')} | **Rumpun Terdeteksi:** `{intent_json.get('kategori_besar')}`")
    
    with st.spinner("🔍 Memfilter database sesuai rumpun..."):
        rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
        
    if rekomendasi:
        st.markdown("---")
        for idx, rec in enumerate(rekomendasi):
            st.subheader(f"{idx+1}. Kode: {rec['kode']}")
            st.write(f"**Uraian:** {rec['uraian']}")
            st.caption(f"**Hierarki:** {rec['hierarchy']}")
            st.markdown("---")
    else:
        st.warning("Tidak ditemukan kode yang cocok dengan rumpun tersebut. Silakan coba uraian lain.")
