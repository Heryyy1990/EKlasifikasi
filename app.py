import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

# ==========================================
# 2. MESIN SISTEM (LOAD DATA & GENERATE MENU)
# ==========================================
@st.cache_resource
def load_system():
    # Load Model & FAISS
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    index = faiss.read_index('vector_sikap_minilm.faiss')
    
    # Load Metadata Asli Anda
    df = pd.read_pickle('metadata_sikap.pkl')
    
    # Ambil Daftar Sekunder (Kode dengan 1 titik, misal 500.17) untuk "Menu" Gemini
    # Ini memastikan Gemini hanya memilih kode yang BENAR-BENAR ADA di metadata Anda
    df['kode_str'] = df['kode'].astype(str)
    df_sekunder = df[df['kode_str'].str.count('\.') == 1][['kode_str', 'uraian']].drop_duplicates()
    menu_sekunder = df_sekunder.apply(lambda x: f"{x['kode_str']} ({x['uraian']})", axis=1).tolist()
    
    kode_dict = dict(zip(df['kode_str'], df['uraian']))
    
    return model, index, df, kode_dict, menu_sekunder

def extract_intent_with_context(client, input_text, menu_sekunder):
    # Kirim daftar kode asli Anda ke Gemini sebagai referensi utama
    menu_text = "\n".join(menu_sekunder[:100]) # Ambil 100 pertama agar tidak kepanjangan
    
    prompt = f"""Anda adalah asisten arsiparis profesional.
Diberikan uraian surat, tugas Anda adalah memilih satu KODE SEKUNDER yang paling relevan dari daftar resmi metadata kami.

DAFTAR KODE SEKUNDER RESMI:
{menu_text}

TUGAS:
1. Analisis inti urusan dari surat user.
2. Pilih satu KODE (hanya angka kodenya saja) dari daftar di atas yang paling cocok.
3. Jika surat tentang Sertifikat Tanah/Pertanahan, pastikan pilih rumpun Pertanahan (500.17).

Input User: "{input_text}"

HANYA KELUARKAN JSON VALID:
{{
  "inti_urusan": "ringkasan urusan",
  "kode_lock": "NOMOR KODE YANG DIPILIH"
}}
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    clean_text = response.text.strip().replace("```json", "").replace("
```", "")
    try:
        return json.loads(clean_text)
    except:
        return {"inti_urusan": input_text, "kode_lock": ""}

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
    query_text = intent_json.get('inti_urusan', '')
    kode_lock = str(intent_json.get('kode_lock', ''))
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode_str'])
        
        # Filter: Hanya ambil level detail (3 atau 4)
        if kode.count('.') < 2: continue
            
        # FILTER KUNCI: Harus diawali kode sekunder yang dipilih Gemini
        if kode_lock and not kode.startswith(kode_lock):
            continue
            
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'score': float(distances[0][i]),
            'hierarchy': build_hierarchy_string(kode, kode_dict)
        })
        
    return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🗂️ SIKAP - Metadata-Driven Mode")
st.write("Sistem yang benar-benar membaca metadata Anda untuk klasifikasi presisi.")

with st.spinner("Membaca Metadata & Menyiapkan Vektor..."):
    model, index, df, kode_dict, menu_sekunder = load_system()

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("API Key Bermasalah!")
    st.stop()

user_input = st.text_area("Input Uraian Surat:", placeholder="Misal: Sertifikat tanah pembangunan perpustakaan...", height=100)

if st.button("Analisis Klasifikasi", type="primary"):
    with st.spinner("🤖 Gemini sedang mencocokkan dengan daftar Sekunder Anda..."):
        intent_json = extract_intent_with_context(client, user_input, menu_sekunder)
    
    lock = intent_json.get('kode_lock', '')
    st.info(f"**Inti Urusan:** {intent_json.get('inti_urusan')} | **Rumpun Terkunci:** 🔒 `{lock}`")
    
    with st.spinner("🔍 Mencari kode detail di dalam rumpun tersebut..."):
        rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
        
    if rekomendasi:
        st.markdown("---")
        for idx, rec in enumerate(rekomendasi):
            st.subheader(f"{idx+1}. Kode: {rec['kode']}")
            st.write(f"**Uraian:** {rec['uraian']}")
            st.caption(f"**Jejak Hierarki:** {rec['hierarchy']}")
            st.markdown("---")
    else:
        st.warning("Tidak ditemukan kode detail. Cobalah untuk memperjelas uraian surat.")
