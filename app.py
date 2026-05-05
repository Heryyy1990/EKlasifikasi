import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT LOCKING (SATPAM)
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

# Ini adalah "Buku Saku" untuk Gemini agar bisa mengunci kode Sekunder dengan tepat
EXTRACTION_PROMPT = """Anda adalah Arsiparis Pengendali. Tugas Anda mengekstrak niat surat dan MENENTUKAN KODE SEKUNDER (Prefix) dari tata naskah dinas.

ATURAN KODE SEKUNDER (HAFALKAN!):
- Jika tentang Tanah Instansi / Sertifikat Tanah Pemda -> "500.17" (Pertanahan)
- Jika tentang Aset / Gedung / Barang Milik Daerah / Kendaraan -> "000.2" (Perlengkapan)
- Jika tentang SPPD / Perjalanan Dinas -> "000.1" (Ketatausahaan)
- Jika tentang Cuti / Izin Pegawai -> "800.1" (Kepegawaian)
- Jika tentang Anggaran / Pencairan Dana -> "900.1" (Keuangan)
- Jika di luar ini, kosongkan nilai "kode_sekunder_lock".

ATURAN EKSTRAKSI: Hapus kata pengantar (surat, permohonan) dan nama lokasi/objek spesifik.

Contoh Input: "permohonan surat sertifikat tanah untuk pembangunan perpustakaan"
Output JSON:
{{
  "query_kalimat": "penguatan hak atas tanah administrasi pertanahan",
  "kode_sekunder_lock": "500.17"
}}

Input Surat: "{input_text}"
HANYA KELUARKAN JSON VALID TANPA MARKDOWN.
"""

# ==========================================
# 2. FUNGSI SISTEM DENGAN FILTERING KETAT
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
            "query_kalimat": input_text,
            "kode_sekunder_lock": ""
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

def search_classification(model, index, df, kode_dict, intent_json, top_k=100):
    query_text = intent_json.get('query_kalimat', '')
    kode_lock = intent_json.get('kode_sekunder_lock', '')
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    # Kita ambil 100 teratas dari FAISS untuk disaring
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        level = len(kode.split('.'))
        
        # Wajib buang kode Primer (1) dan Sekunder (2)
        if level < 3:
            continue
            
        # LOGIKA LOCKING (PENGUNCIAN)
        # Jika Gemini memberikan kunci sekunder, buang semua kode yang tidak diawali kunci tersebut
        if kode_lock and not kode.startswith(kode_lock):
            continue
            
        faiss_score = float(distances[0][i])
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': faiss_score,
            'hierarchy': hierarchy_str
        })
        
    # Jika karena suatu hal hasil filter kosong (misal Gemini salah kunci), 
    # kita fallback (mundur) tanpa kunci
    if not results and kode_lock:
        intent_json['kode_sekunder_lock'] = ""
        return search_classification(model, index, df, kode_dict, intent_json, top_k=30)
        
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
st.write("Dilengkapi arsitektur AI Multi-Tahap (Intent Locking + Semantic Search)")

with st.spinner("Menyiapkan Sistem..."):
    model, index, df, kode_dict = load_system()

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("API Key Google Gemini belum di-set di Streamlit Secrets!")
    st.stop()

user_input = st.text_area("Uraian Surat:", placeholder="Contoh: permohonan sertifikat tanah...", height=120)

if st.button("Cari Kode Klasifikasi", type="primary"):
    if not user_input.strip():
        st.warning("Silakan ketik uraian surat terlebih dahulu.")
    else:
        try:
            with st.spinner("🤖 Gemini sedang mengunci rumpun klasifikasi..."):
                intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            kode_lock_display = intent_json.get('kode_sekunder_lock', 'Pencarian Bebas')
            if not kode_lock_display: kode_lock_display = "Pencarian Bebas"
            
            st.success(f"**Target Vektor:** {intent_json.get('query_kalimat', 'N/A')}")
            st.info(f"**Rumpun Terkunci (Prefix):** 🔒 `{kode_lock_display}`")
            
            with st.spinner("🔍 Memindai klaster yang dikunci..."):
                rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
                
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                for idx, rec in enumerate(rekomendasi):
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
