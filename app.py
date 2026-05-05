import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import json

# ==========================================
# 1. KONFIGURASI & PROMPT ULTRA-KEJAM
# ==========================================
st.set_page_config(page_title="SIKAP App", page_icon="🗂️", layout="centered")

EXTRACTION_PROMPT = """Anda adalah Arsiparis Senior yang sangat galak dan presisi.
Tugas Anda menerjemahkan perihal surat dari user menjadi KATA KUNCI BIROKRASI MURNI.

ATURAN MUTLAK (JIKA DILANGGAR ANDA GAGAL):
1. HAPUS SEMUA kata pengantar (surat, permohonan, laporan, penyampaian).
2. HAPUS SEMUA nama lokasi/objek spesifik (contoh: perpustakaan, sekolah, puskesmas, mobil, jalan, dprd, bupati).
3. HANYA tinggalkan TINDAKAN ADMINISTRATIF atau SUBJEK BIROKRASI (contoh: sertifikat tanah, aset daerah, pengadaan barang, perjalanan dinas).
4. Output MAKSIMAL 2-4 kata saja. Tidak boleh lebih!

Contoh 1:
Input: "permohonan surat sertifikat tanah untuk pembangunan perpustakaan"
Output: "sertifikat tanah aset" (Kata 'perpustakaan' WAJIB hilang total!)

Contoh 2:
Input: "Laporan kerusakan mobil dinas bupati"
Output: "pemeliharaan kendaraan dinas"

Input Surat: "{input_text}"

HANYA KELUARKAN JSON VALID.
{{
  "intent_query": "kata kunci birokrasi murni"
}}
"""

# ==========================================
# 2. FUNGSI SISTEM (MURNI SEMANTIC SEARCH)
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
            "intent_query": clean_text[:50]
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
    # Hanya fokus pada query murni, lupakan domain/activity yang tidak akurat
    query_text = intent_json.get('intent_query', '')
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        level = len(kode.split('.'))
        
        # Buang kode Primer (1) dan Sekunder (2)
        if level < 3:
            continue
            
        # Gunakan 100% skor kemiripan Vektor dari FAISS (Jauh lebih akurat)
        score = float(distances[0][i])
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': score,
            'hierarchy': hierarchy_str
        })
        
    level_4_results = [r for r in results if r['level'] >= 4]
    level_3_results = [r for r in results if r['level'] == 3]
    
    level_4_results = sorted(level_4_results, key=lambda x: x['score'], reverse=True)
    final_results = level_4_results[:3]
    
    # Fallback ke Tersier
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
st.write("Masukkan uraian surat. AI akan membersihkan kalimat ambigu dan mencari klasifikasi arsip yang tepat.")

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
            with st.spinner("🤖 Mengisolasi kata kunci birokrasi..."):
                intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            st.success(f"**Vektor Kata Kunci:** {intent_json.get('intent_query', 'N/A')}")
            
            with st.spinner("🔍 Memindai database klasifikasi..."):
                rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
                
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                for idx, rec in enumerate(rekomendasi):
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.markdown(f"**Tingkat Kecocokan:** `{rec['score']:.4f}`")
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan pada level Tersier/Kuartier.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
