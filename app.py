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

EXTRACTION_PROMPT = """Anda adalah pakar tata naskah dinas tingkat nasional.
Surat dari user sering kali rancu. Tugas Anda adalah menerjemahkannya ke dalam "Bahasa Mesin" yang optimal untuk pencarian database arsip.

ATURAN MUTLAK:
1. Buang kata: surat, permohonan, laporan, penyampaian.
2. Ubah nama tempat/objek (perpustakaan, mobil, jalan, sekolah) menjadi istilah birokrasi baku (aset, barang milik daerah, infrastruktur).
3. Output HARUS berformat JSON dengan 2 kunci: "query_kalimat" dan "keywords".

Contoh Input: "permohonan surat sertifikat tanah untuk pembangunan perpustakaan"
Output JSON:
{{
  "query_kalimat": "pengurusan administrasi pertanahan dan legalitas aset barang milik daerah",
  "keywords": ["pertanahan", "aset", "tanah", "barang milik daerah"]
}}

Input Surat: "{input_text}"
HANYA KELUARKAN JSON VALID.
"""

# ==========================================
# 2. FUNGSI SISTEM (TRUE HYBRID SEARCH)
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
            "keywords": []
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

def search_classification(model, index, df, kode_dict, intent_json, top_k=50):
    query_text = intent_json.get('query_kalimat', '')
    keywords = intent_json.get('keywords', [])
    
    query_vector = model.encode([query_text], normalize_embeddings=True)
    # Ambil 50 kandidat untuk jaring yang lebih luas
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
            
        faiss_score = float(distances[0][i])
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        # =======================================================
        # LEXICAL BOOST (SUNTIKAN POIN JIKA KATA KUNCI COCOK)
        # =======================================================
        teks_target = (str(row['uraian']) + " " + hierarchy_str).lower()
        bonus_score = 0.0
        matched_kws = []
        
        for kw in keywords:
            if kw.lower() in teks_target:
                bonus_score += 0.08  # Suntikan nilai agar naik peringkat
                matched_kws.append(kw.lower())
                
        final_score = faiss_score + bonus_score
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': final_score,
            'faiss_score': faiss_score,
            'bonus': bonus_score,
            'matched': ", ".join(matched_kws),
            'hierarchy': hierarchy_str
        })
        
    # Saring Level 4 (Kuartier) dan Level 3 (Tersier)
    level_4_results = [r for r in results if r['level'] >= 4]
    level_3_results = [r for r in results if r['level'] == 3]
    
    level_4_results = sorted(level_4_results, key=lambda x: x['score'], reverse=True)
    final_results = level_4_results[:3]
    
    # Fallback ke Tersier jika Kuartier kurang dari 3
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
st.write("Masukkan perihal/uraian surat. AI akan membersihkan kalimat dan mencari klasifikasi arsip yang tepat.")

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
            with st.spinner("🤖 Menganalisis konteks tata naskah..."):
                intent_json = extract_intent(client, user_input, EXTRACTION_PROMPT)
            
            st.success(f"**Kalimat Semantik:** {intent_json.get('query_kalimat', 'N/A')}")
            st.info(f"**Kata Kunci (Boost):** {', '.join(intent_json.get('keywords', []))}")
            
            with st.spinner("🔍 Memindai database klasifikasi..."):
                rekomendasi = search_classification(model, index, df, kode_dict, intent_json)
                
            if rekomendasi:
                st.markdown("---")
                st.markdown("### 🏆 3 Rekomendasi Teratas")
                for idx, rec in enumerate(rekomendasi):
                    with st.container():
                        st.markdown(f"#### {idx + 1}. Kode: **{rec['kode']}**")
                        st.markdown(f"**Uraian:** {rec['uraian']}")
                        st.markdown(f"**Tingkat Akurasi (Final):** `{rec['score']:.4f}`")
                        
                        if rec['bonus'] > 0:
                            st.caption(f"*(Mendapat Bonus Keyword: {rec['matched']} | FAISS Dasar: {rec['faiss_score']:.4f})*")
                            
                        st.info(f"**Jejak Hierarki:**\n\n{rec['hierarchy']}")
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan klasifikasi yang relevan pada level Tersier/Kuartier.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
