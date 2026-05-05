import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# 1. CACHING AGAR HEMAT RAM 1GB
@st.cache_resource
def load_system():
    # Load Model Lokal
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Load Vektor FAISS
    index = faiss.read_index('vector_sikap_minilm.faiss')
    # Load Metadata (Pickle)
    df = pd.read_pickle('metadata_sikap.pkl')
    
    # Buat kamus (dictionary) kode -> uraian untuk melacak hierarki (Penting!)
    kode_dict = dict(zip(df['kode'].astype(str), df['uraian']))
    
    return model, index, df, kode_dict

# 2. EKSTRAKSI GEMINI (VERSI JSON)
def extract_intent(client, input_text, prompt_template):
    prompt = prompt_template.format(input_text=input_text)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    # Bersihkan format markdown json (jika ada)
    clean_text = response.text.strip().replace("```json", "").replace("
```", "")
    
    try:
        return json.loads(clean_text)
    except Exception as e:
        # Fallback darurat jika Gemini gagal membuat JSON
        return {
            "intent_query": clean_text[:100], 
            "domain_prediksi": "", 
            "activity_prediksi": ""
        }

# 3. FUNGSI MELACAK HIERARKI (Primer -> Sekunder -> Tersier -> Kuartier)
def build_hierarchy_string(kode, kode_dict):
    parts = str(kode).split('.')
    hierarchy = []
    current_kode = ""
    
    for i, part in enumerate(parts):
        if i == 0:
            current_kode = part
        else:
            current_kode = f"{current_kode}.{part}"
            
        # Ambil uraian/nama induk dari kamus
        nama = kode_dict.get(current_kode, "Unknown")
        hierarchy.append(f"{current_kode} ({nama})")
        
    return " ➔ ".join(hierarchy)

# 4. PENCARIAN FAISS DAN LOGIKA FILTERING (SCORING HYBRID)
def search_classification(model, index, df, kode_dict, intent_json, top_k=30):
    # Ekstrak dari JSON
    query_text = intent_json.get('intent_query', '')
    domain_prediksi = str(intent_json.get('domain_prediksi', '')).lower()
    activity_prediksi = str(intent_json.get('activity_prediksi', '')).lower()
    
    # Ubah inti surat jadi vektor
    query_vector = model.encode([query_text], normalize_embeddings=True)
    
    # Cari kandidat di FAISS (Ambil 30 kandidat dulu untuk difilter)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        
        # Hitung jumlah titik untuk tahu level
        level = len(kode.split('.'))
        
        # ATURAN MUTLAK: Buang kode Primer (1) dan Sekunder (2)
        if level < 3:
            continue
        
        # --- MULAI LOGIKA SCORING ---
        faiss_score = float(distances[0][i])
        
        domain_row = str(row.get('domain', '')).lower()
        activity_row = str(row.get('activity', '')).lower()
        
        # Cek kecocokan (Match)
        domain_match = 1.0 if domain_prediksi and domain_prediksi in domain_row else 0.0
        activity_match = 1.0 if activity_prediksi and activity_prediksi in activity_row else 0.0
        
        # RUMUS MAHAKARYA ANDA
        final_score = (faiss_score * 0.7) + (domain_match * 0.2) + (activity_match * 0.1)
        # ----------------------------
        
        # Bangun jejak hierarkinya
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': final_score, # Gunakan skor final
            'faiss_score': faiss_score, # Simpan skor asli jika ingin dilihat
            'hierarchy': hierarchy_str
        })
        
    # Pisahkan Level 4 (Kuartier) dan Level 3 (Tersier)
    level_4_results = [r for r in results if r['level'] >= 4]
    level_3_results = [r for r in results if r['level'] == 3]
    
    # Urutkan Level 4 berdasarkan skor tertinggi
    level_4_results = sorted(level_4_results, key=lambda x: x['score'], reverse=True)
    
    # Prioritaskan Kuartier, ambil maksimal 3
    final_results = level_4_results[:3]
    
    # FALLBACK: Jika Kuartier kurang dari 3, tambal pakai Tersier
    if len(final_results) < 3:
        needed = 3 - len(final_results)
        # Urutkan Level 3 berdasarkan skor tertinggi
        level_3_results = sorted(level_3_results, key=lambda x: x['score'], reverse=True)
        final_results.extend(level_3_results[:needed])
        
    # Urutkan ulang campuran hasil akhirnya
    final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    
    return final_results[:3]
