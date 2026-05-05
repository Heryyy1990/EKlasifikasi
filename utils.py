import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

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

# 2. EKSTRAKSI GEMINI
def extract_intent(client, input_text, prompt_template):
    prompt = prompt_template.format(input_text=input_text)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text.strip().lower()

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

# 4. PENCARIAN FAISS DAN LOGIKA FILTERING
def search_classification(model, index, df, kode_dict, query, top_k=30):
    # Ubah inti surat jadi vektor
    query_vector = model.encode([query], normalize_embeddings=True)
    
    # Cari kandidat di FAISS (Ambil 30 kandidat dulu untuk difilter)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        row = df.iloc[idx]
        kode = str(row['kode'])
        # Hitung jumlah titik untuk tahu level (000 = 1, 000.1 = 2, 000.1.1 = 3, dll)
        level = len(kode.split('.'))
        
        # ATURAN MUTLAK: Buang kode Primer (1) dan Sekunder (2)
        if level < 3:
            continue
            
        # Bangun jejak hierarkinya
        hierarchy_str = build_hierarchy_string(kode, kode_dict)
        
        results.append({
            'kode': kode,
            'uraian': row['uraian'],
            'level': level,
            'score': float(distances[0][i]),
            'hierarchy': hierarchy_str
        })
        
    # Pisahkan Level 4 (Kuartier) dan Level 3 (Tersier)
    level_4_results = [r for r in results if r['level'] >= 4]
    level_3_results = [r for r in results if r['level'] == 3]
    
    # Prioritaskan Kuartier, ambil maksimal 3
    final_results = level_4_results[:3]
    
    # FALLBACK: Jika Kuartier kurang dari 3, tambal pakai Tersier
    if len(final_results) < 3:
        needed = 3 - len(final_results)
        final_results.extend(level_3_results[:needed])
        
    # Urutkan berdasarkan skor kemiripan tertinggi
    final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    
    return final_results[:3]
