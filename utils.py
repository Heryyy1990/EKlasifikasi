import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

@st.cache_resource
def load_system():
    # Load Model Lokal
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Load Vektor FAISS
    index = faiss.read_index('vector_sikap_minilm.faiss')
    # Load Metadata (Pickle)
    df = pd.read_pickle('metadata_sikap.pkl')
    
    # Buat kamus (dictionary) kode -> uraian untuk melacak hierarki
    kode_dict = dict(zip(df['kode'].astype(str), df['uraian']))
    
    return model, index, df, kode_dict

def extract_intent(client, input_text, prompt_template):
    prompt = prompt_template.format(input_text=input_text)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    # Bersihkan format markdown json
    clean_text = response.text.strip().replace("```json", "").replace("
```", "")
    
    try:
        return json.loads(clean_text)
    except Exception as e:
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
        
    return " ➔ ".join(hierarchy)

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
        
        # Buang kode Primer (1) dan Sekunder (2)
        if level < 3:
            continue
        
        faiss_score = float(distances[0][i])
        
        domain_row = str(row.get('domain', '')).lower()
        activity_row = str(row.get('activity', '')).lower()
        
        domain_match = 1.0 if domain_prediksi and domain_prediksi in domain_row else 0.0
        activity_match = 1.0 if activity_prediksi and activity_prediksi in activity_row else 0.0
        
        # Rumus Scoring SIKAP
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
