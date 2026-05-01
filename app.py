# app.py
# ==================================================
# SIKAP FINAL v3
# Google GenAI NEW SDK + Gemini 2.5 Flash
# FAISS + Metadata Filter + Cross Encoder Rerank
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import json
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

# ==================================================
# 1. PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="SIKAP",
    page_icon="📁",
    layout="wide"
)

st.title("📁 SIKAP")
st.caption("Sistem Identifikasi Klasifikasi Arsip Pintar")

# ==================================================
# 2. LOAD API KEY
# Streamlit Secrets:
# GOOGLE_API_KEY="xxxxx"
# ==================================================
client = genai.Client(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

# ==================================================
# 3. LOAD ALL FILES
# ==================================================
@st.cache_resource
def load_all():

    # embedding model
    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # reranker
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # faiss
    index_q = faiss.read_index("faiss_kuartier.index")
    index_t = faiss.read_index("faiss_tersier.index")

    # metadata
    with open("meta_kuartier.pkl", "rb") as f:
        meta_q = pickle.load(f)

    with open("meta_tersier.pkl", "rb") as f:
        meta_t = pickle.load(f)

    # csv master
    df = pd.read_csv("klasifikasi_sikap_final_v2.csv")

    return embed_model, reranker, index_q, index_t, meta_q, meta_t, df


embed_model, reranker, index_q, index_t, meta_q, meta_t, df = load_all()

# ==================================================
# 4. BUILD PRIMER & SEKUNDER MAP
# ==================================================
primer_map = {}
sekunder_map = {}

for _, row in df.iterrows():

    kode = str(row["kode"])
    uraian = str(row["uraian"])

    # primer
    if "." not in kode:
        primer_map[kode] = uraian

    # sekunder
    if kode.count(".") == 1:
        sekunder_map[kode] = uraian

# ==================================================
# 5. GEMINI ROUTER
# ==================================================
def route_query(user_text):

    prompt = f"""
    Anda adalah analis klasifikasi arsip pemerintah.

    Analisis isi surat berikut.

    Tentukan:
    1. domain utama
    2. kata kunci penting
    3. kemungkinan kode primer (contoh: 000,100,500,900)

    Surat:
    {user_text}

    Jawab hanya JSON valid:

    {{
      "domain":"...",
      "keywords":["...","..."],
      "primer":"..."
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ==================================================
# 6. PARSE JSON GEMINI
# ==================================================
def parse_router(raw):

    try:
        raw = raw.strip()

        raw = raw.replace("```json", "")
        raw = raw.replace("```", "")

        data = json.loads(raw)

        return data

    except:
        return {
            "domain": "umum",
            "keywords": [],
            "primer": "000"
        }

# ==================================================
# 7. SEARCH FAISS
# ==================================================
def search_faiss(query, index, topk=20):

    vec = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(vec)

    D, I = index.search(vec, topk)

    return D[0], I[0]

# ==================================================
# 8. FILTER BY PRIMER
# ==================================================
def filter_candidates(indices, meta, primer):

    hasil = []

    for idx in indices:

        row = meta[idx]
        kode = str(row["kode"])

        if kode.startswith(primer):
            hasil.append(row)

    return hasil

# ==================================================
# 9. RERANK
# ==================================================
def rerank(query, candidates):

    if len(candidates) == 0:
        return []

    pairs = []

    for c in candidates:
        pairs.append(
            [query, c["embedding_text"]]
        )

    scores = reranker.predict(pairs)

    for i in range(len(candidates)):
        candidates[i]["score"] = float(scores[i])

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[:3]

# ==================================================
# 10. FALLBACK SEARCH TERSIER
# ==================================================
def fallback_tersier(query):

    D, I = search_faiss(query, index_t, topk=10)

    cands = []

    for idx in I:
        cands.append(meta_t[idx])

    final = rerank(query, cands)

    return final

# ==================================================
# 11. UI INPUT
# ==================================================
user_text = st.text_area(
    "Masukkan uraian surat / perihal surat",
    height=180
)

# ==================================================
# 12. BUTTON PROCESS
# ==================================================
if st.button("Cari Klasifikasi"):

    if user_text.strip() == "":
        st.warning("Silakan isi uraian surat terlebih dahulu.")
        st.stop()

    # ----------------------------------------------
    # A. Gemini Router
    # ----------------------------------------------
    with st.spinner("Gemini menganalisis isi surat..."):

        raw = route_query(user_text)
        info = parse_router(raw)

    primer = str(info["primer"])

    st.subheader("🧠 Hasil Analisis Gemini")

    st.write("Domain :", info["domain"])
    st.write("Primer :", primer, "-", primer_map.get(primer, ""))
    st.write("Keywords :", ", ".join(info["keywords"]))

    # ----------------------------------------------
    # B. Search Kuartier
    # ----------------------------------------------
    with st.spinner("Mencari kandidat kuartier..."):

        D, I = search_faiss(user_text, index_q, topk=20)

        candidates = filter_candidates(
            I, meta_q, primer
        )

        final = rerank(user_text, candidates)

    # ----------------------------------------------
    # C. Confidence Check
    # ----------------------------------------------
    use_fallback = False

    if len(final) == 0:
        use_fallback = True

    elif final[0]["score"] < 0.45:
        use_fallback = True

    # ----------------------------------------------
    # D. Fallback Tersier
    # ----------------------------------------------
    if use_fallback:

        st.info(
            "Confidence kuartier rendah. "
            "Menggunakan fallback tersier..."
        )

        final = fallback_tersier(user_text)

    # ----------------------------------------------
    # E. Output
    # ----------------------------------------------
    st.subheader("🎯 Top 3 Rekomendasi")

    for i, row in enumerate(final, start=1):

        st.markdown(
            f"""
### {i}. {row['kode']}

**Uraian:** {row['uraian']}  
**Level:** {row['level']}  
**Score:** {round(row['score'],4)}
"""
        )
