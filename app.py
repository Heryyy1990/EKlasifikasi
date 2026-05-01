import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="SIKAP",
    page_icon="📁",
    layout="wide"
)

st.title("📁 SIKAP")
st.caption("Sistem Identifikasi Klasifikasi Arsip Pintar")

# =====================================================
# API KEY
# =====================================================
client = genai.Client(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

# =====================================================
# LOAD FILES
# =====================================================
@st.cache_resource
def load_all():

    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # metadata
    with open("meta_kuartier.pkl", "rb") as f:
        meta_q = pickle.load(f)

    with open("meta_tersier.pkl", "rb") as f:
        meta_t = pickle.load(f)

    # csv
    df = pd.read_csv("klasifikasi_sikap_final_v2.csv")

    # precompute embeddings
    emb_q = embed_model.encode(
        [x["embedding_text"] for x in meta_q],
        convert_to_numpy=True
    )

    emb_t = embed_model.encode(
        [x["embedding_text"] for x in meta_t],
        convert_to_numpy=True
    )

    return embed_model, reranker, meta_q, meta_t, emb_q, emb_t, df


embed_model, reranker, meta_q, meta_t, emb_q, emb_t, df = load_all()

# =====================================================
# BUILD MAP
# =====================================================
primer_map = {}
sekunder_map = {}

for _, row in df.iterrows():

    kode = str(row["kode"])
    uraian = str(row["uraian"])

    if "." not in kode:
        primer_map[kode] = uraian

    if kode.count(".") == 1:
        sekunder_map[kode] = uraian

# =====================================================
# GEMINI ROUTER
# =====================================================
def route_query(text):

    prompt = f"""
    Analisis surat berikut.

    Tentukan:
    1. domain utama
    2. kata kunci
    3. kemungkinan kode primer

    Surat:
    {text}

    Jawab JSON:

    {{
      "domain":"...",
      "keywords":["..."],
      "primer":"..."
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def parse_json(raw):

    try:
        raw = raw.replace("```json", "").replace("```", "")
        return json.loads(raw)

    except:
        return {
            "domain": "umum",
            "keywords": [],
            "primer": "000"
        }

# =====================================================
# SEARCH ENGINE
# =====================================================
def semantic_search(query, emb_matrix, meta, topk=20):

    q = embed_model.encode([query], convert_to_numpy=True)

    scores = cosine_similarity(q, emb_matrix)[0]

    idx = np.argsort(scores)[::-1][:topk]

    results = []

    for i in idx:
        row = meta[i].copy()
        row["sim_score"] = float(scores[i])
        results.append(row)

    return results

# =====================================================
# FILTER PRIMER
# =====================================================
def filter_primer(rows, primer):

    return [
        r for r in rows
        if str(r["kode"]).startswith(primer)
    ]

# =====================================================
# RERANK
# =====================================================
def rerank(query, rows):

    if len(rows) == 0:
        return []

    pairs = [
        [query, r["embedding_text"]]
        for r in rows
    ]

    scores = reranker.predict(pairs)

    for i in range(len(rows)):
        rows[i]["score"] = float(scores[i])

    rows = sorted(
        rows,
        key=lambda x: x["score"],
        reverse=True
    )

    return rows[:3]

# =====================================================
# UI
# =====================================================
user_text = st.text_area(
    "Masukkan uraian surat / perihal surat",
    height=180
)

if st.button("Cari Klasifikasi"):

    if user_text.strip() == "":
        st.warning("Silakan isi uraian surat.")
        st.stop()

    # -----------------------------------------
    # Gemini Routing
    # -----------------------------------------
    with st.spinner("Gemini menganalisis surat..."):

        raw = route_query(user_text)
        info = parse_json(raw)

    primer = str(info["primer"])

    st.subheader("🧠 Hasil Analisis")

    st.write("Domain:", info["domain"])
    st.write("Primer:", primer, "-", primer_map.get(primer, ""))
    st.write("Keywords:", ", ".join(info["keywords"]))

    # -----------------------------------------
    # Search Kuartier
    # -----------------------------------------
    with st.spinner("Mencari kuartier..."):

        rows = semantic_search(
            user_text,
            emb_q,
            meta_q,
            topk=20
        )

        rows = filter_primer(rows, primer)

        final = rerank(user_text, rows)

    # fallback
    if len(final) == 0 or final[0]["score"] < 0.45:

        st.info("Fallback ke level tersier...")

        rows = semantic_search(
            user_text,
            emb_t,
            meta_t,
            topk=10
        )

        final = rerank(user_text, rows)

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    st.subheader("🎯 Top 3 Rekomendasi")

    for i, row in enumerate(final, start=1):

        st.markdown(f"""
### {i}. {row['kode']}

**Uraian:** {row['uraian']}  
**Level:** {row['level']}  
**Score:** {round(row['score'],4)}
""")
