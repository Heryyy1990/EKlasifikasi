"""
build_index.py
──────────────────────────────────────────────────────────────────
Script LOKAL untuk membangun:
  - meta_kuartier.pkl  & faiss_kuartier.index
  - meta_tersier.pkl   & faiss_tersier.index

Jalankan SEKALI di komputer lokal, lalu push semua file ke GitHub.
Tidak perlu dijalankan di Streamlit Cloud.

Prasyarat:
  pip install google-genai faiss-cpu pandas numpy tqdm
  export GOOGLE_API_KEY="AIza..."

Jalankan:
  python build_index.py
──────────────────────────────────────────────────────────────────
"""

import os, pickle, time
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from google import genai

# ── CONFIG ──────────────────────────────────────────────────────
CSV_PATH        = "klasifikasi_sikap_final_v2.csv"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBED_DIM       = 768
BATCH_DELAY     = 1.2   # detik antar request (hindari rate-limit free-tier)
# ────────────────────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Set environment variable GOOGLE_API_KEY terlebih dahulu.")
    return genai.Client(api_key=api_key)

def embed_text(client, text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def build_for_level(client, rows: list[dict], level_name: str):
    """Embed semua baris dan buat FAISS index + metadata."""
    vectors = []
    meta    = []

    print(f"\n[{level_name.upper()}] Memproses {len(rows)} baris…")
    for row in tqdm(rows):
        text = row.get("embedding_text") or row.get("uraian", "")
        if not text.strip():
            text = row.get("kode", "?")
        try:
            vec = embed_text(client, text)
        except Exception as e:
            print(f"  ⚠ Skip kode {row.get('kode','?')}: {e}")
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
        vectors.append(vec)

        # simpan kolom yang diperlukan
        meta.append({
            "kode"       : row.get("kode", ""),
            "uraian"     : row.get("uraian", ""),
            "penjelasan" : row.get("penjelasan", ""),
            "konteks"    : row.get("konteks", ""),
            "domain"     : row.get("domain", ""),
            "keywords"   : row.get("keywords", ""),
            "level"      : row.get("level", level_name),
        })
        time.sleep(BATCH_DELAY)

    mat = np.vstack(vectors)

    # Inner product index (setara cosine similarity karena sudah L2-norm)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(mat)

    # Simpan
    idx_path  = f"faiss_{level_name}.index"
    meta_path = f"meta_{level_name}.pkl"
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"  ✓ Tersimpan: {idx_path} ({index.ntotal} vektor) | {meta_path}")

def main():
    client = get_client()

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    print(f"Dataset dimuat: {len(df)} baris | Kolom: {list(df.columns)}")

    # Pastikan kolom 'level' ada
    if "level" not in df.columns:
        raise ValueError("Kolom 'level' tidak ditemukan di CSV. "
                         "Pastikan CSV sudah memiliki kolom: kode, uraian, level, embedding_text, …")

    kuartier_rows = df[df["level"] == "kuartier"].to_dict(orient="records")
    tersier_rows  = df[df["level"] == "tersier"].to_dict(orient="records")

    if not kuartier_rows:
        raise ValueError("Tidak ada baris dengan level='kuartier' di CSV.")
    if not tersier_rows:
        print("  ⚠ Tidak ada baris tersier – tersier index akan kosong.")

    build_for_level(client, kuartier_rows, "kuartier")
    if tersier_rows:
        build_for_level(client, tersier_rows,  "tersier")

    print("\n✅ Semua index berhasil dibangun.")
    print("   Push file-file berikut ke GitHub:")
    print("     klasifikasi_sikap_final_v2.csv")
    print("     meta_kuartier.pkl  | faiss_kuartier.index")
    print("     meta_tersier.pkl   | faiss_tersier.index")
    print("     app.py             | requirements.txt")

if __name__ == "__main__":
    main()
