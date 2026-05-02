# 🗂️ SIKAP — Sistem Klasifikasi Arsip Terpadu

Aplikasi klasifikasi arsip otomatis berbasis AI yang menggunakan pendekatan hybrid:
**LLM (Gemini Flash) + Embedding (BGE-M3) + Vector Search (FAISS) + Hierarchical Classification**

---

## ✨ Fitur Utama

- **Ekstraksi Inti Surat** — Gemini Flash mengambil topik inti dari teks bebas
- **Semantic Search** — BGE-M3 embedding + FAISS mencari klasifikasi yang paling mirip
- **Hierarchical Classification** — Prioritas kuartier (4 level), fallback ke tersier
- **Weighted Scoring** — `0.7×similarity + 0.2×domain + 0.1×activity`
- **Top 3 Rekomendasi** — Kode + uraian + skor confidence
- **UI Modern** — Streamlit dengan tampilan bersih dan profesional

---

## 📁 Struktur Project

```
sikap-app/
├── app.py              ← Main Streamlit app
├── utils.py            ← Load model, embed, search, scoring
├── prompts.py          ← Prompt Gemini untuk ekstraksi inti
├── requirements.txt    ← Dependencies
├── sikap_bge.index     ← FAISS index (wajib ada, ~11 MB)
├── metadata.pkl        ← Metadata klasifikasi (wajib ada, ~1.4 MB)
├── README.md
└── .gitignore
```

---

## 🚀 Deploy ke Streamlit Community Cloud

### Langkah 1 — Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit SIKAP"
git remote add origin https://github.com/username/sikap-app.git
git push -u origin main
```

> ⚠️ **Pastikan `sikap_bge.index` dan `metadata.pkl` ikut di-push.**
> File ini tidak ada di `.gitignore`.

### Langkah 2 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik **"New app"**
3. Pilih repository, branch `main`, file `app.py`
4. Klik **"Deploy!"**

### Langkah 3 — Tambahkan API Key

Di Streamlit Cloud:
1. Buka **Settings → Secrets**
2. Tambahkan:

```toml
GOOGLE_API_KEY = "AIzaSy..."
```

---

## 💻 Menjalankan Secara Lokal

### Prasyarat

- Python 3.10 atau 3.11
- Git

### Instalasi

```bash
# 1. Clone repository
git clone https://github.com/username/sikap-app.git
cd sikap-app

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Buat file secrets lokal
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "AIzaSy..."' > .streamlit/secrets.toml

# 5. Jalankan aplikasi
streamlit run app.py
```

Buka browser: `http://localhost:8501`

---

## ⚙️ Konfigurasi Teknis

### Model Gemini

File: `app.py` → fungsi `extract_inti_surat()`

```python
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
```

Ganti dengan model ID terbaru jika diperlukan. Cek di:
https://ai.google.dev/gemini-api/docs/models

### Threshold Similarity

File: `utils.py`

```python
SIMILARITY_THRESHOLD_QUARTIER = 0.45   # min untuk kuartier diterima
SIMILARITY_THRESHOLD_TERTIER  = 0.30   # min untuk tersier diterima
```

Turunkan nilai jika rekomendasi terlalu sedikit.
Naikkan nilai jika rekomendasi kurang relevan.

### Formula Scoring

```python
FINAL_SCORE = (embedding_similarity × 0.70)
            + (domain_match        × 0.20)
            + (activity_match      × 0.10)
```

Edit di `utils.py` → fungsi `score_candidates()`.

---

## 📊 Format metadata.pkl

File ini harus mengandung salah satu format berikut:

**List of dict:**
```python
[
  {
    "kode": "500.2.1.4",
    "uraian": "Pengadaan Sistem Arsip Digital",
    "penjelasan": "...",
    "konteks": "...",
    "domain": "teknologi informasi",
    "activity": "pengadaan",
    "embedding_text": "..."
  },
  ...
]
```

**Atau pandas DataFrame** (akan dikonversi otomatis).

---

## ⚡ Optimasi Resource (Streamlit Cloud Gratis)

| Optimasi | Implementasi |
|---|---|
| Model BGE-M3 load sekali | `@st.cache_resource` |
| FAISS index load sekali | `@st.cache_resource` |
| Metadata load sekali | `@st.cache_resource` |
| Gemini: 1 call per proses | `max_output_tokens=30` |
| BGE-M3 hemat memori | `use_fp16=True` |
| Search terbatas | `top_k=40` kandidat |

> 💡 **Cold start** pertama membutuhkan ~60–90 detik untuk download model BGE-M3.
> Setelah itu, semua request berikutnya sangat cepat.

---

## 🐛 Troubleshooting

| Error | Solusi |
|---|---|
| `FileNotFoundError: sikap_bge.index` | Push file ke GitHub, pastikan tidak di `.gitignore` |
| `GOOGLE_API_KEY tidak ditemukan` | Tambahkan di Streamlit Secrets |
| `Out of Memory` | Streamlit Cloud free tier: 1GB RAM. Coba restart app |
| `Model not found` | Ganti model ID Gemini di `app.py` dengan yang tersedia |

---

## 📄 Lisensi

Internal use only.
