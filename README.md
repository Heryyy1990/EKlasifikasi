# SIKAP (Sistem Informasi Klasifikasi Arsip Pintar)

Aplikasi hybrid Semantic Search untuk klasifikasi tata naskah dinas.

## Arsitektur
1. **Google Gemini Flash 2.5**: Mengekstrak intent / inti surat dari kalimat ambigu (LLM).
2. **MiniLM**: Model Sentence-Transformers lokal untuk mengubah intent menjadi vektor.
3. **FAISS**: Vector database untuk pencarian instan secara offline.

## Cara Deploy ke Streamlit Community Cloud
1. Push repository ini ke akun GitHub Anda.
2. Login ke [share.streamlit.io](https://share.streamlit.io).
3. Buat aplikasi baru dan pilih repository SIKAP.
4. **SANGAT PENTING**: Sebelum menekan deploy, klik **Advanced settings...**
5. Di bagian **Secrets**, masukkan API Key Google AI Studio Anda dengan format:
   ```toml
   GOOGLE_API_KEY="AIzaSy...masukkan_api_key_anda_di_sini..."
