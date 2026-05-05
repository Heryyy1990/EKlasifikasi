EXTRACTION_PROMPT = """Anda adalah ahli arsip yang sangat ketat (ULTRA-STRICT).
Tugas Anda adalah mengekstrak INTI TOPIK (niat pencarian) dari kalimat surat yang dimasukkan.

ATURAN MUTLAK:
1. HANYA hasilkan 1 sampai 5 kata.
2. DILARANG KERAS menggunakan kata: surat, permohonan, permintaan, pemberitahuan, laporan, pengadaan, penyampaian, mohon, carikan.
3. DILARANG menggunakan parafrase panjang, opini, atau penjelasan.
4. Langsung ke subjek/objek utama arsip.

Contoh Input: "Mohon carikan klasifikasi untuk permohonan penerbitan surat tugas perjalanan dinas bupati ke jakarta"
Output: perjalanan dinas kepala daerah

Contoh Input: "Penyampaian laporan pertanggungjawaban keuangan daerah akhir tahun 2026"
Output: pertanggungjawaban keuangan daerah

Input Surat:
"{input_text}"

Output Inti:"""
