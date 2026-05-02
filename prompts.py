# ============================================================
# prompts.py — Prompt template untuk ekstraksi inti surat
# SIKAP — Sistem Klasifikasi Arsip Terpadu
# ============================================================

EXTRACT_INTI_PROMPT = """\
Tugasmu adalah mengekstrak INTI TOPIK dari teks surat berikut.

ATURAN WAJIB — HARUS DIPATUHI SEPENUHNYA:
1. Keluarkan HANYA 1 sampai 5 kata
2. Murni inti substansi/topik utama surat
3. DILARANG menggunakan kata-kata administratif berikut:
   surat, permohonan, permintaan, pemberitahuan, pengajuan,
   laporan, undangan, penyampaian, terkait, perihal, mengenai,
   kepada, dari, dengan hormat, bersama ini
4. DILARANG membuat kalimat panjang
5. DILARANG menambahkan opini, interpretasi, atau penjelasan
6. Gunakan bahasa Indonesia ringkas

CONTOH BENAR:
Input: "Permintaan pengadaan sistem arsip digital untuk mendukung pengelolaan dokumen"
Output: pengadaan sistem arsip digital

Input: "Surat permohonan cuti tahunan pegawai negeri sipil tahun 2024"
Output: cuti tahunan pegawai

Input: "Pemberitahuan pelaksanaan diklat kepemimpinan tingkat II untuk pejabat eselon"
Output: diklat kepemimpinan eselon

Input: "Undangan rapat koordinasi perencanaan anggaran tahun depan"
Output: rapat koordinasi anggaran

Teks surat yang harus diproses:
{text}

Inti surat (1-5 kata saja, tanpa tanda baca):"""
