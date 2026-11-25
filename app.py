"""
app.py - Antarmuka web Streamlit untuk Sistem Temu Kembali Berita PoWeR-BM25.
Jalankan dengan: py -m streamlit run app.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from engine import (
    BASELINE_WEIGHTS,
    PoWeRBM25,
    evaluate_ranking,
    normalize_dataframe,
    preprocess_detail,
    stem_word,
)
from fetcher import fetch_live, load_cache, save_to_cache

st.set_page_config(page_title="PoWeR-BM25 Pencari Berita", page_icon="📰", layout="wide")

BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "dataset_berita.csv"
SAMPLE_JUDGMENT_CSV = BASE_DIR / "contoh_penilaian.csv"
CACHE_CSV = BASE_DIR / "live_cache.csv"

SRC_LOCAL = "📁 Dataset lokal (CSV)"
SRC_LIVE = "🌐 Internet (Live Search)"
SRC_CACHE = "🗃️ Cache hasil live"


# --------------------------------------------------------------------------
# Inisialisasi file sampel otomatis jika CSV kosong/belum ada
# --------------------------------------------------------------------------
def ensure_sample_files() -> None:
    if not DEFAULT_CSV.exists() or DEFAULT_CSV.stat().st_size == 0:
        sample_news = pd.DataFrame([
            {"id": 1, "judul": "Banjir Rendam Sejumlah Ruas Jalan Utama di Jakarta Timur", "intro": "Hujan deras sejak dini hari menyebabkan banjir setinggi 50 sentimeter di beberapa titik Jakarta Timur.", "isi": "Petugas penanganan prasarana dan sarana umum dikerahkan untuk menyedot genangan banjir di jalan protokol Jakarta Timur. Lalu lintas sempat dialihkan guna menghindari kemacetan panjang akibat banjir yang merendam permukiman warga."},
            {"id": 2, "judul": "Pemprov DKI Siapkan Pompa Mobile Antisipasi Banjir Musim Hujan", "intro": "Pemerintah Provinsi DKI Jakarta menyiagakan ratusan pompa air mobile di wilayah rawan banjir.", "isi": "Langkah mitigasi banjir dilakukan dengan pengerukan saluran air serta pengecekan pintu air di seluruh wilayah Jakarta. Dinas Sumber Daya Air memastikan pompa siap beroperasi penuh saat curah hujan tinggi."},
            {"id": 3, "judul": "Warga Kampung Melayu Mulai Mengungsi Akibat Luapan Kali Ciliwung", "intro": "Ketinggian air Kali Ciliwung meningkat drastis memicu banjir di kawasan Kampung Melayu Jakarta.", "isi": "Sejumlah warga mulai mengevakuasi barang berharga ke posko pengungsian terdekat karena banjir mulai memasuki rumah warga. Bantuan logistik dan perahu karet telah disiagakan di lokasi terdampak banjir."},
            {"id": 4, "judul": "Harga Cabai Rawit Merah di Pasar Tradisional Tembus Rp80 Ribu per Kilogram", "intro": "Kenaikan harga cabai rawit merah dikeluhkan pedagang dan pembeli di sejumlah pasar tradisional.", "isi": "Pasokan cabai dari daerah sentra produksi berkurang akibat cuaca ekstrem sehingga harga cabai melonjak tajam. Pedagang berharap distribusi pasokan segera normal agar harga cabai kembali stabil."},
            {"id": 5, "judul": "Kementan Gelar Operasi Pasar Tekan Lonjakan Harga Cabai dan Bawang", "intro": "Kementerian Pertanian menggelar operasi pasar murah untuk menstabilkan harga cabai yang melambung.", "isi": "Melalui operasi pasar ini, masyarakat dapat membeli cabai rawit dan cabai merah keriting dengan harga di bawah pasaran. Pemerintah bekerja sama dengan asosiasi petani cabai untuk mempercepat distribusi stok."},
            {"id": 6, "judul": "Geger! Rahasia di Balik Harga Cabai yang Bikin Ibu-Ibu Syok Berat", "intro": "Fenomena kenaikan kebutuhan pokok akhir-akhir ini menjadi perbincangan hangat di media sosial.", "isi": "Banyak warganet membagikan kisah penghematan belanja harian di tengah kondisi ekonomi yang dinamis. Pengamat ekonomi menyarankan masyarakat mengatur prioritas pengeluaran rumah tangga dengan bijak."},
            {"id": 7, "judul": "Timnas Indonesia Fokus Matangkan Taktik Jelang Kualifikasi Piala Dunia", "intro": "Pelatih Timnas Indonesia menggelar sesi latihan intensif untuk mematangkan strategi permainan tim.", "isi": "Para pemain Timnas Indonesia mengikuti latihan taktikal dan fisik di lapangan utama. Fokus utama tim pelatih adalah memperkuat transisi bertahan ke menyerang menjelang laga penting kualifikasi Piala Dunia."},
            {"id": 8, "judul": "Jadwal Pertandingan Timnas Indonesia di Stadion Utama Gelora Bung Karno", "intro": "PSSI resmi merilis jadwal penjualan tiket laga kandang Timnas Indonesia di SUGBK Jakarta.", "isi": "Antusiasme suporter untuk menyaksikan aksi Timnas Indonesia sangat tinggi. Panitia pelaksana mengimbau penonton membeli tiket melalui kanal resmi dan datang lebih awal ke stadion."},
            {"id": 9, "judul": "Mengejutkan! Pemain Timnas Indonesia Ini Bikin Pelatih Lawan Tak Bisa Tidur", "intro": "Sorotan media asing tertuju pada persiapan skuad Garuda menjelang turnamen internasional bulan depan.", "isi": "Berbagai analisis bermunculan mengenai kekuatan sepak bola Asia Tenggara yang semakin kompetitif. Federasi sepak bola terus mendorong pembinaan pemain muda demi prestasi jangka panjang."},
            {"id": 10, "judul": "Bank Indonesia Pertahankan Suku Bunga Acuan Demi Jaga Stabilitas Rupiah", "intro": "Rapat Dewan Gubernur Bank Indonesia memutuskan mempertahankan BI-Rate pada level yang stabil.", "isi": "Keputusan ini diambil untuk menjaga inflasi tetap terkendali dan memperkuat stabilitas nilai tukar rupiah di tengah ketidakpastian pasar keuangan global."},
            {"id": 11, "judul": "BMKG Peringatkan Potensi Cuaca Ekstrem dan Banjir Pesisir di Utara Jakarta", "intro": "Badan Meteorologi Klimatologi dan Geofisika mengeluarkan peringatan dini potensi banjir rob di pesisir Jakarta.", "isi": "Fenomena pasang maksimum air laut berpotensi memicu banjir pesisir di wilayah utara Jakarta dan sekitarnya. Warga di kawasan pesisir diimbau waspada terhadap peningkatan genangan banjir."},
            {"id": 12, "judul": "Heboh! Cara Ajaib Atasi Banjir Jakarta Tanpa Biaya Sepeser Pun", "intro": "Sebuah unggahan viral di jagat maya memicu perdebatan sengit di kalangan warganet ibu kota.", "isi": "Diskusi mengenai tata kota dan lingkungan hidup memang selalu menarik perhatian publik. Para pakar tata ruang menekankan perlunya kolaborasi antara pemerintah dan kesadaran warga menjaga kebersihan sungai."},
            {"id": 13, "judul": "Petani Cabai di Jawa Timur Panen Raya, Pasokan Mulai Membaik", "intro": "Memasuki musim panen di sentra pertanian Jawa Timur, distribusi cabai ke pasar induk meningkat.", "isi": "Peningkatan hasil panen cabai diharapkan mampu meredam gejolak harga cabai di tingkat konsumen dalam beberapa pekan ke depan."},
            {"id": 14, "judul": "Tak Disangka! Ternyata Ini Alasan Utama Timnas Indonesia Sulit Dikalahkan", "intro": "Perkembangan olahraga nasional menunjukkan tren positif dalam beberapa tahun terakhir.", "isi": "Dukungan fasilitas olahraga modern dan sport science menjadi faktor penting dalam meningkatkan kebugaran atlet profesional di berbagai cabang olahraga."},
        ])
        sample_news.to_csv(DEFAULT_CSV, index=False, encoding="utf-8")

    if not SAMPLE_JUDGMENT_CSV.exists() or SAMPLE_JUDGMENT_CSV.stat().st_size == 0:
        sample_judg = pd.DataFrame([
            {"kueri": "banjir jakarta", "relevan_id": "1;2;3;11"},
            {"kueri": "harga cabai", "relevan_id": "4;5;13"},
            {"kueri": "timnas indonesia", "relevan_id": "7;8"},
        ])
        sample_judg.to_csv(SAMPLE_JUDGMENT_CSV, index=False, encoding="utf-8")


ensure_sample_files()


# --------------------------------------------------------------------------
# Styling CSS: Kalem & Seimbang di Light, Deep Navy di Dark, Tanpa Tombol Deploy
# --------------------------------------------------------------------------
st.markdown("""
<style>
    /* ======================================================================
       1. BAR HEADER PALING ATAS & SEMBUNYIKAN TOMBOL DEPLOY
       ====================================================================== */
    header[data-testid="stHeader"] {
        background: light-dark(
            linear-gradient(90deg, #e8eef6 0%, #f1f5f9 100%),
            linear-gradient(90deg, #090e1a 0%, #172554 100%)
        ) !important;
        border-bottom: 1.5px solid light-dark(#cbd5e1, #38bdf8);
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] span {
        color: light-dark(#1e293b, #f8fafc) !important;
        fill: light-dark(#1e293b, #f8fafc) !important;
    }

    /* Hapus ikon/tombol Deploy di pojok kanan atas (menu titik tiga tetap ada) */
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    /* ======================================================================
       2. LATAR BELAKANG HALAMAN UTAMA (Lembut & Tidak Silau di Mode Light)
       ====================================================================== */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main {
        background: light-dark(
            #e9eef5,
            linear-gradient(180deg, #0b1120 0%, #0f172a 100%)
        ) !important;
        color: light-dark(#0f172a, #f1f5f9) !important;
        accent-color: light-dark(#264d78, #38bdf8);
    }
    .block-container {
        padding-top: 4.2rem;
        padding-bottom: 2.5rem;
    }

    /* Warna Teks Seluruh Elemen di Area Utama */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stRadio"] label * {
        color: light-dark(#1e293b, #f1f5f9) !important;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] small {
        color: light-dark(#64748b, #94a3b8) !important;
    }
    [data-testid="stMain"] code {
        background-color: light-dark(#e4ecf7, rgba(56, 189, 248, 0.16)) !important;
        color: light-dark(#1e3a5f, #7dd3fc) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.3));
    }
    [data-testid="stMain"] hr {
        border-color: light-dark(#cbd5e1, rgba(148, 163, 184, 0.22)) !important;
    }

    /* Blok Kode Tahapan Preprocessing */
    [data-testid="stMain"] [data-testid="stCode"],
    [data-testid="stMain"] [data-testid="stCode"] pre,
    [data-testid="stMain"] pre {
        background-color: light-dark(#f1f5f9, #0b1120) !important;
        color: light-dark(#1e3a5f, #7dd3fc) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.32)) !important;
        border-radius: 8px !important;
    }
    [data-testid="stMain"] [data-testid="stCode"] code {
        background-color: transparent !important;
        border: none !important;
        color: light-dark(#1e3a5f, #7dd3fc) !important;
    }

    /* Kotak Input Teks, Angka, Select & Multiselect di Area Utama */
    [data-testid="stMain"] [data-baseweb="input"],
    [data-testid="stMain"] [data-baseweb="base-input"],
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="base-input"] > div,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background-color: light-dark(#f8fafc, #162033) !important;
        border-color: light-dark(#cbd5e1, rgba(56, 189, 248, 0.35)) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
        border-radius: 10px !important;
    }
    [data-testid="stMain"] input {
        background-color: light-dark(#f8fafc, #162033) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#0f172a, #f8fafc) !important;
        caret-color: light-dark(#264d78, #38bdf8) !important;
    }
    [data-testid="stMain"] input::placeholder {
        color: light-dark(#64748b, #94a3b8) !important;
        -webkit-text-fill-color: light-dark(#64748b, #94a3b8) !important;
    }
    [data-testid="stMain"] [data-baseweb="tag"] {
        background-color: light-dark(#264d78, #1d4ed8) !important;
        color: #ffffff !important;
    }
    [data-testid="stMain"] [data-baseweb="tag"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stMain"] [data-testid="stNumberInput"] button {
        background-color: light-dark(#e2e8f0, #1e293b) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
        fill: light-dark(#0f172a, #f8fafc) !important;
        border-color: light-dark(#cbd5e1, rgba(148, 163, 184, 0.3)) !important;
    }
    [data-testid="stMain"] [data-testid="stNumberInput"] button:hover {
        background-color: light-dark(#264d78, #2563eb) !important;
        color: #ffffff !important;
    }

    /* Tombol Kueri Cepat, Unduh & Link di Area Utama */
    [data-testid="stMain"] .stButton > button,
    [data-testid="stMain"] .stDownloadButton > button,
    [data-testid="stMain"] .stLinkButton > a {
        background: light-dark(
            #f8fafc,
            linear-gradient(180deg, #162033 0%, #1e293b 100%)
        ) !important;
        color: light-dark(#1e3a5f, #e0f2fe) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.35)) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: light-dark(0 1px 3px rgba(15, 23, 42, 0.05), 0 2px 8px rgba(2, 6, 23, 0.35));
        transition: all 0.15s ease;
    }
    [data-testid="stMain"] .stButton > button:hover,
    [data-testid="stMain"] .stDownloadButton > button:hover,
    [data-testid="stMain"] .stLinkButton > a:hover {
        background: light-dark(
            linear-gradient(90deg, #1e3a5f 0%, #2b547e 100%),
            linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%)
        ) !important;
        border-color: light-dark(#264d78, #38bdf8) !important;
        color: #ffffff !important;
    }

    /* ======================================================================
       3. BILAH TAB NAVIGASI (Garis Tunggal Rapi + Indikator Biru Tema)
       ====================================================================== */
    [data-testid="stMain"] [role="tablist"] {
        background-color: light-dark(#f8fafc, #131c31) !important;
        padding: 8px 14px !important;
        border-radius: 12px;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.28));
        box-shadow: light-dark(0 2px 6px rgba(15, 23, 42, 0.04), 0 4px 14px rgba(2, 6, 23, 0.3));
        margin-bottom: 14px;
        gap: 6px;
    }
    [data-testid="stMain"] [data-baseweb="tab-border"],
    [data-testid="stMain"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    [data-testid="stMain"] button[role="tab"] {
        color: light-dark(#64748b, #94a3b8) !important;
        font-weight: 600 !important;
        padding-bottom: 6px !important;
        border-bottom: 3px solid transparent !important;
    }
    [data-testid="stMain"] button[role="tab"][aria-selected="true"] {
        color: light-dark(#1e3a5f, #38bdf8) !important;
        border-bottom: 3px solid light-dark(#264d78, #38bdf8) !important;
    }

    /* ======================================================================
       4. KOTAK ALERT (Garis Tunggal Bersih, Tanpa Double Warna/Garis)
       ====================================================================== */
    [data-testid="stMain"] [data-testid="stAlert"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    [data-testid="stMain"] [data-testid="stAlert"] > div {
        background-color: light-dark(#e4ecf7, #162447) !important;
        border: 1px solid light-dark(#bfd0e6, rgba(56, 189, 248, 0.42)) !important;
        border-left: 4px solid light-dark(#264d78, #38bdf8) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    [data-testid="stMain"] [data-testid="stAlert"] > div div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stMain"] [data-testid="stAlert"] * {
        color: light-dark(#1e3a5f, #f1f5f9) !important;
        fill: light-dark(#264d78, #38bdf8) !important;
    }

    /* ======================================================================
       5. KARTU METRIK, EXPANDER & KARTU BERITA DI AREA UTAMA
       ====================================================================== */
    [data-testid="stMain"] [data-testid="stMetric"] {
        background-color: light-dark(#f8fafc, #131c31) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.28)) !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: light-dark(0 2px 6px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(2, 6, 23, 0.28));
    }
    [data-testid="stMain"] [data-testid="stMetricLabel"] * {
        color: light-dark(#64748b, #94a3b8) !important;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"] * {
        color: light-dark(#0f172a, #f8fafc) !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"],
    [data-testid="stMain"] [data-testid="stExpander"] details,
    [data-testid="stMain"] [data-testid="stExpanderDetails"] {
        background-color: light-dark(#f8fafc, #131c31) !important;
        border-color: light-dark(#cbd5e1, rgba(56, 189, 248, 0.28)) !important;
        border-radius: 12px !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"] summary,
    [data-testid="stMain"] [data-testid="stExpander"] summary * {
        background-color: light-dark(#f8fafc, #131c31) !important;
        color: light-dark(#1e293b, #f1f5f9) !important;
        fill: light-dark(#1e293b, #f1f5f9) !important;
    }
    [data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: light-dark(#f8fafc, #131c31) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.28)) !important;
        border-radius: 14px !important;
        box-shadow: light-dark(0 2px 8px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(2, 6, 23, 0.35)) !important;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
        background-color: light-dark(#f1f5f9, #162033) !important;
        border: 1.5px dashed light-dark(#94a3b8, rgba(56, 189, 248, 0.45)) !important;
        border-radius: 10px !important;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] * {
        color: light-dark(#334155, #e2e8f0) !important;
    }

    /* Menu Dropdown Popover Global */
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: light-dark(#f8fafc, #1e293b) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
    }
    li[role="option"]:hover {
        background-color: light-dark(#e4ecf7, #2563eb) !important;
    }

    /* ======================================================================
       6. SIDEBAR KIRI MENYATU PENUH (Kalem di Mode Light, Deep Navy di Dark)
       ====================================================================== */
    section[data-testid="stSidebar"] {
        background: light-dark(
            #dde6f2,
            linear-gradient(180deg, #090e1a 0%, #172554 45%, #090e1a 100%)
        ) !important;
        border-right: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.45));
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        background: transparent !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: light-dark(#1e293b, #f1f5f9) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] small {
        color: light-dark(#64748b, #94a3b8) !important;
    }
    section[data-testid="stSidebar"] code {
        background-color: light-dark(#e4ecf7, rgba(56, 189, 248, 0.16)) !important;
        color: light-dark(#1e3a5f, #7dd3fc) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(56, 189, 248, 0.3));
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: light-dark(#f8fafc, rgba(15, 23, 42, 0.68)) !important;
        border: 1px solid light-dark(#cbd5e1, rgba(148, 163, 184, 0.25)) !important;
        border-radius: 12px !important;
        box-shadow: light-dark(0 1px 6px rgba(15, 23, 42, 0.04), 0 4px 14px rgba(2, 6, 23, 0.32));
    }

    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"],
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] [data-baseweb="base-input"] > div {
        background-color: light-dark(#f1f5f9, #1e293b) !important;
        border-color: light-dark(#cbd5e1, rgba(148, 163, 184, 0.4)) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] input[type="number"],
    section[data-testid="stSidebar"] input[type="text"] {
        background-color: light-dark(#f1f5f9, #1e293b) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#0f172a, #f8fafc) !important;
        caret-color: light-dark(#264d78, #38bdf8) !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
    section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
    section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] {
        background-color: light-dark(#e2e8f0, #334155) !important;
        color: light-dark(#0f172a, #f8fafc) !important;
        fill: light-dark(#0f172a, #f8fafc) !important;
        border-color: light-dark(#cbd5e1, rgba(148, 163, 184, 0.3)) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
        background-color: light-dark(#264d78, #2563eb) !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: light-dark(#f1f5f9, #1e293b) !important;
        border: 1px dashed light-dark(#94a3b8, rgba(125, 211, 252, 0.5)) !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: light-dark(#334155, #e2e8f0) !important;
    }

    /* Perbaikan Kontras Teks & Ikon Putih di Dalam Tombol Upload */
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: light-dark(
            linear-gradient(90deg, #1e3a5f 0%, #2b547e 100%),
            linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%)
        ) !important;
        border: 1px solid light-dark(#264d78, #3b82f6) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button *,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        background: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        fill: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: light-dark(
            linear-gradient(90deg, #1e3a5f 0%, #2b547e 100%),
            linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%)
        ) !important;
        color: #ffffff !important;
        border: 1px solid light-dark(#264d78, #3b82f6) !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: light-dark(
            linear-gradient(90deg, #172e4d 0%, #1e3a5f 100%),
            linear-gradient(90deg, #1e40af 0%, #1d4ed8 100%)
        ) !important;
        border-color: #60a5fa !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
        fill: light-dark(#64748b, #94a3b8) !important;
        color: light-dark(#64748b, #94a3b8) !important;
    }

    /* Kartu Brand Sidebar (Steel Navy Kalem di Light, Deep Navy di Dark) */
    .sidebar-brand {
        background: light-dark(
            linear-gradient(135deg, #1e3a5f 0%, #2b547e 100%),
            linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)
        );
        border: 1px solid light-dark(#3b6998, rgba(56, 189, 248, 0.45));
        border-left: 4px solid light-dark(#7dd3fc, #38bdf8);
        border-radius: 12px;
        padding: 13px 15px;
        margin-bottom: 12px;
        box-shadow: light-dark(0 3px 10px rgba(30, 58, 95, 0.14), 0 4px 12px rgba(15, 23, 42, 0.25));
    }
    .sidebar-brand-title {
        font-size: 1.03rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0 0 4px 0;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: light-dark(#dbeafe, #bae6fd) !important;
        margin: 0;
        line-height: 1.35;
    }

    /* ======================================================================
       7. BANNER HERO HEADER UTAMA (Steel Navy Kalem di Light, Deep Navy di Dark)
       ====================================================================== */
    .hero-box {
        background: light-dark(
            linear-gradient(135deg, #1e3a5f 0%, #2b547e 60%, #37699c 100%),
            linear-gradient(135deg, #111c38 0%, #1e3a8a 58%, #1d4ed8 100%)
        );
        color: #ffffff;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        border: 1px solid light-dark(#4a7eb5, rgba(56, 189, 248, 0.35));
        border-bottom: 4px solid light-dark(#7dd3fc, #38bdf8);
        box-shadow: light-dark(0 6px 18px rgba(30, 58, 95, 0.14), 0 10px 28px rgba(15, 23, 42, 0.28));
    }
    .hero-tag {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.14);
        color: light-dark(#e0f2fe, #7dd3fc) !important;
        border: 1px solid rgba(255, 255, 255, 0.28);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.72rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0 0 6px 0;
        line-height: 1.25;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: light-dark(#e2e8f0, #e2e8f0) !important;
        margin: 0;
        line-height: 1.45;
    }
    .hero-sub b {
        color: #ffffff !important;
    }

    /* ======================================================================
       8. LENCANA STATUS, PIL TF & HIGHLIGHT KATA KUNCI
       ====================================================================== */
    .badge-cb {
        display: inline-block;
        background-color: #dc2626;
        color: #ffffff !important;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ok {
        display: inline-block;
        background-color: #16a34a;
        color: #ffffff !important;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .tf-pill {
        display: inline-block;
        border: 1px solid light-dark(#cbd5e1, rgba(148, 163, 184, 0.32));
        background-color: light-dark(#eef2f7, #1e293b);
        color: light-dark(#1e293b, #e2e8f0) !important;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.76rem;
        margin-right: 4px;
    }
    .meta-line {
        font-size: 0.85rem;
        color: light-dark(#64748b, #94a3b8) !important;
        margin-bottom: 8px;
    }
    .news-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: light-dark(#0f172a, #f8fafc) !important;
        margin: 4px 0 8px 0;
        line-height: 1.35;
    }
    mark.hl-term {
        background-color: light-dark(#fef08a, rgba(250, 204, 21, 0.26));
        color: light-dark(#0f172a, #fef08a) !important;
        border: 1px solid light-dark(#eab308, rgba(250, 204, 21, 0.45));
        padding: 1px 4px;
        border-radius: 4px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Utilitas Pemuatan Data, Indeks, & Highlight Kata Kunci
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(DEFAULT_CSV))


def read_uploaded_csv(file) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return normalize_dataframe(pd.read_csv(file, encoding=enc))
        except UnicodeDecodeError:
            continue
    raise ValueError("Encoding file tidak didukung (gunakan UTF-8).")


@st.cache_resource(show_spinner="Membangun indeks (stemming Sastrawi)...")
def get_index(df: pd.DataFrame) -> PoWeRBM25:
    return PoWeRBM25(df)


def set_quick_query(q: str) -> None:
    st.session_state["query_box"] = q


def highlight_text(text: str, q_stems: set[str]) -> str:
    """Sorot kata pada teks berita yang kata dasarnya (stem) cocok dengan kueri."""
    raw = str(text)
    if not q_stems:
        return html.escape(raw)
    parts = re.split(r"([A-Za-z0-9]+)", raw)
    out = []
    for part in parts:
        if len(part) > 1 and part.isalnum() and stem_word(part.lower()) in q_stems:
            out.append(f'<mark class="hl-term">{html.escape(part)}</mark>')
        else:
            out.append(html.escape(part))
    return "".join(out)


# --------------------------------------------------------------------------
# Sidebar: Menyatu Penuh dari Atas sampai Bawah
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⚙️ Panel Kendali PoWeR-BM25</div>
        <div class="sidebar-brand-sub">Konfigurasi korpus berita, bobot zona &amp; penalti clickbait</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("##### 🌐 Sumber Berita")
        source = st.radio(
            "Sumber data",
            [SRC_LOCAL, SRC_LIVE, SRC_CACHE],
            label_visibility="collapsed",
            help="Lokal: dataset CSV. Internet: cari berita terbaru lewat RSS lalu unduh "
                 "isinya secara real-time. Cache: seluruh hasil live yang pernah tersimpan.",
        )
        df = None
        dataset_name = ""
        max_art, use_cache_idf, refresh = 12, True, False
        if source == SRC_LOCAL:
            uploaded = st.file_uploader("Unggah CSV baru (kolom: judul, intro, isi)", type=["csv"])
            df = load_default_dataset()
            dataset_name = "dataset_berita.csv"
            if uploaded is not None:
                try:
                    df = read_uploaded_csv(uploaded)
                    dataset_name = uploaded.name
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Gagal membaca CSV: {exc}")
                    st.caption("Memakai dataset bawaan.")
        elif source == SRC_LIVE:
            max_art = st.slider(
                "Jumlah artikel diambil", 5, 25, 12,
                help="Semakin banyak artikel semakin lama proses pengunduhan.",
            )
            use_cache_idf = st.checkbox(
                "Stabilkan IDF dengan korpus cache", True,
                help="IDF dihitung dari seluruh artikel di cache agar bobot kata lebih stabil.",
            )
            refresh = st.button("🔄 Cari ulang di internet", use_container_width=True)
            st.caption("Otomatis disimpan ke `live_cache.csv`.")
        else:
            dataset_name = "live_cache.csv"

        corpus_info_slot = st.empty()

    with st.container(border=True):
        st.markdown("##### ⚖️ Bobot Posisi (PoWeR)")
        w1 = st.slider("w₁ — Bobot Judul", 0.0, 10.0, 3.0, 0.5)
        w2 = st.slider("w₂ — Bobot Intro", 0.0, 10.0, 2.0, 0.5)
        w3 = st.slider("w₃ — Bobot Isi Berita", 0.0, 10.0, 1.0, 0.5)
        if w1 + w2 + w3 == 0:
            st.warning("Minimal satu bobot harus > 0.")

    with st.container(border=True):
        st.markdown("##### 🧮 Parameter BM25")
        k1 = st.number_input("k₁ (saturasi frekuensi term)", 0.0, 5.0, 1.5, 0.1)
        b = st.number_input("b (normalisasi panjang dokumen)", 0.0, 1.0, 0.75, 0.05)

    with st.container(border=True):
        st.markdown("##### 🛡️ Penanganan Clickbait")
        cb_min = st.number_input(
            "Ambang: total TF kueri di isi ≤", 0, 20, 1, 1,
            help="Berita ditandai clickbait jika kata kueri muncul di judul, "
                 "tetapi kemunculannya di isi berita tidak melebihi ambang ini.",
        )
        cb_penalty = st.slider(
            "Faktor penalti skor clickbait", 0.1, 1.0, 0.5, 0.05,
            help="Skor berita clickbait dikalikan faktor ini agar turun peringkat. "
                 "Geser ke 1.0 untuk melihat skor murni tanpa penalti.",
        )
        top_n = st.slider("Maksimum hasil tampil", 1, 50, 10)

WEIGHTS = (w1, w2, w3)

# --------------------------------------------------------------------------
# Hero Header Utama
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-tag">Information Retrieval · News Search Engine</div>
    <div class="hero-title">📰 Sistem Temu Kembali Informasi Berita — PoWeR-BM25</div>
    <div class="hero-sub">
        Implementasi <b>Position Weighted Retrieval BM25 (PoWeR-BM25)</b> dengan
        deteksi ketidaksesuaian judul–isi dan penalti skor otomatis untuk penanganan <b>clickbait</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Area Isi Utama: Pencarian & Tab
# --------------------------------------------------------------------------
query = st.text_input(
    "🔍 Masukkan Kata Kunci Pencarian:",
    key="query_box",
    placeholder="Ketik kata kunci berita lalu tekan Enter (misal: harga cabai, banjir jakarta, kebakaran)...",
)

st.write("💡 **Kueri Cepat Pengujian:**")
q_cols = st.columns(4)
samples = (
    ["banjir jakarta", "harga cabai", "timnas indonesia", "suku bunga rupiah"]
    if source == SRC_LOCAL
    else ["harga beras", "timnas indonesia", "cuaca ekstrem", "kebakaran"]
)
for col, sample_q in zip(q_cols, samples):
    col.button(sample_q.title(), on_click=set_quick_query, args=(sample_q,), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------
# Pengambilan Berita Live (Mode Internet) & Penentuan Korpus Aktif
# --------------------------------------------------------------------------
candidate_ids = None
if source == SRC_LIVE:
    if query.strip():
        key = (query.strip().lower(), max_art)
        if refresh or st.session_state.get("live_key") != key:
            bar = st.progress(0.0, text="Mencari berita terbaru dari RSS...")

            def _progress(done: int, total: int) -> None:
                bar.progress(done / max(total, 1), text=f"Mengunduh & mengekstrak artikel {done}/{total}...")

            new_df, stats = fetch_live(query.strip(), max_art, progress=_progress)
            bar.empty()
            added = save_to_cache(new_df, CACHE_CSV)
            st.session_state.update(live_key=key, live_df=new_df, live_stats=stats, live_added=added)
        live_df = st.session_state["live_df"]
        stats = st.session_state["live_stats"]
        candidate_ids = set(live_df["id"])
        if stats.galat_pencarian:
            st.error(
                f"Gagal mengambil berita dari internet: {stats.galat_pencarian}. "
                "Periksa koneksi internet, lalu klik 'Cari ulang di internet'."
            )
        else:
            st.info(
                f"🌐 **Status Live Search:** Kandidat ditemukan: **{stats.kandidat}** · "
                f"Berhasil diekstrak: **{stats.berhasil}** · Gagal/Paywall: **{stats.gagal}** · "
                f"Baru masuk cache: **{st.session_state['live_added']}**"
            )
        df = load_cache(CACHE_CSV) if use_cache_idf else live_df
    else:
        df = load_cache(CACHE_CSV)
        st.info("Ketik kata kunci pada kolom pencarian di atas lalu tekan **Enter** untuk mencari berita terbaru secara online.")
elif source == SRC_CACHE:
    df = load_cache(CACHE_CSV)

if df is None or len(df) == 0:
    st.warning(
        "Belum ada berita pada sumber ini. Pada mode **Internet (Live Search)**, ketik kueri terlebih dahulu; "
        "pada mode **Cache**, jalankan pencarian live minimal sekali."
    )
    st.stop()

if source != SRC_LOCAL:
    df = normalize_dataframe(df)
    corpus_info_slot.caption(f"Korpus aktif: **{dataset_name or 'live cache'}** ({len(df)} berita)")
else:
    corpus_info_slot.caption(f"Korpus aktif: **{dataset_name}** ({len(df)} berita)")

index = get_index(df)


def run_search(q: str, wts, penalty: float, restrict: bool = True) -> list[dict]:
    res = index.search(q, wts, k1, b, cb_min, penalty)
    if restrict and candidate_ids is not None:
        res = [r for r in res if r["id"] in candidate_ids]
        for rank, r in enumerate(res, start=1):
            r["rank"] = rank
    return res


results: list[dict] = []
baseline: list[dict] = []
if query.strip() and w1 + w2 + w3 > 0:
    results = run_search(query, WEIGHTS, cb_penalty)
    baseline = run_search(query, BASELINE_WEIGHTS, 1.0)

tab_search, tab_analysis, tab_eval, tab_data = st.tabs(
    ["🔎 Pencarian & Hasil", "📊 Analisis & Perbandingan", "🧪 Evaluasi Metrik", "🗂️ Dataset & Preprocessing"]
)

# --------------------------------------------------------------------------
# Tab 1: Pencarian & Hasil
# --------------------------------------------------------------------------
with tab_search:
    if not query.strip():
        st.info("Masukkan kata kunci pencarian atau pilih salah satu tombol **Kueri Cepat** di atas.")
    else:
        info = preprocess_detail(query)
        q_stems = set(info["stem"])

        with st.expander("🔤 Tahapan Preprocessing Kueri (Sastrawi)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown("**1. Case Folding**")
            c1.code(info["case_folding"] or "-", language=None)
            c2.markdown("**2. Tokenisasi**")
            c2.code(", ".join(info["tokens"]) or "-", language=None)
            c3.markdown("**3. Stopword Removal**")
            c3.code(", ".join(info["tanpa_stopword"]) or "-", language=None)
            c4.markdown("**4. Stemming (Kata Dasar)**")
            c4.code(", ".join(info["stem"]) or "-", language=None)

        if not info["stem"]:
            st.warning("Kueri hanya berisi stopword atau karakter non-huruf. Gunakan kata kunci lain.")
        else:
            n_cb = sum(r["clickbait"] for r in results)
            n_rel = len(results) - n_cb

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Berita Terambil", f"{len(results)} dokumen")
            m2.metric("Berita Relevan Normal", f"{n_rel} berita")
            m3.metric("Indikasi Clickbait", f"{n_cb} berita")
            m4.metric("Skor Akhir Tertinggi", f"{results[0]['score']:.4f}" if results else "0.0000")

            st.markdown("---")

            mode = st.radio(
                "Saring Tampilan Hasil:",
                ["Tampilkan Semua", "Hanya Berita Relevan", "Hanya Indikasi Clickbait"],
                horizontal=True,
            )
            if mode == "Hanya Berita Relevan":
                filtered = [r for r in results if not r["clickbait"]]
            elif mode == "Hanya Indikasi Clickbait":
                filtered = [r for r in results if r["clickbait"]]
            else:
                filtered = results

            if not filtered:
                st.info("Tidak ada berita yang cocok dengan filter yang dipilih.")

            for r in filtered[:top_n]:
                with st.container(border=True):
                    col_t, col_s = st.columns([4.5, 1.5])
                    with col_t:
                        badge_html = (
                            '<span class="badge-cb">⚠️ Indikasi Clickbait</span>'
                            if r["clickbait"]
                            else '<span class="badge-ok">✓ Relevan</span>'
                        )
                        tf_pills = (
                            f'<span class="tf-pill">TF Judul: <b>{r["tf_judul"]}</b></span>'
                            f'<span class="tf-pill">TF Intro: <b>{r["tf_intro"]}</b></span>'
                            f'<span class="tf-pill">TF Isi: <b>{r["tf_isi"]}</b></span>'
                        )
                        st.markdown(
                            f'<div class="meta-line">Peringkat <b>#{r["rank"]}</b> · ID: <code>{r["id"]}</code> &nbsp; {badge_html} &nbsp; {tf_pills}</div>',
                            unsafe_allow_html=True,
                        )
                        hl_judul = highlight_text(r["judul"], q_stems)
                        st.markdown(f'<div class="news-title">{hl_judul}</div>', unsafe_allow_html=True)

                    with col_s:
                        st.metric("Skor Akhir", f"{r['score']:.4f}")
                        if r["clickbait"] and cb_penalty < 1.0:
                            st.caption(f"Murni: `{r['raw_score']:.4f}` (×{cb_penalty:g})")

                    if r["clickbait"]:
                        st.warning(
                            f"⚠️ **Penanganan Clickbait Aktif:** Kata kueri muncul di judul "
                            f"(TF={r['tf_judul']}), tetapi kemunculannya di isi hanya {r['tf_isi']}× "
                            f"(≤ ambang {cb_min}). Skor murni dikalikan faktor penalti **{cb_penalty:g}**."
                        )

                    meta = r.get("extra", {})
                    if meta.get("sumber") or meta.get("tanggal"):
                        st.caption(f"🗞️ Sumber: **{meta.get('sumber', '-')}** · 📅 {meta.get('tanggal', '-')}")

                    hl_intro = highlight_text(r["intro"], q_stems)
                    st.markdown(f"**Intro:** <i>{hl_intro}</i>", unsafe_allow_html=True)

                    isi = r["isi"]
                    snippet = isi[:380] + "…" if len(isi) > 380 else isi
                    hl_isi = highlight_text(snippet, q_stems)
                    st.markdown(f"**Isi Berita:** {hl_isi}", unsafe_allow_html=True)

                    if str(meta.get("url", "")).startswith("http"):
                        st.link_button("Baca Artikel Asli ↗", meta["url"])

                    with st.expander("🔬 Rincian Pembobotan Zona & Kalkulasi Skor"):
                        st.dataframe(r["detail"], hide_index=True, use_container_width=True)
                        st.caption(
                            f"Bobot aktif: w₁={w1:g} (Judul), w₂={w2:g} (Intro), w₃={w3:g} (Isi) | "
                            f"Skor Murni PoWeR-BM25 = {r['raw_score']:.4f} | "
                            f"Penalti Clickbait = {cb_penalty if r['clickbait'] else 1.0:g} | "
                            f"Skor Akhir = {r['score']:.4f}"
                        )

# --------------------------------------------------------------------------
# Tab 2: Analisis & Perbandingan
# --------------------------------------------------------------------------
with tab_analysis:
    if not results:
        st.info("Lakukan pencarian pada tab **Pencarian & Hasil** terlebih dahulu untuk melihat grafik analisis.")
    else:
        top = results[:top_n]
        base_map = {r["id"]: r for r in baseline}
        order = [f"#{r['rank']} {r['judul'][:42]}{'…' if len(r['judul']) > 42 else ''}" for r in top]

        rows = []
        for lbl, r in zip(order, top):
            rows.append({"Berita": lbl, "Metode": "PoWeR-BM25 (+ Penanganan Clickbait)", "Skor": r["score"]})
            rows.append({
                "Berita": lbl,
                "Metode": "BM25 Standar (1:1:1 Tanpa Penalti)",
                "Skor": base_map[r["id"]]["score"] if r["id"] in base_map else 0.0,
            })
        long_df = pd.DataFrame(rows)

        st.markdown("##### Grafik Perbandingan Skor: PoWeR-BM25 vs BM25 Standar")
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                y=alt.Y("Berita:N", sort=order, title=None),
                x=alt.X("Skor:Q", title="Skor Relevansi"),
                yOffset="Metode:N",
                color=alt.Color(
                    "Metode:N",
                    title="Metode",
                    scale=alt.Scale(range=["#2b547e", "#f59e0b"]),
                ),
                tooltip=["Berita", "Metode", alt.Tooltip("Skor:Q", format=".4f")],
            )
            .properties(height=max(240, 60 * len(top)))
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("##### Tabel Pergeseran Peringkat (Bahan Analisis Artikel Ilmiah)")
        cmp_rows = []
        for r in top:
            br = base_map.get(r["id"])
            cmp_rows.append({
                "ID": r["id"],
                "Judul Berita": r["judul"],
                "Rank PoWeR": r["rank"],
                "Rank BM25": br["rank"] if br else None,
                "Pergeseran": (br["rank"] - r["rank"]) if br else None,
                "Skor Murni PoWeR": round(r["raw_score"], 4),
                "Skor Akhir PoWeR": round(r["score"], 4),
                "Skor BM25 Standar": round(br["score"], 4) if br else None,
                "Status Clickbait": "⚠️ Clickbait" if r["clickbait"] else "✓ Normal",
            })
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Pergeseran positif (+) berarti peringkat naik; negatif (−) berarti peringkat turun "
            "(misalnya berita clickbait yang berhasil diturunkan oleh penalti)."
        )

# --------------------------------------------------------------------------
# Tab 3: Evaluasi Metrik (Precision@K, Recall@K, F1@K)
# --------------------------------------------------------------------------
with tab_eval:
    st.markdown("##### Evaluasi Performa Temu Kembali Informasi")
    st.caption(
        "Ukur nilai **Precision@K**, **Recall@K**, dan **F1@K** antara **PoWeR-BM25 (+ Penanganan Clickbait)** "
        "dengan **BM25 Standar (1:1:1)**."
    )
    eval_mode = st.radio("Mode Evaluasi:", ["Satu Kueri (Manual)", "Batch (Unggah CSV Penilaian)"], horizontal=True)
    k_eval = int(st.number_input("Nilai K (Top-K Dokumen)", 1, 50, 5, 1))
    configs = [
        ("PoWeR-BM25 (+ Penanganan Clickbait)", WEIGHTS, cb_penalty),
        ("BM25 Standar (1:1:1)", BASELINE_WEIGHTS, 1.0),
    ]

    if eval_mode == "Satu Kueri (Manual)":
        eq = st.text_input("Kueri Uji:", value=query, placeholder="contoh: budidaya")
        filter_gt = st.text_input(
            "🔎 Saring Kandidat Berita (cari kata kunci di judul atau isi):",
            value=eq,
            placeholder="Ketik kata kunci untuk menyaring daftar berita di bawah...",
        )

        if filter_gt.strip():
            mask = (
                df["judul"].astype(str).str.contains(filter_gt.strip(), case=False, na=False)
                | df["isi"].astype(str).str.contains(filter_gt.strip(), case=False, na=False)
            )
            df_kandidat = df[mask].head(100)
            st.caption(f"Menampilkan {len(df_kandidat)} kandidat berita (maks. 100 teratas agar ringan).")
        else:
            df_kandidat = df.head(100)
            st.caption("Menampilkan 100 berita pertama. Ketik kata kunci di atas untuk menyaring berita lain.")

        id_list = df_kandidat["id"].tolist()
        title_of = dict(zip(df_kandidat["id"], df_kandidat["judul"]))
        selected_ids = st.multiselect(
            "Pilih ID Berita yang Benar-Benar Relevan (Ground Truth):",
            id_list,
            format_func=lambda i: f"[{i}] {str(title_of.get(i, ''))[:85]}",
        )

        manual_id_text = st.text_input(
            "Atau ketik langsung nomor ID relevan (pisahkan dengan titik koma, contoh: 12; 45; 108):",
            placeholder="contoh: 12; 45; 108",
        )
        typed_ids = [x.strip() for x in manual_id_text.replace(",", ";").split(";") if x.strip()]
        relevant = list(dict.fromkeys([str(x) for x in selected_ids] + typed_ids))

        if eq.strip() and relevant and w1 + w2 + w3 > 0:
            out = []
            for name, wts, pen in configs:
                res = run_search(eq, wts, pen, restrict=False)
                m = evaluate_ranking([r["id"] for r in res], relevant, k_eval)
                out.append({
                    "Metode": name,
                    f"Precision@{k_eval}": round(m["precision"], 4),
                    f"Recall@{k_eval}": round(m["recall"], 4),
                    f"F1@{k_eval}": round(m["f1"], 4),
                })
            st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
        else:
            st.info("Masukkan kueri uji dan pilih/ketik minimal satu berita relevan untuk melihat hasil evaluasi.")
    else:
        st.markdown(
            "Format CSV penilaian: kolom `kueri` dan `relevan_id` "
            "(beberapa ID dipisah titik koma, contoh: `1;2;3;11`)."
        )
        if SAMPLE_JUDGMENT_CSV.exists():
            st.download_button(
                "⬇️ Unduh Contoh File Penilaian (CSV)",
                SAMPLE_JUDGMENT_CSV.read_bytes(),
                file_name="contoh_penilaian.csv",
                mime="text/csv",
            )
        judg_file = st.file_uploader("Unggah CSV Penilaian", type=["csv"], key="judg")
        if judg_file is not None and w1 + w2 + w3 > 0:
            try:
                jdf = pd.read_csv(judg_file, dtype=str)
                jdf.columns = [c.strip().lower() for c in jdf.columns]
                if not {"kueri", "relevan_id"} <= set(jdf.columns):
                    raise ValueError("CSV harus memiliki kolom 'kueri' dan 'relevan_id'.")
                rows = []
                for _, row in jdf.iterrows():
                    rel = [x.strip() for x in str(row["relevan_id"]).split(";") if x.strip()]
                    for name, wts, pen in configs:
                        res = run_search(row["kueri"], wts, pen, restrict=False)
                        m = evaluate_ranking([r["id"] for r in res], rel, k_eval)
                        rows.append({
                            "Kueri": row["kueri"],
                            "Metode": name,
                            f"Precision@{k_eval}": m["precision"],
                            f"Recall@{k_eval}": m["recall"],
                            f"F1@{k_eval}": m["f1"],
                        })
                res_df = pd.DataFrame(rows)
                metric_cols = [c for c in res_df.columns if "@" in c]
                st.markdown("**Rata-Rata Makro (Macro Average)**")
                st.dataframe(res_df.groupby("Metode")[metric_cols].mean().round(4), use_container_width=True)
                st.markdown("**Detail Evaluasi Per Kueri**")
                st.dataframe(res_df.round(4), hide_index=True, use_container_width=True)
                st.download_button(
                    "⬇️ Unduh Hasil Evaluasi (CSV)",
                    res_df.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_evaluasi.csv",
                    mime="text/csv",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Gagal memproses file penilaian: {exc}")
                
# --------------------------------------------------------------------------
# Tab 4: Dataset & Inspeksi Preprocessing Dokumen
# --------------------------------------------------------------------------
with tab_data:
    st.markdown("##### Ringkasan Korpus Berita Aktif")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Berita", len(df))
    m2.metric("Rata-rata Kata (Isi)", int(df["isi"].str.split().str.len().mean()))
    m3.metric("Total Term Unik", len(index.doc_freq))

    st.markdown("---")
    with st.expander("🔬 Inspeksi Tahapan Preprocessing Dokumen (Sastrawi)", expanded=True):
        st.caption(
            "Pilih salah satu dokumen berita untuk melihat bagaimana teks pada zona **Judul**, **Intro**, dan **Isi** "
            "diproses melalui tahapan *Case Folding*, *Tokenisasi*, *Stopword Removal*, hingga *Stemming Sastrawi*."
        )
        doc_ids = df["id"].tolist()
        doc_titles = dict(zip(df["id"], df["judul"]))
        selected_doc_id = st.selectbox(
            "Pilih Dokumen Berita:",
            doc_ids,
            format_func=lambda i: f"[{i}] {doc_titles[i][:95]}",
        )
        doc_row = df[df["id"] == selected_doc_id].iloc[0]

        prep_rows = []
        for zone_label, col_name in [("Judul (w₁)", "judul"), ("Intro (w₂)", "intro"), ("Isi Berita (w₃)", "isi")]:
            raw_val = str(doc_row[col_name])
            p_info = preprocess_detail(raw_val)
            prep_rows.append({
                "Zona": zone_label,
                "Teks Asli": raw_val[:140] + ("…" if len(raw_val) > 140 else ""),
                "1. Case Folding": p_info["case_folding"][:140] + ("…" if len(p_info["case_folding"]) > 140 else ""),
                "2. Tokenisasi": ", ".join(p_info["tokens"][:25]) + ("…" if len(p_info["tokens"]) > 25 else ""),
                "3. Stopword Removal": ", ".join(p_info["tanpa_stopword"][:25]) + ("…" if len(p_info["tanpa_stopword"]) > 25 else ""),
                "4. Stemming Sastrawi": ", ".join(p_info["stem"][:25]) + ("…" if len(p_info["stem"]) > 25 else ""),
            })
        st.dataframe(pd.DataFrame(prep_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Tabel Seluruh Dokumen Korpus")
        max_preview = 500
        df_preview = df.head(max_preview)
        if len(df) > max_preview:
            st.caption(f"Menampilkan pratinjau {max_preview} baris pertama dari total {len(df):,} dokumen (seluruh dokumen tetap aktif pada pencarian).")

        if "url" in df_preview.columns:
            st.dataframe(
            df_preview, hide_index=True, use_container_width=True,
            column_config={"url": st.column_config.LinkColumn("url")},
            )
        else:
            st.dataframe(df_preview, hide_index=True, use_container_width=True)