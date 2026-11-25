"""
kumpulkan_berita_asli.py
Mengunduh berita asli bersumber dari portal berita Indonesia via fetcher.py
dan menyimpannya ke dataset_berita_asli.csv.
"""
import pandas as pd
from fetcher import fetch_live

daftar_kueri = [
    "banjir jakarta",
    "harga cabai",
    "timnas indonesia",
    "suku bunga rupiah",
    "kebakaran hutan",
    "gempa bumi",
]

semua_df = []
for q in daftar_kueri:
    print(f"Mengambil berita asli untuk kueri: '{q}'...")
    df_hasil, stats = fetch_live(q, max_articles=12)
    print(f"  -> Berhasil: {stats.berhasil} artikel")
    if len(df_hasil) > 0:
        semua_df.append(df_hasil)

if semua_df:
    gabungan = pd.concat(semua_df, ignore_index=True)
    # Hapus duplikat berdasarkan judul atau URL
    gabungan = gabungan.drop_duplicates(subset=["judul"]).reset_index(drop=True)
    gabungan["id"] = range(1, len(gabungan) + 1)
    gabungan.to_csv("dataset_berita_asli.csv", index=False, encoding="utf-8")
    print(f"\nSelesai! Total {len(gabungan)} berita asli tersimpan di 'dataset_berita_asli.csv'.")