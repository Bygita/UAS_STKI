# Sistem Temu Kembali Informasi Berita — PoWeR-BM25

Aplikasi web (Streamlit) untuk pencarian berita berbahasa Indonesia dengan
**Position Weighted Retrieval BM25 (PoWeR-BM25)** dan deteksi indikasi **clickbait**.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur file

| File | Fungsi |
|---|---|
| `engine.py` | Preprocessing (case folding, stopword, stemming Sastrawi), indeks, skor PoWeR-BM25, deteksi clickbait, metrik evaluasi |
| `app.py` | Antarmuka Streamlit (4 tab) |
| `dataset_berita.csv` | Dataset contoh (14 berita, termasuk 4 contoh clickbait) |
| `contoh_penilaian.csv` | Contoh kunci jawaban untuk modul evaluasi batch |

## Rumus

Setiap berita dibagi menjadi zona **Judul**, **Intro**, **Isi**.

```
TF_w(t,d) = w1·TF_judul + w2·TF_intro + w3·TF_isi
|d|_w     = w1·|judul| + w2·|intro| + w3·|isi|
IDF(t)    = ln( (N − df + 0.5) / (df + 0.5) + 1 )

skor(q,d) = Σ_t  IDF(t) · TF_w · (k1+1) / ( TF_w + k1·(1 − b + b·|d|_w / avgdl_w) )
```

Dengan `w1 = w2 = w3 = 1`, hasilnya setara BM25 standar (dipakai sebagai baseline
di tab Analisis dan Evaluasi).

## Deteksi clickbait

Berita diberi badge `⚠️ Indikasi Clickbait` bila **kata kueri muncul di judul**, tetapi
**total kemunculannya di isi ≤ ambang** (default 1, bisa diubah di sidebar).

## Format dataset CSV

Kolom wajib: `judul`, `isi`. Kolom opsional: `id`, `intro`.
Jika `intro` tidak ada, aplikasi mengambil 2 kalimat pertama `isi` sebagai intro.

## Format file penilaian (evaluasi batch)

```csv
kueri,relevan_id
banjir jakarta,1;3;11
harga cabai,4;5
```

## Saran skenario uji untuk artikel

1. Kueri `harga cabai`, bobot 1:1:1 → berita clickbait (ID 6) berada di peringkat 4 karena judulnya cocok.
2. Naikkan `w1` → ID 6 naik ke peringkat 2 (judul dibobot lebih besar), dan badge clickbait tetap menandainya. Amati perubahannya di tab **Analisis**. Hasil ini menunjukkan bahwa menaikkan bobot judul saja menguntungkan clickbait, sehingga perlu dikombinasikan dengan bobot isi (`w3`) yang lebih besar atau penalti clickbait.
3. Bandingkan Precision@K / Recall@K / F1@K antara PoWeR-BM25 dan BM25 standar di tab **Evaluasi**.
4. Ulangi dengan variasi `k1` (mis. 1.2, 1.5, 2.0) dan `b` (mis. 0.5, 0.75, 1.0).
