"""
engine.py - Mesin temu kembali informasi PoWeR-BM25
(Position Weighted Retrieval BM25) dengan penanganan clickbait.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

ZONES = ("judul", "intro", "isi")
BASELINE_WEIGHTS = (1.0, 1.0, 1.0)

_STOPWORDS = frozenset(StopWordRemoverFactory().get_stop_words())
_STEMMER = None


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        _STEMMER = StemmerFactory().create_stemmer()
    return _STEMMER


@lru_cache(maxsize=200_000)
def stem_word(word: str) -> str:
    return _get_stemmer().stem(word)


def preprocess_detail(text: str) -> dict:
    """Pipeline: case folding -> tokenisasi -> stopword removal -> stemming."""
    folded = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", folded)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    no_stop = [t for t in tokens if t not in _STOPWORDS]
    stems = [s for s in (stem_word(t) for t in no_stop) if s]
    return {
        "case_folding": " ".join(cleaned.split()),
        "tokens": tokens,
        "tanpa_stopword": no_stop,
        "stem": stems,
    }


def preprocess(text: str) -> list[str]:
    return preprocess_detail(text)["stem"]


# --------------------------------------------------------------------------
# Normalisasi Dataset
# --------------------------------------------------------------------------
def _make_intro(isi: str, n_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", str(isi).strip())
    return " ".join(sentences[:n_sentences])


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom: id, judul, intro, isi. Intro dibuat otomatis jika tidak ada."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "title": "judul", "headline": "judul",
        "content": "isi", "konten": "isi", "body": "isi", "teks": "isi",
        "lead": "intro", "ringkasan": "intro", "paragraf_pertama": "intro",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})

    missing = [c for c in ("judul", "isi") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {', '.join(missing)}. "
            "CSV harus memiliki kolom: judul, isi (opsional: id, intro)."
        )

    df["judul"] = df["judul"].fillna("").astype(str)
    df["isi"] = df["isi"].fillna("").astype(str)
    if "intro" not in df.columns:
        df["intro"] = df["isi"].map(_make_intro)
    df["intro"] = df["intro"].fillna("").astype(str)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    df["id"] = df["id"].astype(str)

    first = ["id", "judul", "intro", "isi"]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest].reset_index(drop=True)


# --------------------------------------------------------------------------
# Indeks & Pencarian PoWeR-BM25 + Penanganan Clickbait
# --------------------------------------------------------------------------
class PoWeRBM25:
    def __init__(self, df: pd.DataFrame):
        self.df = normalize_dataframe(df)
        self.n = len(self.df)
        self.extra_cols = [c for c in self.df.columns if c not in ("id", "judul", "intro", "isi")]
        if self.n == 0:
            raise ValueError("Dataset kosong.")

        self.zone_tf: list[dict[str, Counter]] = []
        doc_freq: Counter = Counter()
        for judul, intro, isi in zip(self.df["judul"], self.df["intro"], self.df["isi"]):
            z = {
                "judul": Counter(preprocess(judul)),
                "intro": Counter(preprocess(intro)),
                "isi": Counter(preprocess(isi)),
            }
            self.zone_tf.append(z)
            doc_freq.update(set().union(*z.values()))
        self.doc_freq = doc_freq
        self.zone_len = np.array(
            [[sum(z[zn].values()) for zn in ZONES] for z in self.zone_tf], dtype=float
        )

    def idf(self, term: str) -> float:
        df_t = self.doc_freq.get(term, 0)
        return math.log((self.n - df_t + 0.5) / (df_t + 0.5) + 1.0)

    def search(
        self,
        query: str,
        weights=(3.0, 2.0, 1.0),
        k1: float = 1.5,
        b: float = 0.75,
        clickbait_min_tf: int = 1,
        clickbait_penalty: float = 0.5,
    ) -> list[dict]:
        """Kembalikan dokumen ber-skor > 0, terurut dari skor akhir tertinggi.

        Penanganan clickbait:
        Jika kata kueri muncul di judul tetapi total kemunculannya di isi <= `clickbait_min_tf`,
        dokumen ditandai clickbait dan skor murninya dikalikan `clickbait_penalty`.
        """
        q_terms = list(dict.fromkeys(preprocess(query)))
        w = np.asarray(weights, dtype=float)
        dl = self.zone_len @ w
        avgdl = float(dl.mean()) if dl.mean() > 0 else 1.0
        idf = {t: self.idf(t) for t in q_terms}

        results = []
        for i, z in enumerate(self.zone_tf):
            raw_score = 0.0
            rows = []
            for t in q_terms:
                tfs = np.array([z[zn].get(t, 0) for zn in ZONES], dtype=float)
                tf_w = float(w @ tfs)
                denom = tf_w + k1 * (1 - b + b * dl[i] / avgdl)
                contrib = idf[t] * tf_w * (k1 + 1) / denom if denom > 0 else 0.0
                raw_score += contrib
                rows.append({
                    "Istilah (stem)": t,
                    "DF": self.doc_freq.get(t, 0),
                    "IDF": round(idf[t], 4),
                    "TF Judul": int(tfs[0]),
                    "TF Intro": int(tfs[1]),
                    "TF Isi": int(tfs[2]),
                    "TF Terbobot": round(tf_w, 3),
                    "Kontribusi Skor": round(contrib, 4),
                })
            if raw_score <= 0:
                continue

            detail = pd.DataFrame(rows)
            title_hits = int((detail["TF Judul"] > 0).sum())
            body_tf = int(detail["TF Isi"].sum())
            clickbait = title_hits > 0 and body_tf <= clickbait_min_tf

            final_score = raw_score * float(clickbait_penalty) if clickbait else raw_score

            results.append({
                "id": self.df.at[i, "id"],
                "judul": self.df.at[i, "judul"],
                "intro": self.df.at[i, "intro"],
                "isi": self.df.at[i, "isi"],
                "raw_score": raw_score,
                "score": final_score,
                "clickbait": clickbait,
                "tf_judul": int(detail["TF Judul"].sum()),
                "tf_intro": int(detail["TF Intro"].sum()),
                "tf_isi": body_tf,
                "detail": detail,
                "extra": {c: self.df.at[i, c] for c in self.extra_cols},
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        for rank, r in enumerate(results, start=1):
            r["rank"] = rank
        return results


# --------------------------------------------------------------------------
# Metrik Evaluasi
# --------------------------------------------------------------------------
def evaluate_ranking(ranked_ids: list[str], relevant_ids, k: int) -> dict:
    """Precision@K, Recall@K, dan F1@K untuk satu kueri."""
    relevant = {str(x) for x in relevant_ids}
    topk = [str(x) for x in ranked_ids[:k]]
    hits = len(set(topk) & relevant)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": hits}