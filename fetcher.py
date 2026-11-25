"""
fetcher.py - Pengambil berita live dari internet (Bing News + Google News RSS)
dengan ekstraksi artikel cerdas (Trafilatura + JSON-LD + HTML Fallback).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, quote_plus, urlparse

import pandas as pd
import requests

try:
    import trafilatura
except Exception:  # noqa: BLE001
    trafilatura = None

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:  # noqa: BLE001
    gnewsdecoder = None

from engine import _make_intro

GNEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id"
BING_NEWS_RSS = "https://www.bing.com/news/search?q={q}&format=rss&mkt=id-ID"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
CACHE_COLUMNS = ["id", "judul", "intro", "isi", "sumber", "url", "tanggal", "kueri", "diambil_pada"]
MIN_BODY_CHARS = 120
MAX_BODY_CHARS = 20_000

_decode_lock = Lock()


@dataclass
class FetchStats:
    kandidat: int = 0
    berhasil: int = 0
    gagal: int = 0
    gagal_decode: int = 0
    terlalu_pendek: int = 0
    galat_pencarian: str = ""
    alasan_gagal: list[str] = field(default_factory=list)


def make_id(url: str) -> str:
    return "L" + hashlib.md5(url.encode("utf-8")).hexdigest()[:10]


def _clean_title(title: str, source: str = "") -> str:
    title = html.unescape(title).strip()
    if source and title.endswith(f" - {source}"):
        title = title[: -(len(source) + 3)].rstrip()
    return title


def _clean_html_tags(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(raw or ""))
    return " ".join(text.split())


# --------------------------------------------------------------------------
# 1. Pencarian Kandidat Berita (Bing News RSS + Google News RSS)
# --------------------------------------------------------------------------
def search_bing_news(
    query: str,
    max_items: int = 15,
    session: requests.Session | None = None,
    timeout: float = 12,
) -> list[dict]:
    """Ambil kandidat berita dari Bing News RSS (tautan langsung tanpa enkripsi)."""
    sess = session or requests
    url = BING_NEWS_RSS.format(q=quote_plus(query))
    resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    items = []
    for it in root.iter("item"):
        raw_link = (it.findtext("link") or "").strip()
        if not raw_link:
            continue
        parsed = urlparse(raw_link)
        if "bing.com" in parsed.netloc:
            qs = parse_qs(parsed.query)
            raw_link = qs["url"][0] if qs.get("url") else raw_link

        source = ""
        for child in it:
            if child.tag.endswith("Source") and child.text:
                source = child.text.strip()
                break
        if not source:
            source = urlparse(raw_link).netloc.replace("www.", "")

        pub = it.findtext("pubDate") or ""
        try:
            tanggal = parsedate_to_datetime(pub).strftime("%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            tanggal = pub

        items.append({
            "judul": _clean_title(it.findtext("title") or "", source),
            "link": raw_link,
            "sumber": source,
            "tanggal": tanggal,
            "ringkasan": _clean_html_tags(it.findtext("description") or ""),
        })
        if len(items) >= max_items:
            break
    return items


def search_google_news(
    query: str,
    max_items: int = 15,
    session: requests.Session | None = None,
    timeout: float = 12,
) -> list[dict]:
    """Ambil kandidat berita dari Google News RSS."""
    sess = session or requests
    url = GNEWS_RSS.format(q=quote_plus(query))
    resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    items = []
    for it in root.iter("item"):
        source_el = it.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        link = (it.findtext("link") or "").strip()
        if not link:
            continue
        pub = it.findtext("pubDate") or ""
        try:
            tanggal = parsedate_to_datetime(pub).strftime("%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            tanggal = pub

        items.append({
            "judul": _clean_title(it.findtext("title") or "", source),
            "link": link,
            "sumber": source,
            "tanggal": tanggal,
            "ringkasan": _clean_html_tags(it.findtext("description") or ""),
        })
        if len(items) >= max_items:
            break
    return items


def resolve_link(link: str) -> str:
    """Ubah link redirect Google News menjadi URL artikel asli."""
    if "news.google.com" not in link:
        return link
    if gnewsdecoder is not None:
        with _decode_lock:
            try:
                res = gnewsdecoder(link, interval=0.3)
                if isinstance(res, dict) and res.get("status") and res.get("decoded_url"):
                    return res["decoded_url"]
            except Exception:  # noqa: BLE001
                pass
    return link


# --------------------------------------------------------------------------
# 2. Ekstraksi Konten Artikel (Trafilatura + JSON-LD + Paragraf HTML)
# --------------------------------------------------------------------------
class _ParagraphExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "noscript", "form", "figure"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_p = False
        self.buf: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag == "p" and not self.skip_depth:
            self.in_p = True
            self.buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "p" and self.in_p:
            text = " ".join("".join(self.buf).split())
            if len(text) >= 35:
                self.paragraphs.append(text)
            self.in_p = False

    def handle_data(self, data):
        if self.in_p and not self.skip_depth:
            self.buf.append(data)


def _extract_json_ld_body(page_html: str) -> str:
    """Ekstrak teks penuh dari schema.org JSON-LD (umum di portal berita Indonesia)."""
    scripts = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        page_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    best = ""

    def _walk(obj):
        nonlocal best
        if isinstance(obj, dict):
            body = obj.get("articleBody")
            if isinstance(body, str) and len(body) > len(best):
                best = body
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for raw in scripts:
        try:
            data = json.loads(raw.strip())
            _walk(data)
        except Exception:  # noqa: BLE001
            continue
    return html.unescape(best)


def extract_text(page_html: str, url: str = "") -> str:
    candidates = []

    # 1. Trafilatura
    if trafilatura is not None:
        try:
            t_text = trafilatura.extract(
                page_html,
                url=url or None,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
            ) or ""
            if t_text:
                candidates.append(t_text)
        except Exception:  # noqa: BLE001
            pass

    # 2. JSON-LD articleBody
    ld_text = _extract_json_ld_body(page_html)
    if ld_text:
        candidates.append(ld_text)

    # 3. HTML <p> extractor
    parser = _ParagraphExtractor()
    try:
        parser.feed(page_html)
        if parser.paragraphs:
            candidates.append(" ".join(parser.paragraphs))
    except Exception:  # noqa: BLE001
        pass

    best_text = max(candidates, key=len) if candidates else ""
    best_text = re.sub(r"\s*\n\s*", " ", best_text)
    return " ".join(best_text.split())[:MAX_BODY_CHARS]


def _download_one(cand: dict, session: requests.Session, timeout: float, delay: float):
    time.sleep(delay)
    url = resolve_link(cand["link"])
    if "news.google.com" in url:
        return None, "decode"
    try:
        resp = session.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.google.com/",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        body = extract_text(resp.text, resp.url)
    except Exception as exc:  # noqa: BLE001
        return None, f"{urlparse(url).netloc or url}: {type(exc).__name__}"

    if len(body) < MIN_BODY_CHARS:
        return None, "pendek"

    final_url = resp.url
    return {
        "id": make_id(final_url),
        "judul": cand["judul"],
        "intro": _make_intro(body),
        "isi": body,
        "sumber": cand["sumber"],
        "url": final_url,
        "tanggal": cand["tanggal"],
    }, None


# --------------------------------------------------------------------------
# 3. Fungsi Utama Live Search
# --------------------------------------------------------------------------
def fetch_live(
    query: str,
    max_articles: int = 12,
    workers: int = 5,
    timeout: float = 12,
    delay: float = 0.2,
    progress=None,
) -> tuple[pd.DataFrame, FetchStats]:
    stats = FetchStats()
    session = requests.Session()
    cands: list[dict] = []
    errors: list[str] = []

    # Sumber 1: Bing News RSS (tautan portal berita langsung)
    try:
        cands.extend(search_bing_news(query, max_items=max_articles + 5, session=session))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Bing: {type(exc).__name__}")

    # Sumber 2: Google News RSS (pelengkap kandidat berita)
    try:
        cands.extend(search_google_news(query, max_items=max_articles + 5, session=session))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Google: {type(exc).__name__}")

    if not cands and errors:
        stats.galat_pencarian = " | ".join(errors)
        return pd.DataFrame(columns=CACHE_COLUMNS), stats

    # Hilangkan duplikat judul
    seen, uniq = set(), []
    for c in cands:
        key = c["judul"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    uniq = uniq[:max_articles]
    stats.kandidat = len(uniq)

    records = []
    if uniq:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_download_one, c, session, timeout, delay) for c in uniq]
            for done, fut in enumerate(as_completed(futures), start=1):
                rec, why = fut.result()
                if rec:
                    records.append(rec)
                    stats.berhasil += 1
                else:
                    stats.gagal += 1
                    if why == "decode":
                        stats.gagal_decode += 1
                    elif why == "pendek":
                        stats.terlalu_pendek += 1
                    elif why:
                        stats.alasan_gagal.append(why)
                if progress:
                    progress(done, len(uniq))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame(records, columns=CACHE_COLUMNS[:7])
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df["kueri"] = query
    df["diambil_pada"] = now
    return df[CACHE_COLUMNS], stats


# --------------------------------------------------------------------------
# 4. Cache CSV
# --------------------------------------------------------------------------
def load_cache(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=CACHE_COLUMNS)
    for col in CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CACHE_COLUMNS]


def save_to_cache(df: pd.DataFrame, path: str | Path) -> int:
    if df is None or df.empty:
        return 0
    old = load_cache(path)
    merged = pd.concat([old, df[CACHE_COLUMNS].astype(str)], ignore_index=True)
    merged = merged.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    merged.to_csv(path, index=False, encoding="utf-8")
    return len(merged) - len(old)