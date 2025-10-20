import os
import time
import json
import csv
import sqlite3
from datetime import datetime
import requests
import pandas as pd
import hashlib

API_BASE = "https://api.nytimes.com/svc/archive/v1"
DB_PATH   = "nyt_archive.db"
FULL_CSV  = "nyt_articles_full.csv"
SLIM_CSV  = "nyt_articles_slim.csv"
DAILY_CSV = "nyt_daily_counts.csv"
START_YEAR, START_MONTH = 2016, 1
END_YEAR,   END_MONTH   = 2025, 9
SLEEP_BETWEEN_CALLS = 1.5 
API_KEY = os.getenv("NYT_API_KEY")

def ensure_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT PRIMARY KEY,
        uri TEXT,
        section TEXT,
        subsection TEXT,
        headline TEXT,
        abstract TEXT,
        byline TEXT,
        item_type TEXT,
        published_at TEXT,
        updated_at TEXT,
        source TEXT,
        news_desk TEXT,
        type_of_material TEXT,
        document_type TEXT,
        section_name TEXT,
        subsection_name TEXT,
        keywords TEXT,      -- JSON list
        raw_json TEXT       -- full record
    )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_pub ON articles(published_at)")
    con.commit()
    return con

def month_iter(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m = 1 if m == 12 else m + 1
        y = y + 1 if m == 1 else y

def fetch_archive(year: int, month: int):
    url = f"{API_BASE}/{year}/{month}.json"
    params = {"api-key": API_KEY}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[{year}-{month:02d}] 429 Too Many Requests. Sleeping {wait}s…")
                time.sleep(wait); continue
            r.raise_for_status()
            data = r.json()
            docs = (data.get("response") or {}).get("docs", [])
            out = []
            for d in docs:
                if d.get("document_type") != "article":
                    continue
                if d.get("lang") and d.get("lang") != "en":
                    continue
                out.append(d)
            return out
        except requests.RequestException as e:
            wait = 3 * (attempt + 1)
            print(f"[{year}-{month:02d}] Error: {e}. Retry in {wait}s.")
            time.sleep(wait)
    print(f"[{year}-{month:02d}] Failed after retries.")
    return []

def upsert(con: sqlite3.Connection, doc: dict) -> int:
    url = doc.get("web_url") or doc.get("url")
    if not url:
        return 0
    headline = (doc.get("headline") or {}).get("main") if isinstance(doc.get("headline"), dict) else doc.get("title")
    byline   = (doc.get("byline")   or {}).get("original") if isinstance(doc.get("byline"), dict) else doc.get("byline")
    published_at = doc.get("pub_date") or doc.get("published_date")
    updated_at   = doc.get("updated_date") or doc.get("update_date")
    section      = doc.get("section_name") or doc.get("section")
    subsection   = doc.get("subsection_name") or doc.get("subsection")
    keywords     = doc.get("keywords") if isinstance(doc.get("keywords"), list) else None

    row = {
        "url": url,
        "uri": doc.get("uri"),
        "section": section,
        "subsection": subsection,
        "headline": headline,
        "abstract": doc.get("abstract"),
        "byline": byline,
        "item_type": doc.get("type_of_material") or doc.get("item_type"),
        "published_at": published_at,
        "updated_at": updated_at,
        "source": doc.get("source"),
        "news_desk": doc.get("news_desk"),
        "type_of_material": doc.get("type_of_material"),
        "document_type": doc.get("document_type"),
        "section_name": doc.get("section_name"),
        "subsection_name": doc.get("subsection_name"),
        "keywords": json.dumps(keywords, ensure_ascii=False) if keywords is not None else None,
        "raw_json": json.dumps(doc, ensure_ascii=False),
    }

    try:
        con.execute("""
            INSERT INTO articles (
                url, uri, section, subsection, headline, abstract, byline, item_type,
                published_at, updated_at, source, news_desk, type_of_material,
                document_type, section_name, subsection_name, keywords, raw_json
            ) VALUES (
                :url, :uri, :section, :subsection, :headline, :abstract, :byline, :item_type,
                :published_at, :updated_at, :source, :news_desk, :type_of_material,
                :document_type, :section_name, :subsection_name, :keywords, :raw_json
            )
            ON CONFLICT(url) DO UPDATE SET
                uri=excluded.uri,
                section=excluded.section,
                subsection=excluded.subsection,
                headline=excluded.headline,
                abstract=excluded.abstract,
                byline=excluded.byline,
                item_type=excluded.item_type,
                published_at=excluded.published_at,
                updated_at=excluded.updated_at,
                source=excluded.source,
                news_desk=excluded.news_desk,
                type_of_material=excluded.type_of_material,
                document_type=excluded.document_type,
                section_name=excluded.section_name,
                subsection_name=excluded.subsection_name,
                keywords=excluded.keywords,
                raw_json=excluded.raw_json
        """, row)
        return 1
    except sqlite3.DatabaseError as e:
        print(f"SQLite error for URL={url}: {e}")
        return 0

def export_sqlite_to_csv(db_path: str, csv_path: str, table: str = "articles"):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in cur:
            w.writerow(row)
    con.close()
    print(f"[EXPORT] Full CSV written to {csv_path}")

def make_article_id(u: str) -> str:
    return hashlib.sha256(str(u).encode("utf-8")).hexdigest()

def polish_and_aggregate(full_csv: str, slim_csv: str, daily_csv: str):
    """Create slim per-article CSV and daily counts (ET)."""
    df = pd.read_csv(full_csv)
    ts = pd.to_datetime(df["published_at"], errors="coerce", utc=True)  
    et = ts.dt.tz_convert("America/New_York")
    df["et_date"] = et.dt.date
    df["article_id"] = df["url"].astype(str).apply(make_article_id)

    keep_cols = [
        "article_id", "url", "headline", "abstract",
        "section_name", "subsection_name",
        "published_at", "et_date"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    slim = (
        df.drop_duplicates(subset=["url"])
          [keep_cols]
          .sort_values(["et_date", "article_id"])
          .reset_index(drop=True)
    )
    slim.to_csv(slim_csv, index=False)
    print(f"[EXPORT] Slim per-article CSV written to {slim_csv}")

    daily = (
        slim.dropna(subset=["et_date"])
            .groupby("et_date")
            .agg(n_articles=("url", "nunique"))
            .reset_index()
            .sort_values("et_date")
    )
    daily.to_csv(daily_csv, index=False)
    print(f"[EXPORT] Daily counts CSV written to {daily_csv}")

def main():
    con = ensure_db(DB_PATH)
    total_rows = 0
    for y, m in month_iter(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        print(f"Fetching {y}-{m:02d} …")
        docs = fetch_archive(y, m)
        inserted = 0
        for d in docs:
            inserted += upsert(con, d)
        con.commit()
        total_rows += inserted
        print(f"  Inserted/updated this month: {inserted} (cumulative: {total_rows})")
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"Done. Total rows upserted: {total_rows}")
    export_sqlite_to_csv(DB_PATH, FULL_CSV)
    polish_and_aggregate(FULL_CSV, SLIM_CSV, DAILY_CSV)

if __name__ == "__main__":
    main()
