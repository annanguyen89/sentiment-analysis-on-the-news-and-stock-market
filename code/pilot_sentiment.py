#!/usr/bin/env python3
# code/score_vader.py
import sys
import argparse
from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Default data directory: one level up from this file, then "data"
DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()

def resolve_in_path(name_or_path: str) -> Path:
    """If user passes a bare filename, read it from DATA_DIR; else use as-is."""
    p = Path(name_or_path)
    return (DATA_DIR / p) if not p.is_absolute() and p.parent == Path('.') else p

def resolve_out_path(name_or_path: str) -> Path:
    """If user passes a bare filename, write it to DATA_DIR; else use as-is."""
    p = Path(name_or_path)
    if not p.is_absolute() and p.parent == Path('.'):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return DATA_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def add_vader_sentiment(in_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)
    analyzer = SentimentIntensityAnalyzer()

    def text_sent(row):
        head = row.get("headline", "")
        abst = row.get("abstract", "")
        txt = f"{head if isinstance(head, str) else ''} {abst if isinstance(abst, str) else ''}".strip()
        if not txt:
            return None
        return analyzer.polarity_scores(txt)["compound"]

    df["sentiment"] = df.apply(text_sent, axis=1)

    keep = ["et_date", "article_id", "url", "headline", "abstract", "sentiment"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    if "et_date" in out.columns:
        out["et_date"] = pd.to_datetime(out["et_date"], errors="coerce")
        sort_cols = ["et_date"] + (["article_id"] if "article_id" in out.columns else [])
        out = out.dropna(subset=["et_date"]).sort_values(sort_cols)

    out.to_csv(out_csv, index=False)
    n_scored = int(out["sentiment"].notna().sum()) if "sentiment" in out.columns else 0
    print(f"Wrote {out_csv} with {n_scored:,} scored articles.")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score NYT articles with VADER sentiment.")
    p.add_argument(
        "--in",
        dest="in_csv",
        default="nyt_articles_slim_2016.csv",
        help="Input CSV (default is looked up in ../data/)"
    )
    p.add_argument(
        "--out",
        dest="out_csv",
        default="nyt_articles_sentiment_2016.csv",
        help="Output CSV (default is written to ../data/)"
    )
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    in_path = resolve_in_path(args.in_csv)
    out_path = resolve_out_path(args.out_csv)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    print(f"Reading:  {in_path}")
    print(f"Writing:  {out_path}")
    add_vader_sentiment(in_path, out_path)

if __name__ == "__main__":
    main()
