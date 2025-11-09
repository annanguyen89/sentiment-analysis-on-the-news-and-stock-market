#!/usr/bin/env python3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SLIM = DATA/"nyt_articles_slim.csv"
OUT  = DATA/"nyt_articles_with_sentiment.csv"

def main():
    df = pd.read_csv(SLIM)
    sid = SentimentIntensityAnalyzer()
    def score(row):
        txt = f"{row.get('headline','')} {row.get('abstract','')}".strip()
        return sid.polarity_scores(txt)["compound"] if isinstance(txt, str) and txt else None
    df["sentiment"] = df.apply(score, axis=1)
    df = df.dropna(subset=["et_date"]).sort_values(["et_date","article_id"])
    df.to_csv(OUT, index=False)
    print(f"[SAVE] {OUT}  rows={len(df):,}")

if __name__ == "__main__":
    main()
