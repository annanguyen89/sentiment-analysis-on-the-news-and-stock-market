# code/mood_by_ticker.py
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
ART_SENT = DATA/"nyt_articles_with_sentiment.csv"     # from articleSentiment.py
FIELDS   = DATA/"nyt_articles_fields.csv"             # from field_tagging.py
OUT      = DATA/"daily_ticker_mood.csv"

FIELD_TO_TICKERS = {
    "Semis/AI": ["NVDA","AMD","AVGO"],
    "EV/Auto": ["TSLA","NIO","GM","F"],
    "Mega-cap Tech": ["AAPL","MSFT","AMZN","META","GOOGL"],
    "Biotech": ["MRK","PFE","MRNA","BIIB"],
    "Banks": ["JPM","BAC","GS"],
}

def main():
    sent = pd.read_csv(ART_SENT, parse_dates=["et_date"])
    fields = pd.read_csv(FIELDS, parse_dates=["et_date"])
    # keep only articles with a sentiment value
    sent = sent.dropna(subset=["sentiment"])[["article_id","et_date","sentiment"]]
    df = fields.merge(sent, on=["article_id","et_date"], how="inner")
    # explode field to its tickers
    rows = []
    for _, r in df.iterrows():
        for t in FIELD_TO_TICKERS.get(r["field"], []):
            rows.append({"et_date": r["et_date"], "ticker": t, "mood": r["sentiment"]})
    out = pd.DataFrame(rows)
    if out.empty:
        print("[WARN] No rows produced. Check upstream files.")
    agg = (out.groupby(["et_date","ticker"])
              .agg(mood_mean=("mood","mean"), n_articles=("mood","size"))
              .reset_index())
    agg.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[SAVE] {OUT} rows={len(agg):,}")

if __name__ == "__main__":
    main()
