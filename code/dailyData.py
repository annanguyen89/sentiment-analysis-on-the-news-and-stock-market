import argparse
from pathlib import Path
import pandas as pd
import json

def make_daily_articles_from_slim(
    slim_csv: str = "nyt_articles_slim.csv",
    daily_articles_csv: str = "nyt_daily_articles.csv",
    daily_articles_json: str = None  
):
    # 1) Load slim CSV
    df = pd.read_csv(slim_csv)

    # 2) Ensure et_date exists and is usable
    if "et_date" not in df.columns:
        raise ValueError("Column 'et_date' not found in slim CSV.")
    df["et_date"] = pd.to_datetime(df["et_date"], errors="coerce").dt.date.astype(str)

    # 3) Select columns to keep
    keep = [
        "et_date", "article_id", "url",
        "headline", "abstract",
        "section_name", "subsection_name"
    ]
    keep = [c for c in keep if c in df.columns]
    out = (
        df[keep]
        .dropna(subset=["et_date", "url"])
        .drop_duplicates(subset=["url"])
        .sort_values(["et_date", "article_id"])
        .reset_index(drop=True)
    )

    # 4) Write CSV
    Path(daily_articles_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(daily_articles_csv, index=False)
    print(f"[EXPORT] Daily articles CSV written to {daily_articles_csv} (rows: {len(out):,})")

    # 5) JSON grouped by date
    if daily_articles_json:
        grouped = (
            out.groupby("et_date", sort=True)
               .apply(lambda g: g.drop(columns=["et_date"]).to_dict(orient="records"))
               .to_dict()
        )
        Path(daily_articles_json).parent.mkdir(parents=True, exist_ok=True)
        Path(daily_articles_json).write_text(
            json.dumps(grouped, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[EXPORT] Daily articles JSON written to {daily_articles_json} (days: {len(grouped):,})")

def main():
    parser = argparse.ArgumentParser(description="Build daily article list from slim NYT CSV.")
    parser.add_argument("--slim-csv", default="nyt_articles_slim.csv",
                        help="Input slim CSV (one row per article).")
    parser.add_argument("--out-csv", default="nyt_daily_articles.csv",
                        help="Output CSV with date/headline/abstract/URL per article.")
    parser.add_argument("--out-json", default=None,
                        help="Optional JSON grouped by date (set a filepath to enable).")
    args = parser.parse_args()

    make_daily_articles_from_slim(
        slim_csv=args.slim_csv,
        daily_articles_csv=args.out_csv,
        daily_articles_json=args.out_json,
    )

if __name__ == "__main__":
    main()
