import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

# NYT articles 
nyt = pd.read_csv(DATA/"nyt_daily_articles.csv")  
nyt["et_date"] = pd.to_datetime(nyt["et_date"], errors="coerce")
nyt2016 = nyt[(nyt["et_date"].dt.year == 2016)].copy()
nyt2016.to_csv("nyt_articles_slim_2016.csv", index=False)
print(f"NYT 2016 rows: {len(nyt2016):,}")

# ETF panel 
etf = pd.read_csv(DATA/"sector_panel_daily_long.csv", parse_dates=["date"])
xlk16 = etf[(etf["sector"]=="XLK") & (etf["date"].dt.year==2016)].copy()
xlk16 = xlk16.sort_values("date").reset_index(drop=True)
xlk16.to_csv("sector_panel_daily_xlk_2016.csv", index=False)
print(f"XLK 2016 trading days: {xlk16['date'].nunique():,}")
