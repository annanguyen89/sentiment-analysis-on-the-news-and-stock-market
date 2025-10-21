import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SLIM_CSV        = DATA_DIR / "nyt_articles_slim_2016.csv"
SENTIMENT_CSV   = DATA_DIR / "nyt_articles_sentiment_2016.csv"   
TOPICS_WIDE     = DATA_DIR / "lda_doc_topic_weights.csv"    
MAP_CSV         = DATA_DIR / "topic_to_sector_2016.csv"     
ETF_PANEL_CSV   = DATA_DIR / "sector_panel_daily_xlk_2016.csv"
OUT_DAILY_MOOD_CSV = DATA_DIR / "daily_sector_mood_2016.csv"
OUT_PILOT_DATA_CSV = DATA_DIR / "pilot_XLK_2016_model_data.csv"
PILOT_SECTOR = "XLK"
SECTOR_COLS  = ["w_XLK","w_XLE","w_XLV","w_XLF","w_XLI","w_XLY","w_XLP","w_XLB","w_XLU","w_XLRE","w_XLC"]
HAC_MAXLAGS  = 5

def _require_columns(df: pd.DataFrame, cols, name="DataFrame"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _topic_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("topic_")]

def _attach_sentiment_inline(articles_df: pd.DataFrame, sentiment_csv: Path):
    sent = pd.read_csv(sentiment_csv)

    join_key = None
    if "article_id" in articles_df.columns and "article_id" in sent.columns:
        join_key = "article_id"
    elif "url" in articles_df.columns and "url" in sent.columns:
        join_key = "url"
    else:
        raise ValueError(
            "Cannot attach sentiment: need 'article_id' (preferred) or 'url' present in BOTH "
            f"articles and {sentiment_csv.name}"
        )

    if "sentiment" not in sent.columns:
        raise ValueError(f"{sentiment_csv.name} must contain a 'sentiment' column in [-1,1].")

    sent = sent[[join_key, "sentiment"]].drop_duplicates(subset=[join_key])
    merged = articles_df.merge(sent, on=join_key, how="left")

    share_missing = merged["sentiment"].isna().mean()
    print(f"[INFO] Sentiment attached via '{join_key}'. Missing sentiment after merge: {share_missing:.1%}")
    return merged

def build_daily_sector_mood_for_pilot(
    slim_csv: Path,
    sentiment_csv: Path,
    topics_wide_csv: Path,
    mapping_csv: Path
) -> pd.DataFrame:
    # 1) Articles 
    articles = pd.read_csv(slim_csv, parse_dates=["et_date"])
    articles = _attach_sentiment_inline(articles, sentiment_csv)
    _require_columns(articles, ["article_id","et_date","sentiment"], slim_csv.name)

    # 2) Topics 
    topics_wide = pd.read_csv(topics_wide_csv)
    _require_columns(topics_wide, ["article_id"], topics_wide_csv.name)
    tcols = _topic_cols(topics_wide)
    if not tcols:
        raise ValueError(f"No topic_* columns found in {topics_wide_csv.name}")
    topics_wide = topics_wide[["article_id"] + tcols].copy()

    # 3) Topic→sector mapping 
    mapping = pd.read_csv(mapping_csv)
    _require_columns(mapping, ["topic"] + SECTOR_COLS, mapping_csv.name)

    # 4) Join articles + topics
    df = articles.merge(topics_wide, on="article_id", how="inner")
    print(f"[INFO] Joined articles + topics: {df.shape[0]} rows. et_date present? {'et_date' in df.columns}")

    # 5) Long θ_{i,k}
    long_theta = df.melt(
        id_vars=["article_id","et_date","sentiment"],
        value_vars=tcols,
        var_name="topic_col", value_name="theta"
    )
    long_theta["topic"] = long_theta["topic_col"].str.replace("topic_","", regex=False).astype(int)

    # 6) Attach mapping weights
    theta_map = long_theta.merge(mapping[["topic"] + SECTOR_COLS], on="topic", how="left").fillna(0.0)

    # 7) Long by sector; keep positive weights
    theta_long = theta_map.melt(
        id_vars=["article_id","et_date","sentiment","theta","topic"],
        value_vars=SECTOR_COLS,
        var_name="sector", value_name="w_topic_to_sector"
    )
    theta_long = theta_long[theta_long["w_topic_to_sector"] > 0].copy()

    # 8) Contributions to mood
    theta_long["num_contrib"] = theta_long["sentiment"] * theta_long["theta"] * theta_long["w_topic_to_sector"]
    theta_long["den_contrib"] = theta_long["theta"] * theta_long["w_topic_to_sector"]

    # 9) Aggregate to daily sector mood
    daily_sector = (
        theta_long.groupby(["et_date","sector"], as_index=False)
                  .agg(num=("num_contrib","sum"),
                       den=("den_contrib","sum"),
                       n_articles=("article_id","nunique"))
    )
    daily_sector["mood"] = daily_sector["num"] / daily_sector["den"]
    daily_sector.loc[daily_sector["den"] == 0, "mood"] = np.nan
    daily_sector["sector"] = daily_sector["sector"].str.replace("w_","", regex=False)

    return daily_sector[["et_date","sector","mood","n_articles"]]


def merge_with_etf_daily_for_pilot(daily_mood: pd.DataFrame, etf_panel_csv: Path) -> pd.DataFrame:
    etf = pd.read_csv(etf_panel_csv, parse_dates=["date"])
    _require_columns(etf, ["date","sector","sector_return_next_day","market_return","vix_change"], etf_panel_csv.name)
    etf = etf.rename(columns={"date": "et_date"})

    model_df = etf.merge(daily_mood, on=["et_date","sector"], how="left")
    model_df = model_df.dropna(subset=["mood"]).copy()  
    return model_df


def run_hac_ols(df: pd.DataFrame, sector: str):
    pilot = df[df["sector"] == sector].copy()
    if pilot.empty:
        print(f"[WARN] No rows for sector {sector}.")
        return None

    needed = ["sector_return_next_day", "mood", "market_return", "vix_change"]
    before = len(pilot)
    pilot = pilot.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    after = len(pilot)
    dropped = before - after
    print(f"[CLEAN] Rows before: {before}, after cleaning: {after}, dropped: {dropped}")

    if after == 0:
        print("[ERROR] No rows left after cleaning; cannot run OLS.")
        return None

    if "et_date" in pilot.columns:
        pilot = pilot.sort_values("et_date")

    X = pilot[["mood","market_return","vix_change"]]
    X = sm.add_constant(X)
    y = pilot["sector_return_next_day"]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})

    beta = model.params.get("mood", np.nan)
    sd_mood = pilot["mood"].std()
    effect_per_0p10 = beta * 0.10
    effect_per_1sd  = beta * sd_mood

    print("\n================ OLS (HAC) Summary ================")
    print(model.summary())
    print("===================================================")
    print(f"Effect per 0.10 mood: {effect_per_0p10:.6f} (log ret)  | {effect_per_0p10*10000:.2f} bps")
    print(f"Effect per 1 SD mood: {effect_per_1sd:.6f} (log ret)   | {effect_per_1sd*10000:.2f} bps")
    return model

def main():
    # 0) Check files exist
    for p in [SLIM_CSV, SENTIMENT_CSV, TOPICS_WIDE, MAP_CSV, ETF_PANEL_CSV]:
        if not os.path.exists(p):
            raise SystemExit(f"Missing required file: {p}")

    # 1) Build daily sector mood (2016)
    daily_mood = build_daily_sector_mood_for_pilot(
        slim_csv=SLIM_CSV,
        sentiment_csv=SENTIMENT_CSV,
        topics_wide_csv=TOPICS_WIDE,
        mapping_csv=MAP_CSV
    )
    daily_mood.to_csv(OUT_DAILY_MOOD_CSV, index=False)
    print(f"[SAVE] Daily sector mood → {OUT_DAILY_MOOD_CSV} (rows={len(daily_mood):,})")

    # 2) Merge with ETF returns (daily)
    model_df = merge_with_etf_daily_for_pilot(daily_mood, ETF_PANEL_CSV)

    # 3) Save XLK-only modeling data
    pilot = model_df[model_df["sector"] == PILOT_SECTOR].copy()
    pilot.to_csv(OUT_PILOT_DATA_CSV, index=False)
    print(f"[SAVE] Pilot {PILOT_SECTOR} modeling data → {OUT_PILOT_DATA_CSV} (rows={len(pilot):,})")

    # 4) Run HAC-OLS for XLK
    _ = run_hac_ols(model_df, PILOT_SECTOR)

if __name__ == "__main__":
    main()
