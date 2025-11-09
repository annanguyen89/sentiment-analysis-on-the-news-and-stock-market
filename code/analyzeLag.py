import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
PANEL = DATA / "stock_mood_returns_panel.csv"

MIN_PAIRS = 20  # minimum non-missing pairs to compute a correlation

def pick_return_col(df: pd.DataFrame) -> str:
    # Try common names from your pipeline
    candidates = [
        "stock_return_same_day",  # preferred if present
        "ret", "return", "daily_return", "r"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the return columns found: {candidates}")

def safe_corr(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < MIN_PAIRS:
        return np.nan
    return z.iloc[:,0].corr(z.iloc[:,1])

def main():
    df = pd.read_csv(PANEL, parse_dates=["et_date"])
    # Required columns
    if "ticker" not in df.columns:
        raise KeyError("Missing 'ticker' in panel.")
    if "mood_mean" not in df.columns:
        # fallback to a generic name if you used something else
        mood_cols = [c for c in df.columns if c.startswith("mood")]
        if not mood_cols:
            raise KeyError("Missing mood column (expected 'mood_mean' or a column starting with 'mood').")
        df = df.rename(columns={mood_cols[0]: "mood_mean"})

    ret_col = pick_return_col(df)

    # Create lagged returns per ticker
    df = df.sort_values(["ticker", "et_date"]).copy()
    df["ret_t"]   = df[ret_col]
    df["ret_t-1"] = df.groupby("ticker")["ret_t"].shift(1)
    df["ret_t-2"] = df.groupby("ticker")["ret_t"].shift(2)

    # Per-ticker correlations: market -> news
    rows = []
    for tkr, g in df.groupby("ticker", sort=True):
        r_same  = safe_corr(g["ret_t"],   g["mood_mean"])  # return_t vs mood_t
        r_prev1 = safe_corr(g["ret_t-1"], g["mood_mean"])  # return_{t-1} vs mood_t
        r_prev2 = safe_corr(g["ret_t-2"], g["mood_mean"])  # return_{t-2} vs mood_t
        rows.append({
            "ticker": tkr,
            "r_same": r_same,
            "r_prev1": r_prev1,
            "r_prev2": r_prev2,
            "n": int(g[["ret_t","mood_mean"]].dropna().shape[0])
        })
    res = pd.DataFrame(rows).sort_values("r_prev1", ascending=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:8.6f}"))

    # Pooled correlations (all tickers together)
    pooled_same  = safe_corr(df["ret_t"],   df["mood_mean"])
    pooled_prev1 = safe_corr(df["ret_t-1"], df["mood_mean"])
    pooled_prev2 = safe_corr(df["ret_t-2"], df["mood_mean"])

    # Within-ticker (demeaned) correlations — controls for ticker fixed effects
    df["mood_dm"] = df["mood_mean"] - df.groupby("ticker")["mood_mean"].transform("mean")
    df["ret_t_dm"]   = df["ret_t"]   - df.groupby("ticker")["ret_t"].transform("mean")
    df["ret_t-1_dm"] = df["ret_t-1"] - df.groupby("ticker")["ret_t-1"].transform("mean")
    df["ret_t-2_dm"] = df["ret_t-2"] - df.groupby("ticker")["ret_t-2"].transform("mean")

    pooled_same_within  = safe_corr(df["ret_t_dm"],   df["mood_dm"])
    pooled_prev1_within = safe_corr(df["ret_t-1_dm"], df["mood_dm"])
    pooled_prev2_within = safe_corr(df["ret_t-2_dm"], df["mood_dm"])

    print("\n=== POOLED market→news correlations ===")
    print(f"same-day:  r = {pooled_same: .4f}")
    print(f"lag 1 day: r = {pooled_prev1: .4f}")
    print(f"lag 2 days:r = {pooled_prev2: .4f}")

    print("\n=== WITHIN-TICKER (demeaned) correlations ===")
    print(f"same-day:  r = {pooled_same_within: .4f}")
    print(f"lag 1 day: r = {pooled_prev1_within: .4f}")
    print(f"lag 2 days:r = {pooled_prev2_within: .4f}")

    # Save the per-ticker table if you want to plot later
    out_path = DATA / "market_to_news_corr_by_ticker.csv"
    res.to_csv(out_path, index=False)
    print(f"\n[SAVE] {out_path}")

if __name__ == "__main__":
    main()
