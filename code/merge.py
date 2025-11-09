import sys
import pandas as pd
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"
MOOD_CSV = DATA / "daily_ticker_mood.csv"
RET_CSV  = DATA / "stock_panel_daily_long.csv"
OUT_CSV  = DATA / "stock_mood_returns_panel.csv"


def _standardize_date_column(df: pd.DataFrame, preferred="et_date") -> pd.DataFrame:
    """Ensure the panel uses a unified date column named 'et_date'."""
    if preferred in df.columns:
        return df
    if "date" in df.columns:
        df = df.rename(columns={"date": preferred})
        return df
    raise KeyError("Neither 'et_date' nor 'date' found in DataFrame.")


def _parse_dates_safe(path: Path, date_cols) -> pd.DataFrame:
    """Read CSV and parse listed date columns that actually exist."""
    df = pd.read_csv(path)
    # Only parse columns that exist in the file
    to_parse = [c for c in date_cols if c in df.columns]
    if to_parse:
        df = pd.read_csv(path, parse_dates=to_parse)
    else:
        # Fallback to plain read if no parseable date columns are present at read time
        df = pd.read_csv(path)
        # Try to coerce known names post-read
        for c in ("et_date", "date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def main() -> int:
    # --- Load inputs
    if not MOOD_CSV.exists():
        print(f"[ERROR] Missing {MOOD_CSV}")
        return 1
    if not RET_CSV.exists():
        print(f"[ERROR] Missing {RET_CSV}")
        return 1

    mood = _parse_dates_safe(MOOD_CSV, date_cols=["et_date"])
    px   = _parse_dates_safe(RET_CSV,  date_cols=["et_date", "date"])

    # --- Standardize date col name on price/returns panel
    px = _standardize_date_column(px, preferred="et_date")

    # --- Basic hygiene
    if "ticker" not in mood.columns or "ticker" not in px.columns:
        missing = [c for c in ["ticker"] if c not in mood.columns or c not in px.columns]
        print(f"[ERROR] Missing required columns: {missing}")
        return 1

    # Ensure types are clean
    mood["ticker"] = mood["ticker"].astype(str).str.upper().str.strip()
    px["ticker"]   = px["ticker"].astype(str).str.upper().str.strip()

    # Optional: drop exact duplicate keys to avoid cartesian duplicates on merge
    mood = mood.drop_duplicates(subset=["et_date", "ticker"])
    px   = px.drop_duplicates(subset=["et_date", "ticker", *[c for c in px.columns if c not in ("et_date","ticker")]])

    # --- Merge (left: keep all market days from px; bring mood if present)
    panel = (px.merge(mood, on=["et_date", "ticker"], how="left")
               .sort_values(["ticker", "et_date"])
               .reset_index(drop=True))

    # --- Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")

    # --- Report
    n_rows = len(panel)
    n_tickers = panel["ticker"].nunique() if "ticker" in panel.columns else 0
    d0 = panel["et_date"].min()
    d1 = panel["et_date"].max()
    print(f"[SAVE] {OUT_CSV}  rows={n_rows:,}  tickers={n_tickers}  span={d0.date() if pd.notna(d0) else '?'}→{d1.date() if pd.notna(d1) else '?'}")
    # Quick sanity signal
    mood_cols = [c for c in panel.columns if c.startswith("mood")]
    print(f"[INFO] Mood columns merged: {mood_cols or 'None found'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
