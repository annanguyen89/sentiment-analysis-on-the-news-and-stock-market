import numpy as np
import pandas as pd
import yfinance as yf

START = "2018-01-01"
END   = "2025-10-01"   

STOCK_TICKERS = [
    # Semis/AI
    "NVDA","AMD","AVGO",
    # EV/Auto
    "TSLA","NIO","GM","F",
    # Mega-cap tech
    "AAPL","MSFT","AMZN","META","GOOGL",
    # Biotech
    "MRK","PFE","MRNA","BIIB",
    # Banks
    "JPM","BAC","GS",
]
MARKET_TICKER  = "SPY"
VIX_TICKER     = "^VIX"
ALL_TICKERS    = STOCK_TICKERS + [MARKET_TICKER, VIX_TICKER]

OUT_WIDE_CSV   = "stock_prices_daily_wide.csv"
OUT_LONG_CSV   = "stock_panel_daily_long.csv"

def extract_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            adj = df.xs("Adj Close", axis=1, level=0)
        except KeyError:
            adj = df.xs("Adj Close", axis=1, level=1)

        if VIX_TICKER in adj.columns and adj[VIX_TICKER].isna().all():
            try:
                close = df.xs("Close", axis=1, level=0)
            except KeyError:
                close = df.xs("Close", axis=1, level=1)
            if VIX_TICKER in close.columns:
                adj[VIX_TICKER] = close[VIX_TICKER]
    else:
        adj = df.copy()

    keep = [t for t in ALL_TICKERS if t in adj.columns]
    adj = adj[keep].dropna(how="all")
    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    return adj

def main():
    raw = yf.download(
        tickers=ALL_TICKERS,
        start=START, end=END,
        auto_adjust=False, progress=False, group_by="ticker"
    )

    adj = extract_adj_close(raw)
    logp   = np.log(adj)
    rets_d = logp.diff()
    market_return = rets_d.get(MARKET_TICKER)
    if market_return is None:
        market_return = pd.Series(index=rets_d.index, dtype=float)
    market_return = market_return.rename("market_return")
    vix_series = adj.get(VIX_TICKER)
    if vix_series is None:
        vix_series = pd.Series(index=adj.index, dtype=float)
    vix_change = vix_series.diff().rename("vix_change")
    wide = adj.copy()
    wide.columns = [f"close_{t}" for t in wide.columns]
    wide = pd.concat([wide, market_return, vix_change], axis=1)
    wide.index.name = "date"
    stock_rets = rets_d[[t for t in STOCK_TICKERS if t in rets_d.columns]].copy()
    stock_next = stock_rets.shift(-1)
    stock_next.columns = [f"{c}_next" for c in stock_next.columns]
    df_panel = pd.concat([stock_rets, stock_next, market_return, vix_change], axis=1)
    df_panel.index.name = "date"
    df_panel = df_panel.reset_index()
    value_vars = list(stock_rets.columns) + list(stock_next.columns)
    long = df_panel.melt(
        id_vars=["date", "market_return", "vix_change"],
        value_vars=value_vars,
        var_name="var",
        value_name="value"
    )
    long["is_next"] = long["var"].str.endswith("_next")
    long["ticker"]  = long["var"].str.replace("_next", "", regex=False)

    same = long.loc[~long["is_next"], ["date", "ticker", "value"]].rename(
        columns={"value": "stock_return_same_day"}
    )
    nxt  = long.loc[ long["is_next"], ["date", "ticker", "value"]].rename(
        columns={"value": "stock_return_next_day"}
    )

    panel = same.merge(nxt, on=["date", "ticker"], how="left")
    panel = panel.merge(df_panel[["date", "market_return", "vix_change"]], on="date", how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    wide.to_csv(OUT_WIDE_CSV, index=True, date_format="%Y-%m-%d")
    panel.to_csv(OUT_LONG_CSV, index=False)

    print(f"Daily rows: {len(adj):,} (from {adj.index.min().date()} to {adj.index.max().date()})")
    print(f"Wrote: {OUT_WIDE_CSV}")
    print(f"Wrote: {OUT_LONG_CSV}")

if __name__ == "__main__":
    main()
