import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
PANEL = DATA / "stock_mood_returns_panel.csv"
RET_COL_CANDIDATES = [
    "stock_return_same_day", "ret", "return", "daily_return", "r"
]

def pick_return_col(df: pd.DataFrame) -> str:
    for c in RET_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No return column found. Tried: {RET_COL_CANDIDATES}")

def sm_add_constant_safe(X: pd.DataFrame) -> pd.DataFrame:
    if "const" in X.columns:
        return X
    return X.assign(const=1.0)[["const"] + [c for c in X.columns]]

def main():
    from linearmodels.panel import PanelOLS

    df = pd.read_csv(PANEL, parse_dates=["et_date"])
    if "ticker" not in df.columns:
        raise KeyError("Expected 'ticker' in panel.")
    if "mood_mean" not in df.columns:
        mood_cols = [c for c in df.columns if c.startswith("mood")]
        if not mood_cols:
            raise KeyError("No mood column found (expected 'mood_mean' or a column starting with 'mood').")
        df = df.rename(columns={mood_cols[0]: "mood_mean"})

    ret_col = pick_return_col(df)

    df = df.sort_values(["ticker", "et_date"]).copy()
    df["ret_lag1"] = df.groupby("ticker")[ret_col].shift(1)
    df = df.dropna(subset=["mood_mean", "ret_lag1", "ticker", "et_date"])
    df = df.set_index(["ticker", "et_date"])
    y = df["mood_mean"]
    X = sm_add_constant_safe(df[["ret_lag1"]])
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    beta = float(res.params.get("ret_lag1", np.nan))
    se   = float(res.std_errors.get("ret_lag1", np.nan))
    pval = float(res.pvalues.get("ret_lag1", np.nan))

    r2_within = getattr(res, "rsquared_within", None)
    if r2_within is None:
        r2_within = getattr(res, "rsquared", np.nan)
    r2_within = float(r2_within)

    print("=== Panel FE: market → next-day news mood ===")
    print(f"beta (ret_lag1) = {beta: .6f}")
    print(f"SE (NW HAC)     = {se: .6f}")
    print(f"p-value         = {pval: .6g}")
    print(f"R^2 (within)    = {r2_within: .4f}\n")

    try:
        print(res.summary.tables[0])
        print(res.summary.tables[1])
    except Exception:
        print(res.summary)

if __name__ == "__main__":
    main()
