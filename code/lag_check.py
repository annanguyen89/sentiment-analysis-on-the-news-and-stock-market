# lag_checks_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# -------------------- Config --------------------
DATA_DIR = Path("../data")  # adjust if needed
MOOD_CSV = DATA_DIR / "daily_sector_mood_2016.csv"
RET_CSV  = DATA_DIR / "sector_panel_daily_long.csv"

SECTOR   = "XLK"           # sector to analyze
LAGS     = [-2, -1, 0, 1, 2]  # h in return_{t+h}
HAC_MAXLAGS = 5
SAVE_PREFIX = DATA_DIR / f"lagcheck_{SECTOR}_2016"

def load_and_align(sector: str) -> pd.DataFrame:
    # Mood (t)
    mood = pd.read_csv(MOOD_CSV, parse_dates=["et_date"])
    mood = mood[mood["sector"] == sector].rename(columns={"et_date": "date"})[["date", "mood"]]

    # Returns & controls
    ret = pd.read_csv(RET_CSV, parse_dates=["date"])
    need_cols = ["date", "sector", "sector_return_same_day", "market_return", "vix_change"]
    missing = [c for c in need_cols if c not in ret.columns]
    if missing:
        raise ValueError(f"sector_panel_daily_long.csv is missing: {missing}")

    ret = ret[ret["sector"] == sector][["date", "sector_return_same_day", "market_return", "vix_change"]]

    # Merge on date (inner to keep aligned days)
    df = mood.merge(ret, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return df

def run_hac_beta(df: pd.DataFrame, h: int):
    """
    Fit: return_{t+h} ~ mood_t + market_return_t + vix_change_t (HAC)
    Using sector_return_same_day shifted by -h to represent return_{t+h}.
    """
    # create target aligned to t
    y = df["sector_return_same_day"].shift(-h)  # shift negative to align future return with today mood
    design = pd.DataFrame({
        "const": 1.0,
        "mood": df["mood"],
        "market_return": df["market_return"],
        "vix_change": df["vix_change"]
    })
    d = pd.concat([y.rename("ret_h"), design], axis=1).dropna()

    if len(d) < 20:
        return np.nan, np.nan, np.nan, len(d)

    model = sm.OLS(d["ret_h"], d[["const","mood","market_return","vix_change"]]).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS}
    )
    b = model.params.get("mood", np.nan)
    se = model.bse.get("mood", np.nan)
    p  = model.pvalues.get("mood", np.nan)
    return b, se, p, len(d)

def main():
    df = load_and_align(SECTOR)

    rows = []
    for h in LAGS:
        b, se, p, n = run_hac_beta(df, h)
        rows.append({"lag_h": h, "beta_mood": b, "se": se, "pval": p, "n": n})
    res = pd.DataFrame(rows).sort_values("lag_h").reset_index(drop=True)

    # Compute 95% CI
    res["ci_low"]  = res["beta_mood"] - 1.96 * res["se"]
    res["ci_high"] = res["beta_mood"] + 1.96 * res["se"]

    # ---- Plot 1: HAC β by lag with 95% CI
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.axhline(0, lw=1, alpha=0.5)
    ax1.errorbar(
        res["lag_h"], res["beta_mood"], 
        yerr=1.96*res["se"], fmt="o-", capsize=3
    )
    ax1.set_xlabel("Lag h in return(t+h)")
    ax1.set_ylabel("HAC OLS β on mood")
    ax1.set_title(f"{SECTOR}: Effect of Mood on Return Across Lags\nControls: SPY, ΔVIX; 95% CI; n varies by lag")
    for i, n in enumerate(res["n"]):
        ax1.text(res.loc[i, "lag_h"], res.loc[i, "ci_low"], f"n={int(n)}", ha="center", va="top", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}_hac_beta_by_lag.png", dpi=180)

    # ---- Plot 2: Pearson correlation of mood_t with return_{t+h} (no controls)
    corr_rows = []
    for h in LAGS:
        y = df["sector_return_same_day"].shift(-h)
        tmp = pd.concat([df["mood"], y.rename("ret_h")], axis=1).dropna()
        if len(tmp) < 10:
            corr = np.nan
        else:
            corr = tmp["mood"].corr(tmp["ret_h"])
        corr_rows.append({"lag_h": h, "corr": corr, "n": len(tmp)})
    cres = pd.DataFrame(corr_rows).sort_values("lag_h")

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.0))
    ax2.axhline(0, lw=1, alpha=0.5)
    ax2.bar(cres["lag_h"].astype(str), cres["corr"])
    ax2.set_xlabel("Lag h in return(t+h)")
    ax2.set_ylabel("Pearson corr(mood_t, return_{t+h})")
    ax2.set_title(f"{SECTOR}: Correlation of Mood with Future/Past Returns")
    for i, n in enumerate(cres["n"]):
        ax2.text(i, cres.loc[i, "corr"] + (0.02 if cres.loc[i, "corr"]>=0 else -0.02), f"n={int(n)}", ha="center", va="bottom" if cres.loc[i, "corr"]>=0 else "top", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}_corr_by_lag.png", dpi=180)

    # Console summary
    print("\n=== HAC OLS β by lag (controls: SPY, ΔVIX) ===")
    print(res[["lag_h","beta_mood","se","pval","n"]].to_string(index=False))
    print("\n=== Pearson corr by lag (no controls) ===")
    print(cres.to_string(index=False))
    print(f"\nSaved plots:\n - {SAVE_PREFIX}_hac_beta_by_lag.png\n - {SAVE_PREFIX}_corr_by_lag.png")

if __name__ == "__main__":
    main()
