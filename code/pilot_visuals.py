import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

DATA_CSV = "../data/pilot_XLK_2016_model_data.csv"   # adjust if needed
SECTOR   = "XLK"
OUT_SCATTER = "xlk_scatter_lowess.png"
OUT_TS      = "xlk_timeseries.png"
WINSOR_PCT  = 0.01   # winsorize at 1%/99% for cleaner visuals (set to 0 to disable)

def winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def main():
    # ---- load & filter ----
    df = pd.read_csv(DATA_CSV, parse_dates=["et_date"])
    if "sector" in df.columns:
        df = df[df["sector"] == SECTOR].copy()

    # required cols
    req = ["et_date", "mood", "sector_return_next_day"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # clean NaNs / inf
    for c in ["mood","sector_return_next_day"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mood","sector_return_next_day"]).copy()

    # optional winsorization
    if WINSOR_PCT > 0:
        df["mood_w"]  = winsorize(df["mood"], WINSOR_PCT)
        df["ret_w"]   = winsorize(df["sector_return_next_day"], WINSOR_PCT)
    else:
        df["mood_w"] = df["mood"]
        df["ret_w"]  = df["sector_return_next_day"]

    # -------------- Plot 1: scatter + LOWESS --------------
    x = df["mood_w"].to_numpy()
    y = df["ret_w"].to_numpy()

    # LOWESS (sorted by x for a clean line)
    order = np.argsort(x)
    lo = lowess(y[order], x[order], frac=0.35, it=0, return_sorted=True)  # frac≈ smoothing span
    lx, ly = lo[:,0], lo[:,1]

    # correlations
    # (pandas corr drops NaN automatically)
    pearson  = pd.Series(x).corr(pd.Series(y), method="pearson")
    spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
    kendall  = pd.Series(x).corr(pd.Series(y), method="kendall")
    n = len(df)

    plt.figure(figsize=(8,6))
    # point sizes scaled by article coverage if present
    if "n_articles" in df.columns:
        # normalize sizes (keep in a nice range)
        s = 20 + 2.5 * (df["n_articles"] - df["n_articles"].min()) / max(1, (df["n_articles"].max() - df["n_articles"].min())) * 80
        plt.scatter(x, y, s=s, alpha=0.6, linewidths=0.5)
    else:
        plt.scatter(x, y, alpha=0.6, linewidths=0.5)

    plt.plot(lx, ly)  # LOWESS line
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.title(f"{SECTOR}: Mood (t) vs Next-Day Return (t+1)")
    plt.xlabel("Sector mood on day t")
    plt.ylabel("Next-day log return")

    # annotate stats
    txt = (
        f"n = {n}\n"
        f"Pearson r = {pearson:.3f}\n"
        f"Spearman ρ = {spearman:.3f}\n"
        f"Kendall τ = {kendall:.3f}"
    )
    # put in upper left; a small white bbox helps readability
    plt.text(
        0.02, 0.98, txt, transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=200)
    plt.close()
    print(f"Wrote {OUT_SCATTER}")

    # -------------- Plot 2: time-series (twin y axes) --------------
    ts = df.sort_values("et_date").copy()
    # Align mood_t with return_{t+1} display: just plot both on their own dates
    # (your regression uses next-day; this is purely a visual overlay)
    fig, ax1 = plt.subplots(figsize=(10,4.8))

    ax1.plot(ts["et_date"], ts["mood_w"], label="Mood (t)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mood (t)")

    ax2 = ax1.twinx()
    ax2.plot(ts["et_date"], ts["ret_w"], label="Next-day return (t+1)")
    ax2.set_ylabel("Next-day log return")

    # build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"{SECTOR}: Daily Mood vs Next-Day Return (2016)")

    fig.tight_layout()
    fig.savefig(OUT_TS, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_TS}")

if __name__ == "__main__":
    main()