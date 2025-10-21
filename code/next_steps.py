import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

PILOT_CSV = DATA / "pilot_XLK_2016_model_data.csv"   # has: et_date, sector, mood, sector_return_next_day, market_return, vix_change
ETF_PANEL_OPTIONAL = DATA / "sector_panel_daily_xlk_2016.csv"  # if present, use to fetch sector_return_same_day
OUT_RESULTS = DATA / "xlk_next_steps_results.csv"

SECTOR = "XLK"
HAC_LAGS_LIST = [3, 5, 7]
WINSOR_P = 0.01  # 1% tails

def winsorize(series: pd.Series, p=WINSOR_P):
    lo, hi = series.quantile([p, 1-p])
    return series.clip(lower=lo, upper=hi)


def add_own_lag(df: pd.DataFrame) -> pd.DataFrame:
    if "sector_return_same_day" in df.columns:
        return df.copy()

    if ETF_PANEL_OPTIONAL.exists():
        etf = pd.read_csv(ETF_PANEL_OPTIONAL, parse_dates=["et_date", "date"], infer_datetime_format=True)
        # standardize date column name to et_date
        if "date" in etf.columns and "et_date" not in etf.columns:
            etf = etf.rename(columns={"date": "et_date"})
        need_cols = {"et_date", "sector", "sector_return_same_day"}
        if not need_cols.issubset(etf.columns):
            # Try to reconstruct same-day from next-day by shifting (rare fallback)
            if "sector_return_next_day" in etf.columns:
                etf = etf.sort_values(["sector", "et_date"])
                etf["sector_return_same_day"] = etf.groupby("sector")["sector_return_next_day"].shift(1)
            else:
                raise ValueError("Could not find or reconstruct 'sector_return_same_day'.")
        keep = etf[["et_date","sector","sector_return_same_day"]].drop_duplicates()
        out = df.merge(keep, on=["et_date","sector"], how="left")
        return out
    else:
        # fallback: try reconstruct from this file if it has next_day and is sorted
        tmp = df.sort_values(["sector","et_date"]).copy()
        if "sector_return_next_day" in tmp.columns:
            tmp["sector_return_same_day"] = tmp.groupby("sector")["sector_return_next_day"].shift(1)
            return tmp
        raise ValueError("No same-day return available (and optional panel not found).")


def partial_r2(y: pd.Series, X_controls: pd.DataFrame, X_full: pd.DataFrame) -> float:
    yc = y.values
    Xr = sm.add_constant(X_controls).values
    Xf = sm.add_constant(X_full).values

    m_red = sm.OLS(yc, Xr).fit()
    m_full = sm.OLS(yc, Xf).fit()

    ssr_red = np.sum(m_red.resid**2)
    ssr_full = np.sum(m_full.resid**2)
    return float((ssr_red - ssr_full) / ssr_red) if ssr_red > 0 else np.nan


def fit_linear(y, X, hac_lags=5):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def fit_rcs(y, X_controls, mood, df_spline=4, hac_lags=5):
    """
    Restricted cubic spline for mood.
    Returns (model, spline_cols, control_cols, df_spline, spline_design_info)
    """
    y2 = y.reset_index(drop=True)
    Xc = X_controls.reset_index(drop=True)
    m2 = mood.reset_index(drop=True)

    # Fit-time spline basis on the FULL mood series (stores knots internally)
    basis = dmatrix(f"cr(mood, df={df_spline}) - 1",
                    {"mood": m2},
                    return_type="dataframe")
    spline_di = basis.design_info  # capture design info (knots + column spec)

    # Combine and fit
    X = pd.concat([Xc, basis], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y2, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    spline_cols  = list(basis.columns)
    control_cols = list(Xc.columns)
    return model, spline_cols, control_cols, df_spline, spline_di




def wald_pvalue_for_terms(model, term_names):
    R = np.eye(len(model.params))[ [model.params.index.get_loc(t) for t in term_names] ]
    wtest = model.wald_test(R)
    return float(wtest.pvalue)


def quintile_summary(df: pd.DataFrame):
    tmp = df.dropna(subset=["mood","sector_return_next_day"]).copy()
    tmp["mood_q"] = pd.qcut(tmp["mood"], 5, labels=[1,2,3,4,5])
    summ = tmp.groupby("mood_q")["sector_return_next_day"].mean().reset_index()
    hi = summ.loc[summ["mood_q"]==5, "sector_return_next_day"].values[0]
    lo = summ.loc[summ["mood_q"]==1, "sector_return_next_day"].values[0]
    hl = hi - lo
    return summ, float(hl)


def main():
    if not PILOT_CSV.exists():
        raise SystemExit(f"Missing file: {PILOT_CSV}")

    df = pd.read_csv(PILOT_CSV, parse_dates=["et_date"])
    need = {"sector","et_date","mood","sector_return_next_day","market_return","vix_change"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{PILOT_CSV} missing columns: {sorted(list(need - set(df.columns)))}")

    # Keep XLK only & clean NAs/infs
    df = df[df["sector"] == SECTOR].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    base_n = len(df)
    df = df.dropna(subset=["mood", "sector_return_next_day", "market_return", "vix_change"]).copy()
    print(f"[CLEAN] XLK rows before: {base_n}, after NA/inf drop: {len(df)}")

    # Ensure same-day return exists (own-lag control)
    df = add_own_lag(df)
    # After adding, drop NA in same-day if present at the first row
    if "sector_return_same_day" in df.columns:
        df = df.dropna(subset=["sector_return_same_day"]).copy()

    # ---------------- Baseline: linear ----------------
    y = df["sector_return_next_day"]
    X_controls = df[["market_return","vix_change"]]
    X_full_lin = pd.concat([X_controls, df[["mood"]]], axis=1)

    results = []

    for hac in HAC_LAGS_LIST:
        m_lin = fit_linear(y, X_full_lin, hac_lags=hac)
        beta = m_lin.params.get("mood", np.nan)
        se = m_lin.bse.get("mood", np.nan)
        p  = m_lin.pvalues.get("mood", np.nan)
        eff_0p10 = beta * 0.10
        eff_1sd  = beta * df["mood"].std()
        pr2 = partial_r2(y, X_controls, X_full_lin)

        results.append({
            "model": f"Linear (HAC={hac})",
            "beta_mood": beta,
            "se_mood": se,
            "p_mood": p,
            "bps_per_0.10": eff_0p10 * 10000,
            "bps_per_1sd": eff_1sd * 10000,
            "partial_R2_mood": pr2
        })

    # --------------- Add own-lag r_t -----------------
    X_controls_lag = pd.concat([X_controls, df[["sector_return_same_day"]]], axis=1)
    X_full_lag = pd.concat([X_controls_lag, df[["mood"]]], axis=1)
    for hac in HAC_LAGS_LIST:
        m_lag = fit_linear(y, X_full_lag, hac_lags=hac)
        beta = m_lag.params.get("mood", np.nan)
        se = m_lag.bse.get("mood", np.nan)
        p  = m_lag.pvalues.get("mood", np.nan)
        eff_0p10 = beta * 0.10
        eff_1sd  = beta * df["mood"].std()
        pr2 = partial_r2(y, X_controls_lag, X_full_lag)

        results.append({
            "model": f"Linear + r_t (HAC={hac})",
            "beta_mood": beta,
            "se_mood": se,
            "p_mood": p,
            "bps_per_0.10": eff_0p10 * 10000,
            "bps_per_1sd": eff_1sd * 10000,
            "partial_R2_mood": pr2
        })

    # --------------- Nonlinearity (RCS) --------------
    # keep controls = SPY, dVIX (and try with r_t as well)
    for tag, Xc in [("RCS (controls only)", X_controls),
                    ("RCS (controls + r_t)", X_controls_lag)]:
        for hac in HAC_LAGS_LIST:
            m_rcs, spline_terms, control_cols, df_spline, spline_di = fit_rcs(y, Xc, df["mood"], df_spline=4, hac_lags=hac)
            p_joint = wald_pvalue_for_terms(m_rcs, spline_terms)

            # Finite-difference local effect near the mean mood (+0.10)
            mbar = float(df["mood"].mean())
            eps  = 0.10

            def row_for(mood_value: float):
                # Controls at sample means
                ctrl_means = {c: float(df[c].mean()) for c in control_cols}

                # Build spline basis with the SAME knots/columns as at fit-time
                B_list = build_design_matrices([spline_di], {"mood": [mood_value]})
                B = pd.DataFrame(B_list[0], columns=spline_terms)

                # Assemble a row that matches model.params index exactly
                row = pd.Series(0.0, index=m_rcs.params.index, dtype=float)
                if "const" in row.index:
                    row["const"] = 1.0
                for c, v in ctrl_means.items():
                    if c in row.index:
                        row[c] = v
                for col in spline_terms:
                    if col in row.index:
                        row[col] = float(B[col].iloc[0])
                return row

            row0 = row_for(mbar)
            row1 = row_for(mbar + eps)
            y0   = float(np.dot(row0.values, m_rcs.params.values))
            y1   = float(np.dot(row1.values, m_rcs.params.values))
            eff_0p10 = (y1 - y0)  # log-return change for +0.10 mood near mean

            results.append({
                "model": f"{tag} (HAC={hac})",
                "beta_mood": np.nan, "se_mood": np.nan, "p_mood": np.nan,
                "bps_per_0.10": eff_0p10 * 10000,
                "bps_per_1sd": np.nan,
                "partial_R2_mood": np.nan,
                "rcs_joint_p": p_joint
            })

    df_w = df.copy()
    for col in ["mood","sector_return_next_day","market_return","vix_change","sector_return_same_day"]:
        if col in df_w.columns:
            df_w[col] = winsorize(df_w[col], p=WINSOR_P)

    y_w = df_w["sector_return_next_day"]
    X_controls_w = df_w[["market_return","vix_change"]]
    X_full_w = pd.concat([X_controls_w, df_w[["mood"]]], axis=1)

    for hac in HAC_LAGS_LIST:
        m_w = fit_linear(y_w, X_full_w, hac_lags=hac)
        beta = m_w.params.get("mood", np.nan)
        se = m_w.bse.get("mood", np.nan)
        p  = m_w.pvalues.get("mood", np.nan)
        eff_0p10 = beta * 0.10
        eff_1sd  = beta * df_w["mood"].std()
        pr2 = partial_r2(y_w, X_controls_w, X_full_w)

        results.append({
            "model": f"Winsorized (HAC={hac})",
            "beta_mood": beta,
            "se_mood": se,
            "p_mood": p,
            "bps_per_0.10": eff_0p10 * 10000,
            "bps_per_1sd": eff_1sd * 10000,
            "partial_R2_mood": pr2
        })

    qtab, hl = quintile_summary(df)
    print("\n=== Mood Quintile Avg Next-Day Return (log) ===")
    print(qtab.to_string(index=False))
    print(f"High − Low (Q5 − Q1): {hl:.6f}  ({hl*10000:.2f} bps)")

    out = pd.DataFrame(results)
    out.to_csv(OUT_RESULTS, index=False)
    print(f"\n[WRITE] Summary table → {OUT_RESULTS}")
    print("\nDone")


if __name__ == "__main__":
    main()
