import pandas as pd
import numpy as np
from pathlib import Path

# 11 Select Sector SPDR tickers
SECTORS = ["XLK","XLE","XLV","XLF","XLI","XLY","XLP","XLB","XLU","XLRE","XLC"]
DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
ALLOWED = {0, 0.25, 0.5, 0.75, 1.0}
MAX_NONZERO = 2  # at most two sectors per topic

# 1) Build topic cards (top words + example headlines)
def build_topic_cards(
    topics_top_words_csv=DATA_DIR/"lda_topic_topwords.csv",
    article_topic_weights_csv=DATA_DIR/"lda_doc_topic_weights.csv",
    slim_csv=DATA_DIR/"nyt_articles_slim.csv",
    out_cards_csv=DATA_DIR/"topic_cards.csv",
    top_words=15,
    examples_per_topic=5
):
    import pandas as pd

    # Top words per topic → single row string
    words = pd.read_csv(topics_top_words_csv)
    topw = (words.sort_values(["topic","rank"])
                 .groupby("topic").head(top_words)
                 .groupby("topic")["word"]
                 .apply(lambda s: ", ".join(s.tolist()))
                 .reset_index(name="top_words"))

    # Melt doc–topic to long and grab example headlines
    atw_wide = pd.read_csv(article_topic_weights_csv)
    topic_cols = [c for c in atw_wide.columns if c.startswith("topic_")]
    atw_long = atw_wide.melt(
        id_vars=["article_id","et_date"],
        value_vars=topic_cols,
        var_name="topic_col",
        value_name="theta"
    )
    atw_long["topic"] = atw_long["topic_col"].str.replace("topic_","", regex=False).astype(int)

    # Join headlines for examples
    slim = pd.read_csv(slim_csv, usecols=["article_id","headline"])
    ex_join = atw_long.merge(slim, on="article_id", how="left")

    ex = (ex_join.sort_values(["topic","theta"], ascending=[True, False])
                .groupby("topic")
                .head(examples_per_topic)
                .groupby("topic")["headline"]
                .apply(lambda s: " | ".join(s.dropna().astype(str).tolist()))
                .reset_index(name="example_headlines"))

    # Assemble the “cards” table
    cards = (topw.merge(ex, on="topic", how="outer")
                  .sort_values("topic")
                  .reset_index(drop=True))
    cards["topic_label"] = ""   # you will fill
    cards["notes"] = ""         # optional
    cards.to_csv(out_cards_csv, index=False)
    print(f"[WRITE] Topic cards → {out_cards_csv}")
    return cards

# 2) Export single-annotator sheet with sector weight columns ----------
def export_annotator_sheet(
    topic_cards_csv=DATA_DIR/"topic_cards.csv",
    out_sheet_csv=DATA_DIR/"topic_to_sector_single_annotator.csv"
):
    cards = pd.read_csv(topic_cards_csv)
    sheet = cards.copy()
    for s in SECTORS:
        sheet[f"w_{s}"] = 0.0
    sheet.to_csv(out_sheet_csv, index=False)
    print(f"[WRITE] Single-annotator sheet → {out_sheet_csv}")
    return out_sheet_csv

# ---------- 3) Validate your weights ----------
def _check_row_weights(row):
    weights = row[[f"w_{s}" for s in SECTORS]].values.astype(float)
    ok_values = all(w in ALLOWED for w in weights)
    nonzero = np.count_nonzero(weights)
    ok_nonzero = nonzero <= MAX_NONZERO
    ok_sum = np.isclose(weights.sum(), 1.0)  # must sum to 1
    return ok_values, ok_nonzero, ok_sum

def validate_single_annotator(csv_path):
    df = pd.read_csv(csv_path)
    bad = []
    for _, r in df.iterrows():
        ok_values, ok_nonzero, ok_sum = _check_row_weights(r)
        if not (ok_values and ok_nonzero and ok_sum):
            bad.append({
                "topic": int(r["topic"]),
                "ok_values": ok_values,
                "ok_nonzero(<=2)": ok_nonzero,
                "ok_sum(=1)": ok_sum
            })
    if bad:
        out = pd.DataFrame(bad)
        print("[VALIDATE] Found issues on these topics:")
        print(out.to_string(index=False))
    else:
        print(f"[VALIDATE] {csv_path} looks good ✓")
    return bad

# ---------- 4) Finalize mapping (long form: topic, sector, weight) ----------
def finalize_mapping(csv_path, out_final_long=DATA_DIR/"topic_to_sector_final_long.csv"):
    df = pd.read_csv(csv_path)

    # Enforce: keep top-2 weights (if you accidentally gave >2), renormalize to sum 1
    def enforce_policy(row):
        w = row[[f"w_{s}" for s in SECTORS]].values.astype(float)
        # zero small non-allowed rounding
        w = np.array([float(v) if v in ALLOWED else 0.0 for v in w])
        # keep top-2
        if (w > 0).sum() > MAX_NONZERO:
            idx = np.argsort(-w)[:MAX_NONZERO]
            w_new = np.zeros_like(w); w_new[idx] = w[idx]
            w = w_new
        s = w.sum()
        if s > 0:
            w = w / s
        return pd.Series(w, index=[f"w_{s}" for s in SECTORS])

    W = df.apply(enforce_policy, axis=1)
    wide = pd.concat([df[["topic","topic_label","notes"]], W], axis=1)

    long = (wide.melt(id_vars=["topic","topic_label","notes"],
                      var_name="sector_col", value_name="weight")
                 .assign(sector=lambda x: x["sector_col"].str.replace("w_","",regex=False))
                 .drop(columns=["sector_col"])
                 .query("weight > 0")
                 .sort_values(["topic","weight"], ascending=[True, False])
                 .reset_index(drop=True))

    long.to_csv(out_final_long, index=False)
    print(f"[WRITE] Final mapping (long) → {out_final_long}")
    return long

# ---------- Convenience: one function to run steps 1–4 ----------
def run_single_annotator_pipeline():
    build_topic_cards(  # writes to DATA_DIR/topic_cards.csv
        topics_top_words_csv=DATA_DIR/"lda_topic_topwords.csv",
        article_topic_weights_csv=DATA_DIR/"lda_doc_topic_weights.csv",
        slim_csv=DATA_DIR/"nyt_articles_slim.csv",
        out_cards_csv=DATA_DIR/"topic_cards.csv"
    )
    export_annotator_sheet(
        topic_cards_csv=DATA_DIR/"topic_cards.csv",
        out_sheet_csv=DATA_DIR/"topic_to_sector_single_annotator.csv"
    )
    print("\nNow open 'data/topic_to_sector_single_annotator.csv' and fill:")
    print("  - topic_label (short human name)")
    print("  - w_XLK, w_XLE, ..., w_XLC using {0,.25,.5,.75,1}, ≤2 nonzero, sum=1\n")

# --- calls (all in DATA_DIR) ---
run_single_annotator_pipeline()
validate_single_annotator(DATA_DIR/"topic_to_sector_single_annotator.csv")
finalize_mapping(DATA_DIR/"topic_to_sector_single_annotator.csv")
