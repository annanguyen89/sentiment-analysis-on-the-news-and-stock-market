#!/usr/bin/env python3
import math
import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DATA = Path(__file__).resolve().parents[1] / "data"
SLIM = DATA / "nyt_articles_slim.csv"
OUT  = DATA / "nyt_articles_with_sentiment.csv"

MODEL_ID = "ProsusAI/finbert"

# Load FinBERT once (CPU, safetensors)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, use_safetensors=True)
_clf = pipeline(
    task="text-classification",
    model=_model,
    tokenizer=_tokenizer,
    truncation=True,
    top_k=None,          # replaces return_all_scores=True
    batch_size=32,       # can override via --batch-size
    device=-1            # CPU
)
LABELS = {"negative", "neutral", "positive"}

def signed_score(probs):
    """Convert list[{label, score}] -> scalar in [-1, +1] via P(pos) - P(neg)."""
    d = {x["label"].lower(): float(x["score"]) for x in probs}
    if not LABELS.issubset(d):
        # loose remap in case labels have odd casing
        for k, v in list(d.items()):
            kl = k.lower()
            if "pos" in kl:  d["positive"] = v
            if "neg" in kl:  d["negative"] = v
            if "neut" in kl: d["neutral"]  = v
    return d.get("positive", 0.0) - d.get("negative", 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only score the first N rows (debug).")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for the HF pipeline.")
    args = ap.parse_args()

    if not SLIM.exists():
        raise SystemExit(f"[ERROR] Missing input file: {SLIM}")

    df = pd.read_csv(SLIM)
    # Build input text (headline + abstract)
    texts = (df.get("headline", "").fillna("") + " " + df.get("abstract", "").fillna("")).str.strip()

    mask = texts.str.len() > 0
    idx_to_score = texts[mask].index
    if args.limit:
        idx_to_score = idx_to_score[:args.limit]

    n_total = len(df)
    n_score = len(idx_to_score)
    print(f"[INFO] Articles total: {n_total:,} | to score (non-empty): {n_score:,}")

    if n_score == 0:
        print("[WARN] No non-empty texts to score; writing passthrough file.")
        df["sentiment"] = None
    else:
        # adjust pipeline batch size
        _clf.model.config.use_cache = True
        _clf._batch_size = args.batch_size

        sentiments = pd.Series([None] * n_total, index=df.index, dtype="float64")
        B = args.batch_size
        num_batches = math.ceil(n_score / B)

        for bi in range(num_batches):
            start = bi * B
            end = min((bi + 1) * B, n_score)
            batch_idx = idx_to_score[start:end]
            batch_texts = texts.loc[batch_idx].tolist()

            outputs = _clf(batch_texts)  # list of lists
            batch_scores = [signed_score(probs) if isinstance(probs, list) else None for probs in outputs]
            sentiments.loc[batch_idx] = batch_scores

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                print(f"[INFO] Scored batch {bi+1}/{num_batches} "
                      f"({end}/{n_score} rows, {100*end/n_score:0.1f}%)")

        df["sentiment"] = sentiments

    # Keep your ordering/columns behavior
    if "et_date" in df.columns:
        df = df.dropna(subset=["et_date"]).sort_values(["et_date", "article_id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[SAVE] {OUT}  rows={len(df):,}")

    # quick stats
    s = df["sentiment"].dropna()
    if not s.empty:
        print(f"[INFO] Sentiment stats: mean={s.mean():.4f}, std={s.std():.4f}, "
              f"min={s.min():.4f}, p10={s.quantile(0.1):.4f}, p90={s.quantile(0.9):.4f}, max={s.max():.4f}")

if __name__ == "__main__":
    main()
