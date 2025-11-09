# code/field_tagging.py
import re
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
IN_CSV  = DATA / "nyt_articles_slim.csv"
OUT_CSV = DATA / "nyt_articles_fields.csv"

FIELD_KEYWORDS = {
    "Semis/AI": {
        "generic": [
            r"\bsemiconductor(s)?\b", r"\bchip(s)?\b", r"\bfab(s)?\b",
            r"\bgpu(s)?\b", r"\baccelerator(s)?\b", r"\bai hardware\b",
            r"\bfoundry\b", r"\bnodes?\b", r"\bwafer(s)?\b"
        ],
        "firms": [
            r"\bnvidia\b", r"\bnvda\b", r"\bamd\b", r"\badvanced micro devices\b",
            r"\bbroadcom\b", r"\bavgo\b", r"\barm\b", r"\barm holdings\b", r"\barm plc\b"
        ],
    },
    "EV/Auto": {
        "generic": [
            r"\belectric vehicle(s)?\b", r"\bev(s)?\b", r"\bbattery\b", r"\bcharging\b",
            r"\bautonomous\b", r"\bself[- ]driving\b"
        ],
        "firms": [
            r"\btesla\b", r"\btsla\b", r"\beln?on musk\b", r"\bnio\b",
            r"\bgeneral motors\b", r"\bgm\b", r"\bford motor\b", r"\bford\b"
        ],
    },
    "Mega-cap Tech": {
        "generic": [
            r"\bcloud\b", r"\bsearch\b", r"\bads?\b", r"\bsocial\b", r"\bai\b", r"\bgenai\b",
            r"\bsmartphone\b", r"\bapp store\b", r"\bappstore\b", r"\bos\b", r"\bplatform\b"
        ],
        "firms": [
            r"\bapple\b", r"\baapl\b", r"\biphone\b", r"\bipad\b", r"\bmacbook\b",
            r"\bmicrosoft\b", r"\bmsft\b", r"\bazure\b", r"\bwindows\b",
            r"\bamazon\b", r"\bamzn\b", r"\baws\b",
            r"\bmeta\b", r"\bfacebook\b", r"\binstagram\b", r"\bwhatsapp\b",
            r"\bgoogle\b", r"\balphabet\b", r"\byoutube\b", r"\bandroid\b", r"\bgoogl\b"
        ],
    },
    "Biotech": {
        "generic": [
            r"\bbiotech\b", r"\bclinical trial(s)?\b", r"\bphase [123]\b",
            r"\bfda\b", r"\bapproval\b", r"\bdrug(s)?\b", r"\bvaccine(s)?\b", r"\btherapy\b"
        ],
        "firms": [
            r"\bmerck\b", r"\bmrk\b", r"\bkeytruda\b", r"\bpfizer\b", r"\bpfe\b",
            r"\bmoderna\b", r"\bmrna\b", r"\bbiogen\b", r"\bbiib\b"
        ],
    },
    "Banks": {
        "generic": [
            r"\bbank(s)?\b", r"\bloan(s)?\b", r"\bdeposit(s)?\b", r"\bcredit\b",
            r"\binvestment banking\b", r"\btrading revenue\b", r"\bnet interest\b"
        ],
        "firms": [
            r"\bjpmorgan\b", r"\bjp morgan\b", r"\bjpm\b",
            r"\bbank of america\b", r"\bbofa\b", r"\bbac\b",
            r"\bgoldman sachs\b", r"\bgoldman\b", r"\bgs\b"
        ],
    },
}

def compile_patterns():
    compiled = {}
    for field, dd in FIELD_KEYWORDS.items():
        pats = []
        for cat in ["generic", "firms"]:
            for pat in dd.get(cat, []):
                pats.append(re.compile(pat, flags=re.IGNORECASE))
        compiled[field] = pats
    return compiled

FIELD_PATS = compile_patterns()

def label_fields(text: str):
    """Return a dict of field→score (0 or 1) based on pattern hits in the text."""
    if not isinstance(text, str) or not text.strip():
        return {}
    scores = {}
    for field, pats in FIELD_PATS.items():
        hit = any(p.search(text) for p in pats)
        if hit:
            scores[field] = 1.0
    return scores

def main():
    df = pd.read_csv(IN_CSV)
    # combine headline + abstract
    df["text"] = (df["headline"].fillna("") + " " + df["abstract"].fillna("")).str.strip()

    # label fields
    field_scores = df["text"].apply(label_fields)

    # explode to long (article_id, field, score) so you can join with BERT/VADER sentiment later
    rows = []
    for aid, scores in zip(df["article_id"], field_scores):
        if not scores:
            continue
        for field, sc in scores.items():
            rows.append({"article_id": aid, "field": field, "field_score": sc})

    out = pd.DataFrame(rows)
    out = out.merge(df[["article_id","et_date","url","headline","abstract"]], on="article_id", how="left")
    out = out.sort_values(["et_date","field","article_id"])
    out_path = OUT_CSV
    out.to_csv(out_path, index=False)
    print(f"[WRITE] Field-tagged articles → {out_path} (rows={len(out):,}, unique articles={out['article_id'].nunique():,})")

if __name__ == "__main__":
    main()
