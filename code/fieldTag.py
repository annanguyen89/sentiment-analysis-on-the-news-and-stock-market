import re
from collections import defaultdict
from pathlib import Path
import pandas as pd

def safe_text(x):
    return x if isinstance(x, str) else ""

def any_match(patterns, text):
    txt = safe_text(text)
    for p in patterns:
        if hasattr(p, "search"):          
            if p.search(txt):
                return True
        else:                         
            if re.search(p, txt, flags=re.IGNORECASE):
                return True
    return False

def proximity_boost(text, term_a, term_b, window=8):
    txt = safe_text(text).lower()
    if not txt:
        return 0
    toks = re.split(r"\W+", txt)
    ta = safe_text(term_a).lower()
    tb = safe_text(term_b).lower()
    if not ta or not tb:
        return 0
    pos_a = [i for i, t in enumerate(toks) if t == ta]
    pos_b = [i for i, t in enumerate(toks) if t == tb]
    if not pos_a or not pos_b:
        return 0
    return 1 if any(abs(i - j) <= window for i in pos_a for j in pos_b) else 0

def pattern_head_token(pat_str):
    s = re.sub(r"\\b", "", pat_str)
    s = re.sub(r"\([^)]*\)\??", "", s)   
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip()
    return s.split()[0].lower() if s else ""


USE_SPACY = True
nlp = None
if USE_SPACY:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    except Exception as e:
        print(f"[WARN] spaCy model not available ({e}); falling back to alias-only matching.")
        nlp = None

def extract_orgs(text):
    """Extract ORG/PRODUCT entities via spaCy, or [] if spaCy unavailable."""
    if not nlp:
        return []
    txt = safe_text(text)
    if not txt:
        return []
    doc = nlp(txt)
    return [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT")]


DATA = Path(__file__).resolve().parents[1] / "data"
IN_CSV  = DATA / "nyt_articles_slim.csv"
OUT_CSV = DATA / "nyt_articles_fields_smart.csv"
ALIAS_TO_SECTOR = {
    # Semis/AI
    "nvidia": "Semis/AI", "nvda": "Semis/AI",
    "advanced micro devices": "Semis/AI", "amd": "Semis/AI",
    "broadcom": "Semis/AI", "avgo": "Semis/AI", "arm": "Semis/AI",
    # EV/Auto
    "tesla": "EV/Auto", "tsla": "EV/Auto", "nio": "EV/Auto",
    "general motors": "EV/Auto", "gm": "EV/Auto", "ford": "EV/Auto",
    # Mega-cap Tech
    "apple": "Mega-cap Tech", "aapl": "Mega-cap Tech", "iphone": "Mega-cap Tech",
    "microsoft": "Mega-cap Tech", "msft": "Mega-cap Tech", "azure": "Mega-cap Tech",
    "amazon": "Mega-cap Tech", "amzn": "Mega-cap Tech", "aws": "Mega-cap Tech",
    "meta": "Mega-cap Tech", "facebook": "Mega-cap Tech", "instagram": "Mega-cap Tech", "whatsapp": "Mega-cap Tech",
    "google": "Mega-cap Tech", "alphabet": "Mega-cap Tech", "youtube": "Mega-cap Tech", "android": "Mega-cap Tech", "googl": "Mega-cap Tech",
    # Biotech
    "merck": "Biotech", "mrk": "Biotech", "keytruda": "Biotech",
    "pfizer": "Biotech", "pfe": "Biotech", "moderna": "Biotech", "mrna": "Biotech",
    "biogen": "Biotech", "biib": "Biotech",
    # Banks
    "jpmorgan": "Banks", "jp morgan": "Banks", "jpm": "Banks",
    "bank of america": "Banks", "bofa": "Banks", "bac": "Banks",
    "goldman sachs": "Banks", "goldman": "Banks", "gs": "Banks",
}

CONTEXT_HINTS = {
    "Semis/AI": [r"\bchip(s)?\b", r"\bsemiconductor(s)?\b", r"\bgpu(s)?\b", r"\bfab(s)?\b"],
    "EV/Auto": [r"\belectric vehicle(s)?\b", r"\bev(s)?\b", r"\bbattery\b", r"\bcharging\b"],
    "Mega-cap Tech": [r"\bcloud\b", r"\bgenai\b", r"\bapp store\b", r"\bads?\b"],
    "Biotech": [r"\bclinical trial(s)?\b", r"\bfda\b", r"\bphase [123]\b", r"\bdrug(s)?\b"],
    "Banks": [r"\bloan(s)?\b", r"\bdeposit(s)?\b", r"\bnet interest\b", r"\binvestment banking\b"],
}
CONTEXT_HINTS = {
    k: [re.compile(p, re.IGNORECASE) for p in v]
    for k, v in CONTEXT_HINTS.items()
}

NEGATIVE_PATTERNS = [
    (re.compile(r"\barm\b", re.IGNORECASE),
     re.compile(r"arm of (the )?chair|disarm|arm(y|ed)", re.IGNORECASE)),
]

SECTOR_HEAD_TOKENS = {
    sec: pattern_head_token(p.pattern) for sec, pats in CONTEXT_HINTS.items() for p in pats[:1]
}
SECTOR_HEAD_TOKENS.setdefault("Semis/AI", "chip")
SECTOR_HEAD_TOKENS.setdefault("EV/Auto", "vehicle")
SECTOR_HEAD_TOKENS.setdefault("Mega-cap Tech", "cloud")
SECTOR_HEAD_TOKENS.setdefault("Biotech", "drug")
SECTOR_HEAD_TOKENS.setdefault("Banks", "loan")

def score_article(headline, abstract):
    headline = safe_text(headline)
    abstract = safe_text(abstract)
    text = f"{headline} {abstract}".strip()
    if not text:
        return {}

    for must, avoid in NEGATIVE_PATTERNS:
        if must.search(text) and avoid.search(text):
            text = must.sub("", text)

    votes = defaultdict(float)
    orgs = extract_orgs(text) 
    lc_text = text.lower()

    for alias, sector in ALIAS_TO_SECTOR.items():
        if re.search(rf"\b{re.escape(alias)}\b", lc_text):
            votes[sector] += 1.0
    for ent in orgs:
        a = ent.lower().strip()
        if a in ALIAS_TO_SECTOR:
            votes[ALIAS_TO_SECTOR[a]] += 1.0
    for sector, pats in CONTEXT_HINTS.items():
        if any(p.search(headline) for p in pats):
            votes[sector] += 0.5
        head_tok = SECTOR_HEAD_TOKENS.get(sector, "")
        if head_tok:
            for alias, sec in ALIAS_TO_SECTOR.items():
                if sec != sector:
                    continue
                if proximity_boost(text, alias, head_tok, window=8):
                    votes[sector] += 0.25
    if not votes:
        for sector, pats in CONTEXT_HINTS.items():
            if any(p.search(text) for p in pats):
                votes[sector] += 0.5


    labels = {sec: float(sc) for sec, sc in votes.items() if sc > 0}
    return labels

def main():
    if not IN_CSV.exists():
        raise SystemExit(f"[ERROR] Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    rows = []
    for _, r in df.iterrows():
        labels = score_article(r.get("headline"), r.get("abstract"))
        if not labels:
            continue
        for sec, sc in labels.items():
            rows.append({
                "article_id": r["article_id"],
                "field": sec,
                "field_score": sc,
                "et_date": r.get("et_date"),
                "url": r.get("url"),
                "headline": r.get("headline"),
                "abstract": r.get("abstract"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print(f"[WRITE] {OUT_CSV} rows=0 articles=0 (no matches)")
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        return

    out = out.sort_values(["et_date", "field", "article_id"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[WRITE] {OUT_CSV} rows={len(out):,} articles={out['article_id'].nunique():,}")

if __name__ == "__main__":
    main()
