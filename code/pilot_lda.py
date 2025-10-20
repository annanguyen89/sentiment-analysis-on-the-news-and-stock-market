import argparse
import os
import re
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def _lazy_imports_for_preprocess():
    import spacy
    from nltk.corpus import stopwords
    from gensim.models.phrases import Phrases, Phraser
    from gensim.utils import simple_preprocess
    return spacy, stopwords, Phrases, Phraser, simple_preprocess

def _lazy_imports_for_build():
    from gensim.corpora import Dictionary
    return Dictionary

def _lazy_imports_for_lda():
    from gensim.models.ldamodel import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    return LdaModel, CoherenceModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
MODEL_DIR = ROOT.parent / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SLIM_CSV           = DATA_DIR / "nyt_articles_slim_2016.csv"
OUT_TEXTS_PARQUET  = DATA_DIR / "lda_texts.parquet"
OUT_TOK_PARQUET    = DATA_DIR / "lda_tokens.parquet"
DICT_PKL           = MODEL_DIR / "lda_dict.pkl"
CORPUS_PKL         = MODEL_DIR / "lda_corpus.pkl"
DOC_INDEX_CSV      = DATA_DIR / "lda_doc_index.csv"
TOPIC_WORDS_CSV    = DATA_DIR / "lda_topic_topwords.csv"
DOC_TOPIC_WEIGHTS  = DATA_DIR / "lda_doc_topic_weights.csv"
FINAL_LDA_PATH     = MODEL_DIR / "lda_model.lda"   

# 1) PREPARE: combine headline + abstract
def step_prepare(slim_csv=SLIM_CSV, out_parquet=OUT_TEXTS_PARQUET):
    df = pd.read_csv(slim_csv)
    def join_text(row):
        h = str(row.get("headline", "") or "")
        a = str(row.get("abstract", "") or "")
        return (h + " " + a).strip()
    df["text"] = df.apply(join_text, axis=1)
    df = df[df["text"].str.len() > 0].copy()
    df = df[["article_id","text","et_date"]]
    df.to_parquet(out_parquet, index=False)
    print(f"[prepare] Wrote: {out_parquet}  rows={len(df):,}")


# 2) PREPROCESS: clean, lemmatize, bigrams
def step_preprocess(in_parquet=OUT_TEXTS_PARQUET, out_parquet=OUT_TOK_PARQUET,
                    min_bigram_count=20, bigram_threshold=10):
    spacy, stopwords, Phrases, Phraser, simple_preprocess = _lazy_imports_for_preprocess()

    EN_STOP = set(stopwords.words("english"))
    EN_STOP.update({"said","mr","ms","would","could","also","new","york","times"})

    nlp = spacy.load("en_core_web_sm", disable=["ner","parser","textcat"])

    def clean_and_tokenize(s: str):
        s = s.lower()
        s = re.sub(r"http\S+|www\.\S+", " ", s)
        s = re.sub(r"[^a-z\s]", " ", s)
        toks = simple_preprocess(s, deacc=True, min_len=2, max_len=20)
        doc = nlp(" ".join(toks))
        keep = []
        for t in doc:
            if t.is_stop or t.lemma_ in EN_STOP:
                continue
            if t.pos_ in {"PUNCT","SPACE","SYM"}:
                continue
            lemma = t.lemma_.strip()
            if len(lemma) < 2:
                continue
            keep.append(lemma)
        return keep

    df = pd.read_parquet(in_parquet)
    tqdm.pandas(desc="tokenizing")
    df["tokens"] = df["text"].progress_apply(clean_and_tokenize)

    # bigrams
    phrases = Phrases(df["tokens"], min_count=min_bigram_count, threshold=bigram_threshold)
    bigram = Phraser(phrases)
    df["tokens"] = df["tokens"].apply(lambda xs: bigram[xs])

    df[["article_id","tokens","et_date"]].to_parquet(out_parquet, index=False)
    print(f"[preprocess] Wrote: {out_parquet}  rows={len(df):,}")


# 3) BUILD: dictionary + corpus + doc index
def step_build(tok_parquet=OUT_TOK_PARQUET, dict_pkl=DICT_PKL, corpus_pkl=CORPUS_PKL,
               doc_index_csv=DOC_INDEX_CSV, no_below=20, no_above=0.4, keep_n=100000):
    Dictionary = _lazy_imports_for_build()

    df = pd.read_parquet(tok_parquet)
    texts = df["tokens"].tolist()

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    corpus = [dictionary.doc2bow(t) for t in texts]

    with open(dict_pkl, "wb") as f: pickle.dump(dictionary, f)
    with open(corpus_pkl, "wb") as f: pickle.dump(corpus, f)

    df[["article_id","et_date"]].assign(doc_id=range(len(df))).to_csv(doc_index_csv, index=False)
    print(f"[build] Dictionary size: {len(dictionary):,} | Corpus docs: {len(corpus):,}")
    print(f"[build] Saved: {dict_pkl}, {corpus_pkl}, {doc_index_csv}")


# 4) SELECT_K: try K grid, print coherence
def step_select_k(kgrid, dict_pkl=DICT_PKL, corpus_pkl=CORPUS_PKL, tok_parquet=OUT_TOK_PARQUET,
                  passes=5, iterations=1000, eval_every=2000, random_state=42):
    LdaModel, CoherenceModel = _lazy_imports_for_lda()
    texts = pd.read_parquet(tok_parquet)["tokens"].tolist()
    with open(dict_pkl, "rb") as f: dictionary = pickle.load(f)
    with open(corpus_pkl, "rb") as f: corpus = pickle.load(f)

    results = []
    for K in kgrid:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=int(K),
            random_state=random_state,
            passes=passes,
            iterations=iterations,
            eval_every=eval_every,
            alpha="symmetric",
            eta="auto"
        )
        coh = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
        results.append((int(K), coh))
        print(f"[select_k] K={int(K):2d} | coherence c_v={coh:.4f}")

    bestK, bestC = max(results, key=lambda x: x[1])
    print(f"[select_k] BEST: K={bestK} (c_v={bestC:.4f})")


# 5) TRAIN_FINAL: train & save final model
def step_train_final(k, dict_pkl=DICT_PKL, corpus_pkl=CORPUS_PKL,
                     out_path=FINAL_LDA_PATH, passes=10, iterations=1500, random_state=123):
    LdaModel, _ = _lazy_imports_for_lda()
    with open(dict_pkl, "rb") as f: dictionary = pickle.load(f)
    with open(corpus_pkl, "rb") as f: corpus = pickle.load(f)

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=int(k),
        random_state=random_state,
        passes=passes,
        iterations=iterations,
        eval_every=None,
        alpha="symmetric",
        eta="auto"
    )
    out_path = Path(out_path)
    if out_path.name == "lda_model.lda":
        out_path = out_path.with_name(f"lda_model_K{int(k)}.lda")

    lda.save(str(out_path))
    print(f"[train_final] Saved LDA(K={k}) -> {out_path}")
    return out_path 

# 6) EXPORT_TOPICS: topic→top words
def step_export_topics(lda_path=FINAL_LDA_PATH, dict_pkl=DICT_PKL, out_csv=TOPIC_WORDS_CSV, topn=30):
    LdaModel, _ = _lazy_imports_for_lda()
    with open(dict_pkl, "rb") as f: dictionary = pickle.load(f)
    lda = LdaModel.load(str(lda_path))

    rows = []
    for k in range(lda.num_topics):
        term_probs = lda.get_topic_terms(k, topn=topn)
        words = [(dictionary[id], float(p)) for id, p in term_probs]
        for rank, (w, p) in enumerate(words, 1):
            rows.append({"topic": k, "rank": rank, "word": w, "prob": p})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[export_topics] Wrote {out_csv}")


# 7) DOC_TOPICS: dense doc→topic matrix
def step_doc_topics(lda_path=FINAL_LDA_PATH, dict_pkl=DICT_PKL, corpus_pkl=CORPUS_PKL,
                    doc_index_csv=DOC_INDEX_CSV, out_csv=DOC_TOPIC_WEIGHTS):
    LdaModel, _ = _lazy_imports_for_lda()
    with open(dict_pkl, "rb") as f: dictionary = pickle.load(f)
    with open(corpus_pkl, "rb") as f: corpus = pickle.load(f)
    doc_index = pd.read_csv(doc_index_csv)
    lda = LdaModel.load(str(lda_path))

    rows = []
    for doc_id, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        probs = [0.0] * lda.num_topics
        for k, p in dist:
            probs[k] = float(p)
        row = {"doc_id": doc_id, **{f"topic_{k}": probs[k] for k in range(lda.num_topics)}}
        rows.append(row)

    wide = pd.DataFrame(rows)
    out = doc_index.merge(wide, on="doc_id", how="left").drop(columns=["doc_id"])
    out.to_csv(out_csv, index=False)
    print(f"[doc_topics] Wrote doc→topic weights: {out_csv}")


# 8) ALL: run everything
def step_all(k, kgrid, paths):
    step_prepare(paths.slim_csv, paths.texts_parquet)
    step_preprocess(paths.texts_parquet, paths.tok_parquet)
    step_build(paths.tok_parquet, paths.dict_pkl, paths.corpus_pkl, paths.doc_index_csv)
    step_select_k(kgrid, paths.dict_pkl, paths.corpus_pkl, paths.tok_parquet)
    saved_path = step_train_final(k, paths.dict_pkl, paths.corpus_pkl, paths.lda_path)
    step_export_topics(saved_path, paths.dict_pkl, paths.topic_words_csv)
    step_doc_topics(saved_path, paths.dict_pkl, paths.corpus_pkl, paths.doc_index_csv, paths.doc_topic_csv)

def parse_args():
    p = argparse.ArgumentParser(description="One-file LDA pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_path_flags(sp):
        sp.add_argument("--slim_csv", default=str(SLIM_CSV))
        sp.add_argument("--texts_parquet", default=str(OUT_TEXTS_PARQUET))
        sp.add_argument("--tok_parquet", default=str(OUT_TOK_PARQUET))
        sp.add_argument("--dict_pkl", default=str(DICT_PKL))
        sp.add_argument("--corpus_pkl", default=str(CORPUS_PKL))
        sp.add_argument("--doc_index_csv", default=str(DOC_INDEX_CSV))
        sp.add_argument("--lda_path", default=str(FINAL_LDA_PATH))
        sp.add_argument("--topic_words_csv", default=str(TOPIC_WORDS_CSV))
        sp.add_argument("--doc_topic_csv", default=str(DOC_TOPIC_WEIGHTS))

    sp = sub.add_parser("prepare")
    add_common_path_flags(sp)

    sp = sub.add_parser("preprocess")
    sp.add_argument("--min_bigram_count", type=int, default=20)
    sp.add_argument("--bigram_threshold", type=float, default=10.0)
    add_common_path_flags(sp)

    sp = sub.add_parser("build")
    sp.add_argument("--no_below", type=int, default=20)
    sp.add_argument("--no_above", type=float, default=0.4)
    sp.add_argument("--keep_n", type=int, default=100000)
    add_common_path_flags(sp)

    sp = sub.add_parser("select_k")
    sp.add_argument("--kgrid", type=int, nargs="+", default=[20,30,40,50,60])
    sp.add_argument("--passes", type=int, default=5)
    sp.add_argument("--iterations", type=int, default=1000)
    sp.add_argument("--eval_every", type=int, default=2000)
    sp.add_argument("--random_state", type=int, default=42)
    add_common_path_flags(sp)

    sp = sub.add_parser("train_final")
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--passes", type=int, default=10)
    sp.add_argument("--iterations", type=int, default=1500)
    sp.add_argument("--random_state", type=int, default=123)
    add_common_path_flags(sp)

    sp = sub.add_parser("export_topics")
    sp.add_argument("--topn", type=int, default=30)
    add_common_path_flags(sp)

    sp = sub.add_parser("doc_topics")
    add_common_path_flags(sp)

    sp = sub.add_parser("all")
    sp.add_argument("--k", type=int, required=True, help="final K")
    sp.add_argument("--kgrid", type=int, nargs="+", default=[20,30,40,50,60])
    add_common_path_flags(sp)

    return p.parse_args()

class Paths:
    def __init__(self, ns):
        self.slim_csv = Path(ns.slim_csv)
        self.texts_parquet = Path(ns.texts_parquet)
        self.tok_parquet = Path(ns.tok_parquet)
        self.dict_pkl = Path(ns.dict_pkl)
        self.corpus_pkl = Path(ns.corpus_pkl)
        self.doc_index_csv = Path(ns.doc_index_csv)
        self.lda_path = Path(ns.lda_path)
        self.topic_words_csv = Path(ns.topic_words_csv)
        self.doc_topic_csv = Path(ns.doc_topic_csv)

def main():
    args = parse_args()
    paths = Paths(args)

    if args.cmd == "prepare":
        step_prepare(paths.slim_csv, paths.texts_parquet)

    elif args.cmd == "preprocess":
        step_preprocess(paths.texts_parquet, paths.tok_parquet,
                        args.min_bigram_count, args.bigram_threshold)

    elif args.cmd == "build":
        step_build(paths.tok_parquet, paths.dict_pkl, paths.corpus_pkl, paths.doc_index_csv,
                   args.no_below, args.no_above, args.keep_n)

    elif args.cmd == "select_k":
        step_select_k(args.kgrid, paths.dict_pkl, paths.corpus_pkl, paths.tok_parquet,
                      args.passes, args.iterations, args.eval_every, args.random_state)

    elif args.cmd == "train_final":
        step_train_final(args.k, paths.dict_pkl, paths.corpus_pkl, paths.lda_path,
                         args.passes, args.iterations, args.random_state)

    elif args.cmd == "export_topics":
        step_export_topics(paths.lda_path, paths.dict_pkl, paths.topic_words_csv, args.topn)

    elif args.cmd == "doc_topics":
        step_doc_topics(paths.lda_path, paths.dict_pkl, paths.corpus_pkl, paths.doc_index_csv, paths.doc_topic_csv)

    elif args.cmd == "all":
        step_all(args.k, args.kgrid, paths)

if __name__ == "__main__":
    main()
