import pandas as pd

top = pd.read_csv("../data/lda_topic_topwords.csv")
docw = pd.read_csv("../data/lda_doc_topic_weights.csv")
slim = pd.read_csv("../data/nyt_articles_slim.csv")

# Example: pick topic 17, show top words + sample headlines
print(top.query("topic==17").head(15))
sample = (docw.merge(slim[["article_id","headline"]], on="article_id")
              .nlargest(10, "topic_17")[["headline","topic_17"]])
print(sample)
