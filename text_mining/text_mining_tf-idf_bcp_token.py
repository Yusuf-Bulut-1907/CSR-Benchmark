# =====================
# IMPORTS
# =====================
import os
import re
import spacy
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from spacy.lang.en.stop_words import STOP_WORDS
from stats_and_cleaning import get_cleaned_corpus

# =====================
# OUTPUT FOLDER
# =====================
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================
# STOPWORDS
# =====================
CUSTOM_STOPWORDS = {
    "company","business","group","corporate","organization",
    "report","annual","financial","strategy","approach",
    "management","value","activity","operation",
    "stakeholder","objective","commitment","initiative",
    "policy","framework","implementation",
    "page","website","information","content","document",
    "download","read","visit","contact",
    "cookie","cookies","privacy","gdpr","consent",
    "device","browser","session","personal","datum",
    "linkedin","facebook","instagram","twitter"
}

STOP_WORDS.update(CUSTOM_STOPWORDS)

# =====================
# NLP MODEL
# =====================
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

# =====================
# LOAD CORPUS
# =====================
documents, metadata = get_cleaned_corpus()

df = pd.DataFrame({
    "text": documents,
    "company": [m.get("company") for m in metadata]
})

print(f"📄 Documents : {len(df)}")
print(f"🏢 Entreprises uniques : {df['company'].nunique()}")

# =====================
# LINGUISTIC N-GRAMS
# =====================
def extract_bigrams(doc):
    bigrams = []
    for i in range(len(doc) - 1):
        t1, t2 = doc[i], doc[i+1]
        if (
            t1.is_alpha and t2.is_alpha
            and not t1.is_stop and not t2.is_stop
            and (
                (t1.pos_ == "ADJ" and t2.pos_ == "NOUN") or
                (t1.pos_ == "NOUN" and t2.pos_ == "NOUN") or
                (t1.pos_ == "PROPN" and t2.pos_ == "NOUN")
            )
        ):
            bigrams.append(f"{t1.lemma_}_{t2.lemma_}")
    return bigrams


def extract_trigrams(doc):
    trigrams = []
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i+1], doc[i+2]
        if (
            t1.pos_ in {"NOUN", "PROPN"}
            and t2.pos_ in {"NOUN", "PROPN"}
            and t3.pos_ in {"NOUN", "PROPN"}
            and not (t1.is_stop or t2.is_stop or t3.is_stop)
        ):
            trigrams.append(
                f"{t1.lemma_}_{t2.lemma_}_{t3.lemma_}"
            )
    return trigrams


# =====================
# CLEAN + LEMMATIZE
# =====================
def clean_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    doc = nlp(text)

    unigrams = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token.lemma_) > 2
    ]

    bigrams = extract_bigrams(doc)
    trigrams = extract_trigrams(doc)

    return " ".join(unigrams + bigrams + trigrams)


print("🚿 Cleaning & lemmatization...")
df["text_processed"] = df["text"].apply(clean_and_lemmatize)

# =====================
# AGGREGATE BY COMPANY
# =====================
df_company = (
    df.groupby("company")["text_processed"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)

print(f"✅ Corpus agrégé : {len(df_company)} entreprises")

# =====================
# COUNT VECTORIZER (UNIGRAMS ONLY)
# =====================
cv = CountVectorizer(
    ngram_range=(1, 1),
    min_df=0.05,
    max_df=0.7
)

X_counts = cv.fit_transform(df_company["text_processed"])
features = cv.get_feature_names_out()

print("📐 TDM shape :", X_counts.shape)

# =====================
# TF-IDF UNIGRAMS
# =====================
tfidf_uni = TfidfVectorizer(vocabulary=features)
X_tfidf_uni = tfidf_uni.fit_transform(df_company["text_processed"])

df_tfidf_uni = pd.DataFrame(
    X_tfidf_uni.toarray(),
    index=df_company["company"],
    columns=features
)

df_tfidf_uni.to_csv(f"{OUTPUT_FOLDER}/TFIDF_unigram.csv")
print("💾 TF-IDF unigrams exported")

# =====================
# TF-IDF UNI + BI + TRI
# =====================
tfidf_full = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=0.03,
    max_df=0.85
)

X_tfidf_full = tfidf_full.fit_transform(df_company["text_processed"])
full_features = tfidf_full.get_feature_names_out()

df_tfidf_full = pd.DataFrame(
    X_tfidf_full.toarray(),
    index=df_company["company"],
    columns=full_features
)

df_tfidf_full.to_csv(f"{OUTPUT_FOLDER}/TFIDF_unigram_bigram_trigram.csv")
print(f"📊 Dimensions de la matrice TF-IDF complète : {X_tfidf_full.shape}")
print("💾 TF-IDF uni/bi/tri exported")

# =====================
# LSA (TF-IDF → 100 DIM)
# =====================
svd = TruncatedSVD(n_components=100, random_state=42)
normalizer = Normalizer(copy=False)

lsa = make_pipeline(svd, normalizer)
X_lsa = lsa.fit_transform(X_tfidf_full)

df_lsa = pd.DataFrame(
    X_lsa,
    index=df_company["company"],
    columns=[f"LSA_{i}" for i in range(100)]
)

df_lsa.to_csv(f"{OUTPUT_FOLDER}/TFIDF_LSA_100.csv")
print("💾 LSA (100 dims) exported")

print("🎉 Pipeline completed successfully")