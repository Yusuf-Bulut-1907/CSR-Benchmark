import re
import nltk
import spacy
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack

from load_corpus import load_corpus

nltk.download("stopwords")
nltk.download("punkt")
custom_stopwords = {
    "company", "business", "report", "year", "page", "website", "provide", 
    "include", "information", "service", "client", "group", "pdf", "site", 
    "download", "consent", "cookie", "use", "data", "also", "may", "one", 
    "new", "us", "cookies", "cooky", "privacy", "device", "browser",
    "collect", "processing", "purpose", "request", "contact", "law", 
    "advertising", "third", "party", "personal", "identify", "storage", 
    "que", "und", "para", "die", "siemens", "les", "des", "von", "com", "der"
}
custom_stopwords = set(nltk.corpus.stopwords.words("english")).union(custom_stopwords)
# =====================
# LOAD CORPUS
# =====================
documents, metadata = load_corpus()

df = pd.DataFrame({
    "text": documents,
    "company": [m.get("company") for m in metadata]
})

print(f"📄 Files loaded : {len(df)}")
print(f"🏢 Unique companies : {df['company'].nunique()}")

# =====================
# CLEANING + LEMMATIZATION (SPACY)
# =====================
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

def clean_and_lemmatize(text):
    # nettoyage physique
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.lemma_ not in custom_stopwords
        and len(token.lemma_) > 2
    ]

    return " ".join(tokens)

print("🚿 Cleaning + lemmatization processing ...")
df["text_processed"] = df["text"].apply(clean_and_lemmatize)

# =====================
# GROUP BY COMPANY
# =====================
df_company = (
    df.groupby("company")["text_processed"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)

n_companies = len(df_company)
print(f"✅ Agregated Corpus : {n_companies} entreprises ")

# =====================
# TDM : UNIGRAMS + BIGRAMS
# =====================

# ---- Unigrams ----
cv_uni = CountVectorizer(
    ngram_range=(1, 1),
    min_df=0.03,   # ≥ 3% of companies 
    max_df=0.85
)

X_uni = cv_uni.fit_transform(df_company["text_processed"])
uni_features = cv_uni.get_feature_names_out()

# ---- Bigrams ----
cv_bi = CountVectorizer(
    ngram_range=(2, 2),
    min_df=0.04,   # ≥ 4% of companies → strong relevance
    max_df=0.85
)

X_bi = cv_bi.fit_transform(df_company["text_processed"])
bi_features = cv_bi.get_feature_names_out()

cv_tri = CountVectorizer(
    ngram_range=(3, 3),
    min_df=0.05,   # ≥ 5% of companies → very strong relevance
    max_df=0.85
)
X_tri = cv_tri.fit_transform(df_company["text_processed"])
tri_features = cv_tri.get_feature_names_out()

# ---- Fusion ----
X_tdm = hstack([X_uni, X_bi, X_tri])
features = np.concatenate([uni_features, bi_features, tri_features])

print("📐 TDM shape :", X_tdm.shape)

# =====================
# EXPORT TDM CSV
# =====================
df_tdm = pd.DataFrame(
    X_tdm.toarray(),
    index=df_company["company"],
    columns=features
)

df_tdm.to_csv("data/TDM_unigram_bigram_trigram.csv")
print("💾 TDM exportée (CSV)")

# =====================
# TF-IDF COMPUTATION
# =====================
tfidf = TfidfVectorizer(
    vocabulary=features
)

X_tfidf = tfidf.fit_transform(df_company["text_processed"])

df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    index=df_company["company"],
    columns=features
)

uni_tfidf = TfidfVectorizer(
    vocabulary=uni_features)
X_uni_tfidf = uni_tfidf.fit_transform(df_company["text_processed"])
df_uni_tfidf = pd.DataFrame(
    X_uni_tfidf.toarray(),
    index=df_company["company"],
    columns=uni_features
)
df_uni_tfidf.to_csv("data/TFIDF_unigram.csv")
print("💾 TF-IDF Unigrams exported (CSV)")

df_tfidf.to_csv("data/TFIDF_unigram_bigram_trigram.csv")
print("💾 TF-IDF exported (CSV)")

print("🎉 Pipeline done with success!")