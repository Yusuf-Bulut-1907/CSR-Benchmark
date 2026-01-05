"""
Text Representation Pipeline for CSR / ESG Analysis
--------------------------------------------------

This script transforms a cleaned textual corpus into structured
text representations suitable for quantitative analysis.

Main steps:
- Linguistic cleaning and lemmatization using spaCy
- Extraction of linguistically meaningful unigrams, bigrams, and trigrams
- Aggregation of documents at the company level
- Construction of Term-Document Matrices (TDM)
- Computation of TF-IDF representations

The resulting matrices are exported for downstream analysis
(e.g., clustering, topic modeling, similarity analysis).
"""

import re
import spacy
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from spacy.lang.en.stop_words import STOP_WORDS

from stats_and_cleaning import get_cleaned_corpus

# ====================
# Output folder
# ====================

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)


# Redundant safety check to ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")


# ====================
# Custom stopwords
# ====================

# Domain-specific stopwords capturing generic corporate,
# legal, GDPR-related, and multilingual noise terms
custom_stopwords = {
    "company", "business", "report", "year", "page", "website", "provide", 
    "include", "information", "service", "client", "group", "pdf", "site", 
    "download", "consent", "cookie", "use", "data", "also", "may", "one", 
    "new", "us", "cookies", "cooky", "privacy", "device", "browser",
    "collect", "processing", "purpose", "request", "contact", "law", 
    "advertising", "third", "party", "personal", "identify", "storage", 
    "que", "und", "para", "die", "siemens", "les", "des", "von", "com", "der",
    "gdpr", "privacy", "visit","read", "learn", "access", "store", "session","legal","datum",
    "opt","applicable","necessary","interest","notice","email","performance","functionality",
    "personal_datum","privacy_policy", "adresse", "linkedin","facebook","instagram","twitter","setting", "settings"

}

# Extend spaCy's default stopword list with domain-specific terms
STOP_WORDS.update(custom_stopwords)

# Load spaCy English model with lightweight pipeline
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

# =====================
# LOAD CLEANED CORPUS
# =====================
documents, metadata = get_cleaned_corpus()

df = pd.DataFrame({
    "text": documents,
    "company": [m.get("company") for m in metadata]
})

print(f"📄 Files loaded : {len(df)}")
print(f"🏢 Unique companies : {df['company'].nunique()}")

# =====================
# CLEANING + LEMMATIZATION 
# =====================
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

def extract_noun_trigrams(doc):
    """
    Extract noun-based trigrams:
    - NOUN–NOUN–NOUN
    - PROPN–NOUN–NOUN

    Trigrams are lemmatized and joined using underscores.
    This targets compound CSR/ESG concepts (e.g., carbon_emission_reduction).
    """
    trigrams = []

    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i+1], doc[i+2]

        if (
            t1.pos_ in {"NOUN", "PROPN"} and
            t2.pos_ in {"NOUN", "PROPN"} and
            t3.pos_ in {"NOUN", "PROPN"}
        ):
            if not (t1.is_stop or t2.is_stop or t3.is_stop):
                trigram = f"{t1.lemma_}_{t2.lemma_}_{t3.lemma_}"
                trigrams.append(trigram)

    return trigrams

def extract_filtered_bigrams(doc):
    """
    Extract linguistically meaningful bigrams:
    - ADJ + NOUN
    - NOUN + NOUN
    - PROPN + NOUN

    These structures capture descriptive and thematic expressions
    typical of sustainability discourse.
    """
    bigrams = []

    for i in range(len(doc) - 1):
        t1, t2 = doc[i], doc[i+1]

        if (
            not t1.is_stop and not t2.is_stop and
            t1.is_alpha and t2.is_alpha and
            (
                (t1.pos_ == "ADJ" and t2.pos_ == "NOUN") or
                (t1.pos_ == "NOUN" and t2.pos_ == "NOUN") or
                (t1.pos_ == "PROPN" and t2.pos_ == "NOUN")
            )
        ):
            bigram = f"{t1.lemma_}_{t2.lemma_}"
            bigrams.append(bigram)

    return bigrams

def clean_and_lemmatize(text):
    """
    Perform surface-level cleaning and linguistic normalization:
    - Lowercasing
    - Removal of URLs, HTML tags, and digits
    - Lemmatization using spaCy
    - Retention of noun-based unigrams, bigrams, and trigrams
    """
     # Basic text cleaning
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    doc = nlp(text)

    # Extract lemmatized noun unigrams
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ == "NOUN"
        and len(token.lemma_) > 2
    ]
    nouns_bigrams = extract_filtered_bigrams(doc)
    nouns_trigrams = extract_noun_trigrams(doc)
    return " ".join(tokens +nouns_bigrams + nouns_trigrams)

print("🚿 Cleaning + lemmatization processing ...")
df["text_processed"] = df["text"].apply(clean_and_lemmatize)

# ====================
# AGGREGATION BY COMPANY
# ====================

# Documents are concatenated at the company level
# Each company becomes a single aggregated document
df_company = (
    df.groupby("company")["text_processed"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)

n_companies = len(df_company)
print(f"✅ Agregated Corpus : {n_companies} entreprises ")

# =====================
# TERM-DOCUMENT MATRIX (TDM)
# =====================

# ---- Unigrams ----
cv_uni = CountVectorizer(
    ngram_range=(1, 1),
    min_df=0.03,   # Tem appears in at least 3% of companies
    max_df=0.85,
    #max_features=3000
)

X_uni = cv_uni.fit_transform(df_company["text_processed"])
uni_features = cv_uni.get_feature_names_out()



# ---- Matrix fusion (only unigrams activated here) ----
X_tdm = hstack([X_uni])
features = np.concatenate([uni_features])

print("📐 TDM shape :", X_tdm.shape)

# =====================
# EXPORT TDM 
# =====================
df_tdm = pd.DataFrame(
    X_tdm.toarray(),
    index=df_company["company"],
    columns=features
)

df_tdm.to_csv("data/TDM_unigram_bigram_trigram.csv")
print("💾 TDM exportée (CSV)")

# ====================
# TF-IDF REPRESENTATION
# ====================

# Global TF-IDF using the previously extracted vocabulary
tfidf = TfidfVectorizer(vocabulary=features)
X_tfidf = tfidf.fit_transform(df_company["text_processed"])

df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    index=df_company["company"],
    columns=features
)

# Separate TF-IDF for unigrams only (explicit export)
uni_tfidf = TfidfVectorizer(vocabulary=uni_features)
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