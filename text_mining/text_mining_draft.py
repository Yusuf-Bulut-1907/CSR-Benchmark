import bs4
import re
import unicodedata
import nltk
import spacy
import pandas as pd
import numpy as np
import os
import json
import string
from collections import Counter
from scipy.sparse import hstack

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer

from stats_and_cleaning import get_cleaned_corpus


custom_stopwords = {"company", "business", "report", "year", "page", "website",
    "provide", "include", "information", "service", "client", "group,", "pdf", "site",
    "download"}

custom_stopwords = set(nltk.corpus.stopwords.words("english")).union(custom_stopwords)

# =====================
# LOAD CORPUS
# =====================

documents, metadata = get_cleaned_corpus()

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
    # cleaning
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

print("🚿 Cleaning + lemmatization processing ...(method 1)")
df["text_processed"] = df["text"].apply(clean_and_lemmatize)

#====================
# METHOD 2 : NLTK + PORTER STEMMER
#====================
def nltk_porter_stemmer(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = word_tokenize(text)
    porter = PorterStemmer()
    stemmed_tokens = [
        porter.stem(token)
        for token in tokens
        if token.isalpha()
        and token not in custom_stopwords
        and len(token) > 2
    ]

    return " ".join(stemmed_tokens)

print("🚿 Cleaning + stemming processing ... (method 2)")
df["text_stemmed"] = df["text"].apply(nltk_porter_stemmer)

#====================
# METHOD 3 : SPLIT + BASIC CLEANING
# =====================
def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    cleaned_tokens = [
        token
        for token in tokens
        if token.isalpha()
        and token not in custom_stopwords
        and len(token) > 2
    ]

    return " ".join(cleaned_tokens)

print("🚿 Basic cleaning processing ... (method 3)")
df["text_basic_cleaned"] = df["text"].apply(basic_cleaning)

#====================
# GROUP BY COMPANY
#=====================
df_company = (
    df.groupby("company").agg({
        "text_processed": lambda x: " ".join(x),
        "text_stemmed": lambda x: " ".join(x),
        "text_basic_cleaned": lambda x: " ".join(x)
    }).reset_index()
)

print(f"✅ Agregated Corpus : {len(df_company)} entreprises ")

# =====================
# BEST METHOD SELECTION
# =====================
def evaluate_tokens(list_of_tokens):
    all_tokens = [token for doc in list_of_tokens for token in doc] if isinstance(list_of_tokens[0], list) else list_of_tokens
    avg_len = sum(len(doc) if isinstance(doc, list) else len([token for token in list_of_tokens]) for doc in list_of_tokens) / max(len(list_of_tokens), 1)
    unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)
    return avg_len, unique_ratio

metrics = {}
for method in ["tokens_spacy", "tokens_nltk", "tokens_split"]:
    tokens_list = df_company[method].tolist()
    metrics[method] = evaluate_tokens(tokens_list)

# Print evaluation metrics
for method, (avg_len, unique_ratio) in metrics.items():
    print(f"{method}: avg_tokens={avg_len:.2f}, unique_ratio={unique_ratio:.2f}")

best_method = max(metrics, key=lambda m: (metrics[m][1], metrics[m][0]))
print(f"Best method selected: {best_method}")




