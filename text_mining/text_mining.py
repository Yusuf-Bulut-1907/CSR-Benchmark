import bs4
import re
import unicodedata
import nltk
import pandas as pd
import os
import json
import string
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

from load_corpus import load_corpus

documents, metadata = load_corpus()

print(len(documents))
print(metadata[0])
print(documents[0][:300])

df = pd.DataFrame({
    "text": documents,
    "company": [m.get("company") for m in metadata],
    "url": [m.get("url") for m in metadata],
    "title": [m.get("title") for m in metadata]
})

print(df.head())

import spacy

nlp = spacy.load("en_core_web_md", disable=["ner", "parser"]) # on garde le tokenizer et le lemmatizer
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text) # Remove htlm tag
    text = re.sub(r"http\S+|www\S+", " ", text) #remove url
    text = text.translate(str.maketrans("", "", string.punctuation)) #remove ponctuation
    text = re.sub(r"\d+", " ", text) #remove les nombres
    return text
def preprocess_doc(doc):
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.lemma_ not in stop_words
        and len(token.lemma_) > 2
    ]
    return tokens
texts_clean = df["text"].apply(clean_text).tolist()

docs = nlp.pipe(texts_clean, batch_size=100)

tokens_list = []

for i, doc in enumerate(docs):
    if i % 500 == 0:
        print(f"Processed {i} documents")

    tokens_list.append(preprocess_doc(doc))

df["tokens"] = tokens_list

print(df["tokens"].head())

#=====================
#### TF-IDF ###
#=====================

df["tokens"]

df["text_processed"] = df["tokens"].apply(lambda x: " ".join(x)) #des chaines de caractères, pas des listes pour td-idf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    min_df=5,       # élimine les mots trop rares (réduit le bruit)
    max_df=0.8,     # élimine les mots trop fréquents
    ngram_range=(1, 2)  # mots seuls + bigrammes (climate change for instance)
)
X_tfidf = tfidf_vectorizer.fit_transform(df["text_processed"])

print("TF-IDF matrix shape:", X_tfidf.shape)
