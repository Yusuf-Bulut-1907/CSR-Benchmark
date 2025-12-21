import bs4
import re
import unicodedata
import nltk
import pandas as pd
import numpy as np
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
pd = pd.DataFrame(metadata) # convert metadata to dataframe for easier handling
                            # we don't need to do this for documents as they are just text and libraries like scikit-learn can handle lists of strings directly

# try if it works
#print(len(documents))
#print(metadata[0])
#print(documents[0][:300])

#============================
# Statistics about the corpus
#============================

# Number of documents
print(f"Number of documents: {len(documents)}")
# Average document length
avg_doc_length = np.mean([len(doc.split()) for doc in documents]).round(2)
print(f"Average document length: {avg_doc_length}", "words")
# Maximum and minimum document length
max_doc_length = np.max([len(doc.split()) for doc in documents])
min_doc_length = np.min([len(doc.split()) for doc in documents])
print(f"Maximum document length: {max_doc_length}"," (The title of the longest document is:", metadata[np.argmax([len(doc.split()) for doc in documents])]['title']+")")
print(f"Minimum document length: {min_doc_length}"," (The title of the shortest document is:", metadata[np.argmin([len(doc.split()) for doc in documents])]['title']+")")
print(f"Text of the shortest document:", documents[np.argmin([len(doc.split()) for doc in documents])])
# Vocabulary size (it corresponds)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size = len(vectorizer.get_feature_names_out())
print(f"Vocabulary size: {vocab_size}")

# print all the names of the documents and the name of the companies and the number of the documents
#for i, meta in enumerate(metadata):
#    print(f"{i+1}. Title: {meta['title']}, Company: {meta['company']}") 

