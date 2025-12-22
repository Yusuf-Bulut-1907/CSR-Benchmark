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

from stats_and_cleaning import get_cleaned_corpus

documents, metadata = get_cleaned_corpus()


# Cleaning before tokenization

def clean_text(text):
    # Remove HTML tags
    text = bs4.BeautifulSoup(text, "html.parser").get_text()
    
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove boilerplate (simple heuristic: remove lines with less than 3 words, e.g., all right reserved, navigation links, etc.)
    text = '\n'.join([line for line in text.split('\n') if len(line.split()) >= 3])

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove characters that are not letters or whitespace
    text = re.sub(r"[^a-zà-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

# Apply cleaning to all documents
cleaned_documents = [clean_text(doc) for doc in documents]

# Tokenization