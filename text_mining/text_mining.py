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

