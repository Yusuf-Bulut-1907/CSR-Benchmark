import os
import json
import re
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# 1️⃣ NLTK resources
# -----------------------------
NLTK_RESOURCES = ["stopwords", "punkt", "wordnet"]
for res in NLTK_RESOURCES:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "data/rse_pages"
OUTPUT_FILE = "clean_corpus.json"
MIN_TEXT_LENGTH = 300
LANGUAGE = "english"
STOPWORDS = set(stopwords.words(LANGUAGE))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------
def remove_encoding_artifacts(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\x0c", " ")
    return text

def remove_lists_and_breaks(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[•▪●◦]", " ", text)
    return text

def remove_urls_and_emails(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text)
    return text

def remove_non_alpha(text: str) -> str:
    return re.sub(r"[^A-Za-z\s]", " ", text)

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def tokenize_and_filter(text: str) -> str:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

def clean_text_pipeline(text: str) -> str:
    text = remove_encoding_artifacts(text)
    text = remove_lists_and_breaks(text)
    text = remove_urls_and_emails(text)
    text = remove_non_alpha(text)
    text = normalize_spaces(text)
    text = tokenize_and_filter(text)
    return text

# -----------------------------
# MAIN PROCESS
# -----------------------------
def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_clean_corpus() -> List[Dict]:
    corpus = []

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".json"):
            continue

        company = file.replace("_rse.json", "")
        pages = load_json(os.path.join(INPUT_DIR, file))

        for page in pages:
            # 🔹 concaténation title + subtitles + text
            raw_text_parts = [page.get("title", "")]
            subtitles = page.get("subtitles", [])
            if isinstance(subtitles, list):
                raw_text_parts += subtitles
            raw_text_parts.append(page.get("text", ""))

            raw_text = " ".join(raw_text_parts)

            if len(raw_text) < MIN_TEXT_LENGTH:
                continue

            clean_text = clean_text_pipeline(raw_text)

            if len(clean_text) < 100:
                continue

            corpus.append({
                "company": company,
                "url": page.get("url", ""),
                "clean_text": clean_text
            })

    return corpus

def save_corpus(corpus: List[Dict]):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    corpus = build_clean_corpus()
    save_corpus(corpus)
    print(f"✅ Clean corpus saved: {len(corpus)} documents → {OUTPUT_FILE}")