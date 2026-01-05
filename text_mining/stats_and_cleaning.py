"""
Corpus Statistics and Cleaning Pipeline
---------------------------------------

This script performs descriptive statistics and successive cleaning steps
on a textual corpus previously extracted from CSR / ESG-related web pages.

The objectives are:
- Quantify corpus properties before and after cleaning
- Remove low-quality, noisy, duplicated, or non-English documents
- Produce transparent, reproducible corpus statistics for reporting
"""

import numpy as np
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, DetectorFactory

from load_corpus import load_corpus


# Load raw corpus and metadata
documents, metadata = load_corpus()

# ============================
# Initial statistics
# ============================

# Store the list of companies before any cleaning step
# This allows tracking which companies may be fully removed by filtering
companies_before = set(meta.get("company") for meta in metadata if meta.get("company")) 


# Initial number of documents
num_documents_initial = len(documents)

# Document length distribution (in words)
doc_lengths = [len(doc.split()) for doc in documents]
avg_doc_length_initial = np.mean(doc_lengths).round(2)

# Extreme document lengths
max_doc_length_initial = np.max(doc_lengths)
min_doc_length_initial = np.min(doc_lengths)

# Titles associated with longest and shortest documents
max_doc_title_initial = metadata[np.argmax(doc_lengths)]['title']
min_doc_title_initial = metadata[np.argmin(doc_lengths)]['title']

# Initial vocabulary size (raw corpus)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_initial = len(vectorizer.get_feature_names_out())

# ============================
# Removal of very short documents
# ============================

# Documents shorter than 50 words are removed
# These are typically navigation pages, cookie banners, or empty pages
filtered_documents = []
filtered_metadata = []

for doc, meta in zip(documents, metadata):
    if len(doc.split()) >= 50:
        filtered_documents.append(doc)
        filtered_metadata.append(meta)

documents = filtered_documents
metadata = filtered_metadata

num_documents_after_short = len(documents)

# Vocabulary size after removing short documents
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_after_short = len(vectorizer.get_feature_names_out())


# ============================
# Removal of poorly scraped pages
# ============================

# Regex patterns capturing typical scraping failures or error pages
unwanted_patterns = [
    # Loading / incomplete pages
    r"loading",
    r"please\s+wait",
    r"under\s+construction",
    r"coming\s+soon",

    # HTTP and server errors
    r"404",
    r"403",
    r"500",
    r"502",
    r"503",
    r"http\s+error",
    r"server\s+error",

    # Page not found
    r"page\s+not\s+found",
    r"not\s+found",
    r"an\s+error\s+occurred",
    r"something\s+went\s+wrong",
    r"temporarily\s+unavailable",
    r"service\s+unavailable",

    # Access restriction
    r"access\s+denied",
    r"forbidden",
    r"unauthorized",
    r"permission\s+denied",

    # JavaScript / cookie walls
    r"enable\s+javascript",
    r"cookies\s+required",
    r"verify\s+you\s+are\s+human",
    r"captcha",
]

# Compile regex once for efficiency
cleaned_documents = []
cleaned_metadata = []

pattern_regex = re.compile("|".join(unwanted_patterns), re.IGNORECASE) # Compile regex pattern once for efficiency

for doc, meta in zip(documents, metadata):
    title = meta.get("title", "")
        # Only the beginning of the document is checked to reduce false positives
    text_start = doc[:100].lower()  

    if not pattern_regex.search(title) and not pattern_regex.search(text_start):
        cleaned_documents.append(doc)
        cleaned_metadata.append(meta)

documents = cleaned_documents
metadata = cleaned_metadata

num_documents_after_unwanted = len(documents)

# Vocabulary size after removing unwanted phrases
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_after_unwanted = len(vectorizer.get_feature_names_out())


# ============================
# Duplicate removal
# ============================

# Exact-duplicate removal based on full document text
# This avoids overweighting duplicated pages across websites
unique_documents = []
unique_metadata = []
seen_texts = set()

for doc, meta in zip(documents, metadata):
    if doc not in seen_texts:
        unique_documents.append(doc)
        unique_metadata.append(meta)
        seen_texts.add(doc)

documents = unique_documents
metadata = unique_metadata

num_documents_after_duplicates = len(documents)

# Vocabulary size after removing duplicates
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_after_duplicates = len(vectorizer.get_feature_names_out())


# ============================
# Language detection and filtering
# ============================

# Fix random seed to ensure reproducible language detection
DetectorFactory.seed = 0  # For consistent results
non_english_docs = []

for i, doc in enumerate(documents):
    try:
        # Language detection is applied to a text sample for efficiency
        sample = doc[:500]  
        lang = detect(sample)
        if lang != 'en':
            non_english_docs.append((i, lang, metadata[i].get("title")))
    except:
        # Detection failures are silently ignored
        continue

num_non_english_docs = len(non_english_docs)

# Remove non-English documents from the corpus
final_documents = []
final_metadata = []

non_english_indices = set(i for i, _, _ in non_english_docs)

for i, (doc, meta) in enumerate(zip(documents, metadata)):
    if i not in non_english_indices:
        final_documents.append(doc)
        final_metadata.append(meta)

documents = final_documents
metadata = final_metadata

num_documents_final = len(documents)

# Vocabulary size after removing non-English documents
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_final = len(vectorizer.get_feature_names_out())


# ============================
# Final corpus statistics
# ============================
final_doc_lengths = [len(doc.split()) for doc in documents]
avg_doc_length_final = np.mean(final_doc_lengths).round(2)
max_doc_length_final = np.max(final_doc_lengths)
min_doc_length_final = np.min(final_doc_lengths)


# Document count per company
company_counts = Counter(meta.get("company") for meta in metadata if meta.get("company"))
sorted_companies = company_counts.most_common()  # liste de tuples (company, count)

# Identify companies entirely removed during cleaning
companies_after = set(meta.get("company") for meta in metadata if meta.get("company"))
removed_companies = companies_before - companies_after


# ============================
# Public accessor
# ============================
def get_cleaned_corpus():
    """
    Returns the fully cleaned corpus and associated metadata.
    """
    return documents, metadata


if __name__ == "__main__":

    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    # Download txt file with corpus statistics
    with open("results/corpus_statistics.txt", "w", encoding="utf-8") as f:
        f.write(f"Number of documents: {num_documents_initial}\n")
        f.write(f"Average document length: {avg_doc_length_initial} words\n")
        f.write(
            f"Maximum document length: {max_doc_length_initial} "
            f"({max_doc_title_initial})\n"
        )
        f.write(
            f"Minimum document length: {min_doc_length_initial} "
            f"({min_doc_title_initial})\n"
        )
        f.write(f"Vocabulary size: {vocab_size_initial} unique words\n")

        f.write(f"\nNumber of documents after removing short documents: {num_documents_after_short}\n")
        f.write(f"Vocabulary size: {vocab_size_after_short} unique words\n")

        f.write(f"\nNumber of documents after removing unwanted phrases: {num_documents_after_unwanted}\n")
        f.write(f"Vocabulary size: {vocab_size_after_unwanted} unique words\n")

        f.write(f"\nNumber of documents after removing duplicates: {num_documents_after_duplicates}\n")
        f.write(f"Vocabulary size: {vocab_size_after_duplicates} unique words\n")

        if non_english_docs:
            f.write("\nNon-English documents detected (First 10):\n")
            for i, lang, title in non_english_docs[:10]:
                f.write(f"- Document index: {i}, Detected language: {lang}, Title: {title}\n")

        f.write(f"\nTotal number of non-English documents: {num_non_english_docs}\n")

        f.write(f"\nNumber of documents after removing non-English documents: {num_documents_final}\n")
        f.write(f"Vocabulary size: {vocab_size_final} unique words\n")

        f.write("\nFinal corpus statistics:\n")
        f.write(f"Total number of documents: {num_documents_final}\n")
        f.write(f"Average document length: {avg_doc_length_final} words\n")
        f.write(f"Maximum document length: {max_doc_length_final}\n")
        f.write(f"Minimum document length: {min_doc_length_final}\n")
        f.write(f"Vocabulary size: {vocab_size_final} unique words\n")

        f.write("\nNumber of documents per company:\n")
        for number, (company, count) in enumerate(sorted_companies, start=1):
            f.write(f"{number:>3}) {company:<40} {count:>5} documents\n")
        if removed_companies:
            f.write("\nCompanies removed due to cleaning:\n")
            for company in removed_companies:
                f.write(f"- {company}\n")
    print("\n✅ Corpus statistics saved to 'corpus_statistics.txt'")