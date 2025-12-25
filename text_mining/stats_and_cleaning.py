import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, DetectorFactory

from load_corpus import load_corpus


documents, metadata = load_corpus()

#============================
# Statistics and cleaning
#============================

companies_before = set(meta.get("company") for meta in metadata if meta.get("company")) # Store companies before cleaning


# Number of documents
num_documents_initial = len(documents)

# Average document length
doc_lengths = [len(doc.split()) for doc in documents]
avg_doc_length_initial = np.mean(doc_lengths).round(2)

# Maximum and minimum document length
max_doc_length_initial = np.max(doc_lengths)
min_doc_length_initial = np.min(doc_lengths)
max_doc_title_initial = metadata[np.argmax(doc_lengths)]['title']
min_doc_title_initial = metadata[np.argmin(doc_lengths)]['title']

# Vocabulary size
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_initial = len(vectorizer.get_feature_names_out())


# Remove short documents (less than 50 words)
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


# Delete documents witch are not well scraped (eg. "Loading...", "Error 404", etc.)
unwanted_words = ["loading", "error", "not found", "unavailable", "access denied"]

clean_documents = []
clean_metadata = []

for doc, meta in zip(documents, metadata):
    title = (meta.get("title") or "").lower()
    if not any(word in title for word in unwanted_words):
        clean_documents.append(doc)
        clean_metadata.append(meta)

documents = clean_documents
metadata = clean_metadata

num_documents_after_unwanted = len(documents)

# Vocabulary size after removing unwanted phrases
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab_size_after_unwanted = len(vectorizer.get_feature_names_out())


#Elimination of redundant documents (duplicates)
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


# Detect if ther are non-English documents (using langdetect)
DetectorFactory.seed = 0  # For consistent results
non_english_docs = []

for i, doc in enumerate(documents):
    try:
        lang = detect(doc)
        if lang != 'en':
            non_english_docs.append((i, lang, metadata[i].get("title")))
    except:
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


# Final corpus statistics
final_doc_lengths = [len(doc.split()) for doc in documents]
avg_doc_length_final = np.mean(final_doc_lengths).round(2)
max_doc_length_final = np.max(final_doc_lengths)
min_doc_length_final = np.min(final_doc_lengths)


# List of companies and number of documents per company
company_counts = Counter(meta.get("company") for meta in metadata if meta.get("company"))
sorted_companies = company_counts.most_common()  # liste de tuples (company, count)

# List the companies that have been removed due to cleaning
companies_after = set(meta.get("company") for meta in metadata if meta.get("company"))
removed_companies = companies_before - companies_after


# Return cleaned documents and metadata
def get_cleaned_corpus():
    return documents, metadata


if __name__ == "__main__":

    print(f"Number of documents: {num_documents_initial}")

    print(f"Average document length: {avg_doc_length_initial}", "words")

    print(
        f"Maximum document length: {max_doc_length_initial}",
        "(" + max_doc_title_initial + ")"
    )
    print(
        f"Minimum document length: {min_doc_length_initial}",
        "(" + min_doc_title_initial + ")"
    )

    print(f"Vocabulary size: {vocab_size_initial}", "unique words")

    print(f"\nNumber of documents after removing short documents: {num_documents_after_short}")
    print(f"Vocabulary size: {vocab_size_after_short}", "unique words")

    print(f"\nNumber of documents after removing unwanted phrases: {num_documents_after_unwanted}")
    print(f"Vocabulary size: {vocab_size_after_unwanted}", "unique words")

    print(f"\nNumber of documents after removing duplicates: {num_documents_after_duplicates}")
    print(f"Vocabulary size: {vocab_size_after_duplicates}", "unique words")

    if non_english_docs:
        print("\nNon-English documents detected (first 10):")
        for i, lang, title in non_english_docs[:10]:
            print(f"- Document index: {i}, Detected language: {lang}, Title: {title}")

    print(f"\nTotal number of non-English documents: {num_non_english_docs}")

    print(f"\nNumber of documents after removing non-English documents: {num_documents_final}")
    print(f"Vocabulary size: {vocab_size_final}", "unique words")

    print("\nFinal corpus statistics:")
    print(f"Total number of documents: {num_documents_final}")
    print(f"Average document length: {avg_doc_length_final}", "words")
    print(f"Maximum document length: {max_doc_length_final}")
    print(f"Minimum document length: {min_doc_length_final}")
    print(f"Vocabulary size: {vocab_size_final}", "unique words")

    print("\nNumber of documents per company:")
    for number, (company, count) in enumerate(sorted_companies, start=1):
        print(f"{number:>3}) {company:<40} {count:>5} documents")

    if removed_companies:
        print("\nCompanies removed due to cleaning:")
        for company in removed_companies:
            print(f"- {company}")

    # Download txt file with all the prindted statistics
    with open("corpus_statistics.txt", "w", encoding="utf-8") as f:
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
            f.write("\nNon-English documents detected (first 10):\n")
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





