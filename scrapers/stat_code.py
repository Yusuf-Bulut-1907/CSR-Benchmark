import json
import os
from collections import defaultdict
import math
# =========================
# PATHS
# =========================

SCRAPED_DIR = "your_scraped_data_directory"
CLEAN_FILE = "clean_corpus.json"


# =========================
# UTILS
# =========================

def count_words(text: str) -> int:
    return len(text.split())


def load_scraped_file(path):
    """
    Handle both:
    - list of pages
    - single dict page
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    return data


# =========================
# RAW SCRAPED CORPUS
# =========================

def analyze_raw_corpus():
    per_company = defaultdict(lambda: {"pages": 0, "words": 0, "chars": 0})
    total = {"pages": 0, "words": 0, "chars": 0}

    for file in os.listdir(SCRAPED_DIR):
        if not file.endswith(".json"):
            continue

        company = file.replace("_rse.json", "")
        pages = load_scraped_file(os.path.join(SCRAPED_DIR, file))

        for page in pages:
            text = page.get("text", "")
            if not text:
                continue

            words = count_words(text)
            chars = len(text)

            per_company[company]["pages"] += 1
            per_company[company]["words"] += words
            per_company[company]["chars"] += chars

            total["pages"] += 1
            total["words"] += words
            total["chars"] += chars

    return per_company, total


# =========================
# CLEAN CORPUS
# =========================

def analyze_clean_corpus():
    per_company = defaultdict(lambda: {"pages": 0, "words": 0, "chars": 0})
    total = {"pages": 0, "words": 0, "chars": 0}

    with open(CLEAN_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    for doc in corpus:
        text = doc.get("clean_text", "")
        if not text:
            continue

        company = doc["company"]
        words = count_words(text)
        chars = len(text)

        per_company[company]["pages"] += 1
        per_company[company]["words"] += words
        per_company[company]["chars"] += chars

        total["pages"] += 1
        total["words"] += words
        total["chars"] += chars

    return per_company, total


# =========================
# DISPLAY FUNCTIONS
# =========================

def print_global_comparison(raw_total, clean_total):
    print("\n📌 GLOBAL CORPUS COMPARISON")
    print("-" * 60)

    def pct(removed, base):
        return 100 * removed / base if base else 0

    removed_docs = raw_total["pages"] - clean_total["pages"]
    removed_words = raw_total["words"] - clean_total["words"]
    removed_chars = raw_total["chars"] - clean_total["chars"]

    print(f"Pages : {raw_total['pages']} → {clean_total['pages']} "
          f"(-{removed_docs}, {pct(removed_docs, raw_total['pages']):.2f}%)")

    print(f"Words       : {raw_total['words']} → {clean_total['words']} "
          f"(-{removed_words}, {pct(removed_words, raw_total['words']):.2f}%)")

    print(f"Chars : {raw_total['chars']} → {clean_total['chars']} "
          f"(-{removed_chars}, {pct(removed_chars, raw_total['chars']):.2f}%)")


def print_absolute_ratios(per_company, total, title):
    print(f"\n📊 {title}")
    print("-" * 70)

    for company, values in sorted(per_company.items()):
        print(
            f"{company:15} | "
            f"Pages: {values['pages']:5} ({100*values['pages']/total['pages']:.1f}%) | "
            f"Words: {values['words']:9} ({100*values['words']/total['words']:.1f}%) | "
            f"Chars: {values['chars']:10} ({100*values['chars']/total['chars']:.1f}%)"
        )

# ========================
# STATISTICS FUNCTIONS
# ========================

def mean_and_variance(values):
    n = len(values)
    if n == 0:
        return 0, 0, 0

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)

    return mean, variance, std_dev

def compute_stats(per_company, label):
    pages = [v["pages"] for v in per_company.values()]
    words = [v["words"] for v in per_company.values()]
    chars = [v["chars"] for v in per_company.values()]

    print(f"\n📈 STATISTICS – {label}")
    print("-" * 40)

    for name, values in {
        "Pages": pages,
        "Words": words,
        "Chars": chars
    }.items():
        mean, var, std = mean_and_variance(values)
        print(
            f"{name:<6} → "
            f"Moyenne : {mean:.2f} | "
            f"Variance : {var:.2f} | "
            f"Écart-type : {std:.2f}"
        )

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    raw_company, raw_total = analyze_raw_corpus()
    clean_company, clean_total = analyze_clean_corpus()

    print_global_comparison(raw_total, clean_total)

    print(compute_stats(raw_company,"RAW SCRAPED CORPUS"))
    print_absolute_ratios(raw_company, raw_total, "RAW SCRAPED CORPUS")
    print(compute_stats(clean_company,"CLEAN CORPUS"))
    print_absolute_ratios(clean_company, clean_total, "CLEAN CORPUS")