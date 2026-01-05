"""
Corpus Cleaning and Loading Module
----------------------------------

This script loads previously scraped CSR / ESG-related web pages stored as JSON files,
cleans their textual content using the jusText library, and builds a textual corpus
ready for downstream NLP tasks (e.g., TF-IDF, topic modeling, classification).

Main objectives:
- Remove boilerplate content (navigation menus, footers, ads)
- Preserve only semantically meaningful textual paragraphs
- Associate cleaned text with minimal metadata (company, URL, title)
"""

import os
import json
import justext

#==========================
# PATH TO THE JSON FOLDER
#==========================

# Absolute path to the directory containing scraped JSON files
# Each file is expected to correspond to a single company
JSON_FOLDER_PATH = r"/Users/matteogalizia/Documents/GitHub/CSR-Benchmark/scraped_output" # Specify the path to the folder containing JSON files

#==========================
# LOADING OF THE CORPUS
#==========================

def load_corpus():
    """
    Loads the scraped JSON files, cleans their textual content using jusText,
    and builds a corpus suitable for text mining.

    Returns
    -------
    documents : list of str
        Cleaned textual documents (one document per scraped page).
    metadata : list of dict
        Metadata associated with each document (company, URL, title).
    """
    documents = []  # Stores cleaned textual content
    metadata = []   # Stores metadata aligned with each document

    #Iterate over all files in the JSON directory
    for filename in os.listdir(JSON_FOLDER_PATH):
        
        #Only process JSON files
        if filename.endswith(".json"):
            file_path = os.path.join(JSON_FOLDER_PATH, filename)

            #Open and parse the Json file
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Graceful handling of corrupted or malformed JSON files
                    print(f"Error decoding JSON in {filename}")
                    continue

                #Each JSON file contains a list of scraped pages
                for entry in data:

                    #Ensure that the page contains a list of scraped pages
                    if "text" in entry and entry["text"].strip():

                        # The 'text' field is assumed to contain raw HTML or HTML-like content
                        html = entry["text"] 

                        # jusText is applied to remove boilerplate elements
                        # (menus, headers, footers, ads, repeated navigation blocks)
                        paragraphs = justext.justext(html.encode("utf-8"), justext.get_stoplist("English")) # Extract meaningful text from HTML (removing boilerplate content like navigation menus, ads, etc.)
                        
                        # Keep only paragraphs identified as non-boilerplate
                        cleaned_text = " ".join([p.text for p in paragraphs if not p.is_boilerplate])
                        
                        # Append cleaned document to the corpus
                        documents.append(cleaned_text)

                        #Store minim metadata for the traceability ans analysis
                        metadata.append({
                            "company": entry.get("company"),
                            "url": entry.get("url"),
                            "title": entry.get("title")
                        })

    return documents, metadata


# ==========================
# SCRIPT ENTRY POINT
# ==========================

# number of documents loaded
if __name__ == "__main__":
    docs, meta = load_corpus()

     # Simple sanity check: number of documents successfully loaded
    print(f"Number of documents loaded: {len(docs)}")