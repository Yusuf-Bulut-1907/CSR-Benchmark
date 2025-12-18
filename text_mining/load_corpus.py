from importlib import metadata
import os
import json

#==========================
# PATH TO THE JSON FOLDER
#==========================

JSON_FOLDER_PATH = r"C:\Users\yusuf\OneDrive\Documents\Projet RSE\scraped_output" # Please specify the path to the folder containing JSON files

#==========================
# LOADING OF THE CORPUS
#==========================

"""
documents = []
metadata = []

for filename in os.listdir(JSON_FOLDER_PATH):
    if filename.endswith(".json"):
        file_path = os.path.join(JSON_FOLDER_PATH, filename)
        print(f"Loading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Number of entries in {filename}: {len(data)}")
            for entry in data:
                if "text" in entry and entry["text"].strip() != "":
                    documents.append(entry["text"])

                    metadata.append({
                        "company": entry.get("company"),
                        "url": entry.get("url"),
                        "title": entry.get("title")
                    })
                    print(f"Loaded document from {entry.get('url')}")
                    print(f"✅ Finished processing {filename}.")

print(f"Total number of documents: {len(documents)}")

#Visualize one document and its metadata
print("Sample document text:", documents[0][:500])  # Print first 500 characters of the first document
print("Sample document metadata:", metadata[0])  # Print metadata of the first document 

def load_corpus():
    return documents, metadata

"""

def load_corpus():
    documents = []
    metadata = []

    for filename in os.listdir(JSON_FOLDER_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(JSON_FOLDER_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    if "text" in entry and entry["text"].strip():
                        documents.append(entry["text"])
                        metadata.append({
                            "company": entry.get("company"),
                            "url": entry.get("url"),
                            "title": entry.get("title")
                        })

    return documents, metadata

# try if it works
if __name__ == "__main__":
    docs, meta = load_corpus()
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("First document text:", docs[0][:500])  # Print first 500 characters
        print("First document metadata:", meta[0])  # Print metadata of the first document
