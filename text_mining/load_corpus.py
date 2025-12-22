from importlib import metadata
import os
import json

#==========================
# PATH TO THE JSON FOLDER
#==========================

JSON_FOLDER_PATH = r"" # Specify the path to the folder containing JSON files

#==========================
# LOADING OF THE CORPUS
#==========================

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