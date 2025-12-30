import os
import json
import justext

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
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {filename}")
                    continue
                for entry in data:
                    if "text" in entry and entry["text"].strip():
                        html = entry["text"] 
                        paragraphs = justext.justext(html.encode("utf-8"), justext.get_stoplist("English")) # Extract meaningful text from HTML (removing boilerplate content like navigation menus, ads, etc.)
                        cleaned_text = " ".join([p.text for p in paragraphs if not p.is_boilerplate])
                        documents.append(cleaned_text)
                        metadata.append({
                            "company": entry.get("company"),
                            "url": entry.get("url"),
                            "title": entry.get("title")
                        })

    return documents, metadata

# number of documents loaded
if __name__ == "__main__":
    docs, meta = load_corpus()
    print(f"Number of documents loaded: {len(docs)}")