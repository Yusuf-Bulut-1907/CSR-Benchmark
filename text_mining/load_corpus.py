import pandas as pd
import os
import json
import justext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Ajout expert

analyzer = SentimentIntensityAnalyzer()

#==========================
# PATH TO THE JSON FOLDER
#==========================

JSON_FOLDER_PATH = r"c:\Users\basti\OneDrive - UCL\Master 1\WEB MINING\Projet\scraped_output" # Specify the path to the folder containing JSON files

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
                        sentiment_score = analyzer.polarity_scores(cleaned_text)['compound']
                        documents.append(cleaned_text)
                        metadata.append({
                            "company": entry.get("company"),
                            "url": entry.get("url"),
                            "title": entry.get("title"),
                            "sentiment": sentiment_score
                        })

    return documents, metadata

# number of documents loaded
if __name__ == "__main__":
    docs, meta = load_corpus()
    print(f"Number of documents loaded: {len(docs)}")

    '''print("--- TEST DE FONCTIONNEMENT RAPIDE (5 premières pages) ---")
    for i in range(min(5, len(meta))):
        print(f"Entreprise: {meta[i]['company']}")
        print(f"Sentiment: {meta[i]['sentiment']}")
        print(f"Extrait texte: {docs[i][:100]}...") # Affiche les 100 premiers caractères
        print("-" * 30)'''

    df_results=pd.DataFrame(meta)
    #Calcul du sentiment moyen par entreprise (très utile pour ton Benchmark RSE)
    # Cela permet de comparer si TotalEnergies a un ton plus positif qu'Air France par exemple.
    company_sentiment = df_results.groupby('company')['sentiment'].mean().sort_values(ascending=False)
    print("\n--- Average Sentiment Score by Company ---")
    print(company_sentiment)
    if not os.path.exists("results"):
        os.makedirs("results")
        
    company_sentiment.to_csv("results/sentiment_analysis.csv")
    print("\n✅ Sentiment analysis saved to results/sentiment_analysis.csv")