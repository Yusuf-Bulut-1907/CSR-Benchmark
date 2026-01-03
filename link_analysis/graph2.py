import pandas as pd
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv"
OUTPUT_DIR = "gephi_graph"

TFIDF_THRESHOLD = 0.1
#TOP_N_CONCEPTS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Charger les données TF-IDF
# -------------------------------
df_tfidf = pd.read_csv(TFIDF_PATH, index_col=0)

companies = df_tfidf.index.tolist()
concepts = df_tfidf.columns.tolist()

print(f"✅ TF-IDF data loaded: {len(companies)} companies, {len(concepts)} concepts")

# -------------------------------
# 2. Construire les arêtes
# -------------------------------
edges = []

for company in companies:
    tfidf_values = df_tfidf.loc[company]

    selected_concepts = tfidf_values[tfidf_values >= TFIDF_THRESHOLD]

    if selected_concepts.empty:
        continue
"""
    selected_concepts = (
        selected_concepts
        .sort_values(ascending=False)
        .head(TOP_N_CONCEPTS)
    )
"""
    for concept, weight in selected_concepts.items():
        edges.append({
            "Source": company,
            "Target": concept,
            "Weight": round(np.log1p(weight), 4),
            # 🔑 CLÉ : graphe explicitement non dirigé
            "Type": "Undirected"
        })

edges_df = pd.DataFrame(edges)
edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges_df.to_csv(edges_path, index=False)

print(f"💾 Edges CSV saved: {edges_path} ({len(edges_df)} edges)")

# -------------------------------
# 3. Construire les nœuds
# -------------------------------
nodes = []

# Entreprises
for company in edges_df["Source"].unique():
    nodes.append({
        "Id": company,
        "Label": company,
        "Type": "company"
    })

# Concepts
for concept in edges_df["Target"].unique():
    if concept.count("_") == 0:
        concept_type = "unigram"
    elif concept.count("_") == 1:
        concept_type = "bigram"
    else:
        concept_type = "trigram"

    nodes.append({
        "Id": concept,
        "Label": concept,
        "Type": "concept",
        "ConceptType": concept_type
    })

nodes_df = pd.DataFrame(nodes)
nodes_path = os.path.join(OUTPUT_DIR, "nodes.csv")
nodes_df.to_csv(nodes_path, index=False)

print(f"💾 Nodes CSV saved: {nodes_path} ({len(nodes_df)} nodes)")

print("\n✅ Graph ready to import into Gephi!")