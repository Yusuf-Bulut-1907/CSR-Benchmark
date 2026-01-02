import pandas as pd
import numpy as np
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv" # Path to csv file with bigrams and trigrams
OUTPUT_DIR = "gephi_graph"

TFIDF_THRESHOLD = 0.1  # Minimum TF-IDF score to consider a term relevant"
TOP_N_CONCEPTS = 10  # Number of top concepts to extract per company

os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

# =====================
# LOAD DATA
# =====================

df_tfidf = pd.read_csv(TFIDF_PATH, index_col=0)
companies = df_tfidf.index.tolist()
concepts = df_tfidf.columns.tolist()

print(f"Loaded TF-IDF data with {len(companies)} companies and {len(concepts)} concepts.")

# =====================
# BUILD EDGE LIST
# =====================

edges = []

for company in companies:
    tfidf_scores = df_tfidf.loc[company]
    # Filter concepts by TF-IDF threshold
    relevant_concepts = tfidf_scores[tfidf_scores >= TFIDF_THRESHOLD]
    # Get top N concepts
    if len(relevant_concepts) > TOP_N_CONCEPTS:
        relevant_concepts = relevant_concepts.sort_values(ascending=False).head(TOP_N_CONCEPTS)
    for concept, weight in relevant_concepts.items():
        edges.append({
            "Source": company,
            "Target": concept,
            "Weight": round(weight, 4)
        })

df_edges = pd.DataFrame(edges)
edge_file_path = os.path.join(OUTPUT_DIR, "edges.csv")
df_edges.to_csv(edge_file_path, index=False)

print(f"Edge list with {len(df_edges)} edges saved to {edge_file_path}")

# =====================
# BUILD NODE LIST
# =====================

nodes = []

# Company nodes
for company in companies:
    nodes.append({
        "Id": company,
        "Label": company,
        "Type": "company" 
    })

# Concept nodes (unique concepts from edges)
unique_concepts = df_edges["Target"].unique()
for concept in unique_concepts:
    nodes.append({
        "Id": concept,
        "Label": concept,
        "Type": "concept" 
    })

df_nodes = pd.DataFrame(nodes)
node_file_path = os.path.join(OUTPUT_DIR, "nodes.csv")
df_nodes.to_csv(node_file_path, index=False)
print(f"Node list with {len(df_nodes)} nodes saved to {node_file_path}")

# =====================
# CHECK
# =====================

print ("\n Graph Summary:")
print(f" - Total Companies: {len(companies)}")
print(f" - Total Concepts (connected): {len(unique_concepts)}")
print(f" - Total Edges: {len(df_edges)}")
print(f" - Graph ready for Gephi in directory: {OUTPUT_DIR}")