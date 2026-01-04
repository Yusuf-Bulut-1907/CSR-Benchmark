import pandas as pd
import numpy as np
import os

# --------------------
# Config
# --------------------
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv"
OUTPUT_DIR = "gephi_graph"
COOC_THRESHOLD = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load
# --------------------
df = pd.read_csv(TFIDF_PATH, index_col=0)
concepts = df.columns.tolist()

# --------------------
# Co-occurrence matrix
# --------------------
M = df.values
cooc = np.dot(M.T, M)
np.fill_diagonal(cooc, 0)

cooc_df = pd.DataFrame(cooc, index=concepts, columns=concepts)

# --------------------
# Build edges
# --------------------
edges = []

for i, c1 in enumerate(concepts):
    for j in range(i + 1, len(concepts)):
        c2 = concepts[j]
        weight = cooc_df.loc[c1, c2]

        if weight >= COOC_THRESHOLD:
            edges.append({
                "Source": c1,
                "Target": c2,
                "Weight": round(weight, 4),
                "Type": "Undirected"
            })

edges_df = pd.DataFrame(edges)
edges_df.to_csv(f"{OUTPUT_DIR}/edges_concept_concept.csv", index=False)

# --------------------
# Nodes
# --------------------
nodes_df = pd.DataFrame({
    "Id": concepts,
    "Label": concepts,
    "Type": "concept"
})

nodes_df.to_csv(f"{OUTPUT_DIR}/nodes_concept_concept.csv", index=False)

print("✅ Projection concept–concept prête pour Gephi")
print(f"🔗 Nombre d’arêtes : {len(edges_df)}")
print(f"Nombre de nœuds : {len(concepts)}")