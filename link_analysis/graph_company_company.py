import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# Config
# --------------------
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv"
OUTPUT_DIR = "gephi_graph"
SIM_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load TF-IDF
# --------------------
df = pd.read_csv(TFIDF_PATH, index_col=0)
companies = df.index.tolist()

# --------------------
# Cosine similarity
# --------------------
sim_matrix = cosine_similarity(df.values)
sim_df = pd.DataFrame(sim_matrix, index=companies, columns=companies)

# --------------------
# Build edges
# --------------------
edges = []

for i, c1 in enumerate(companies):
    for j in range(i + 1, len(companies)):
        c2 = companies[j]
        weight = sim_df.loc[c1, c2]

        if weight >= SIM_THRESHOLD:
            edges.append({
                "Source": c1,
                "Target": c2,
                "Weight": round(weight, 4),
                "Type": "Undirected"
            })

edges_df = pd.DataFrame(edges)
edges_df.to_csv(f"{OUTPUT_DIR}/edges_company_company.csv", index=False)

# --------------------
# Nodes
# --------------------
nodes_df = pd.DataFrame({
    "Id": companies,
    "Label": companies,
    "Type": "company"
})

nodes_df.to_csv(f"{OUTPUT_DIR}/nodes_company_company.csv", index=False)

print("✅ Projection entreprise–entreprise prête pour Gephi")
print(f"🔗 Nombre d’arêtes : {len(edges_df)}")
print(f"Nombre de nœuds : {len(companies)}")