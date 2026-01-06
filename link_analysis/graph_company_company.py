"""
Company–Company Graph Projection Module
---------------------------------------

This script constructs a company–company projection of the original
bipartite CSR graph by computing pairwise semantic similarities between
companies based on their TF-IDF representations.

Each company is represented as a high-dimensional TF-IDF vector
(concepts as features). The similarity between two companies is measured
using cosine similarity, which captures the proximity of their CSR
discourses independently of document length.

Main objectives:
- Build a homogeneous company–company graph enabling direct comparison
  of CSR strategies
- Quantify semantic similarity between companies using cosine similarity
- Filter weak similarities to control graph density and improve
  interpretability
- Export node and edge lists compatible with Gephi

This projection complements the bipartite analysis by enabling
cluster detection, similarity-based rankings, and sectoral comparisons
between firms.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# Config
# --------------------

# Path to the TF-IDF matrix (companies × concepts)
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv"

# Output directory for Gephi-compatible files
OUTPUT_DIR = "gephi_graph"

# Minimum cosine similarity required to create an edge
SIM_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load TF-IDF
# --------------------

# Load the TF-IDF matrix
df = pd.read_csv(TFIDF_PATH, index_col=0)

# Extract company identifiers
companies = df.index.tolist()

# --------------------
# Cosine similarity
# --------------------

# Compute pairwise cosine similarity between company TF-IDF vectors
sim_matrix = cosine_similarity(df.values)

# Convert similarity matrix to a DataFrame for readability
sim_df = pd.DataFrame(
    sim_matrix,
    index=companies,
    columns=companies
)

# --------------------
# Build edges
# --------------------

edges = []

# Iterate over all unique pairs of companies
for i, c1 in enumerate(companies):
    for j in range(i + 1, len(companies)):
        c2 = companies[j]
        weight = sim_df.loc[c1, c2]

        # Keep only sufficiently similar company pairs
        if weight >= SIM_THRESHOLD:
            edges.append({
                "Source": c1,
                "Target": c2,
                "Weight": round(weight, 4),
                "Type": "Undirected"
            })

# Export edge list
edges_df = pd.DataFrame(edges)
edges_df.to_csv(
    f"{OUTPUT_DIR}/edges_company_company.csv",
    index=False
)

# --------------------
# Nodes
# --------------------

# Create node list (companies only)
nodes_df = pd.DataFrame({
    "Id": companies,
    "Label": companies,
    "Type": "company"
})

nodes_df.to_csv(
    f"{OUTPUT_DIR}/nodes_company_company.csv",
    index=False
)

print(" Company–company projection ready for Gephi")
print(f" Number of edges : {len(edges_df)}")
print(f" Number of nodes : {len(companies)}")