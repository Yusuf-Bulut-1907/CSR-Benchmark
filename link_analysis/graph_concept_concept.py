"""
Concept–Concept Graph Projection Module
--------------------------------------

This script constructs a concept–concept projection of the original
company–concept bipartite CSR graph. Two concepts are connected when they
co-occur across company documents, with edge weights reflecting the
strength of their joint presence.

The projection is built from a TF-IDF matrix where:
- Rows correspond to companies
- Columns correspond to CSR-related concepts
- Values capture the importance of each concept within company discourse

Concept co-occurrence is computed efficiently using sparse matrix
multiplication, making the approach scalable to large vocabularies.

Main objectives:
- Analyze the thematic structure of CSR discourse
- Identify clusters of closely related CSR concepts
- Detect "bridge concepts" connecting multiple thematic areas
- Export a Gephi-compatible graph for visualization and community detection

This projection complements the bipartite analysis by focusing on the
internal organization of CSR themes.
"""

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

# Config
TFIDF_PATH = "data/TFIDF_RSE_unigram_bigram.csv"
OUTPUT_DIR = "gephi_graph"

# Minimum co-occurrence weight required to create an edge
# Increased slightly to avoid excessive graph density in Gephi
COOC_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Efficient data loading
# -------------------------------

print("Loading data...")

# Load the TF-IDF matrix (companies × concepts)
df = pd.read_csv(TFIDF_PATH, index_col=0)

# Extract concept identifiers
concepts = df.columns.tolist()

# -------------------------------
# 2. Conversion to sparse matrix
# -------------------------------

# Convert the dense TF-IDF matrix to a sparse representation
# This significantly reduces memory usage
M = csr_matrix(df.values)

# -------------------------------
# 3. Sparse co-occurrence computation
# -------------------------------

print("Co-occurence matrix computation (Sparse)...")

# Compute concept–concept co-occurrence via sparse matrix multiplication
cooc = (M.T @ M)

# Remove self-loops (concept–concept)
cooc.setdiag(0)

# Eliminate zero entries to further reduce memory footprint
cooc.eliminate_zeros()

# -------------------------------
# 4. Edge extraction
# -------------------------------

print("Extracting edges...")

# Convert to COO format for efficient iteration over non-zero values
cooc_coo = cooc.tocoo()

edges = []

# Iterate over co-occurring concept pairs
for i, j, v in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):

    # Keep each undirected edge only once (avoid A–B and B–A duplicates)
    # Apply a co-occurrence threshold to control graph density
    if i < j and v >= COOC_THRESHOLD:
        edges.append({
            "Source": concepts[i],
            "Target": concepts[j],
            "Weight": round(float(v), 4),
            "Type": "Undirected"
        })

# -------------------------------
# 5. Export results
# -------------------------------

print(f"Creation of the csv with {len(edges)} edges...")

# Export edge list
edges_df = pd.DataFrame(edges)
edges_df.to_csv(
    f"{OUTPUT_DIR}/edges_concept_concept.csv",
    index=False
)

# Export node list (concepts only)
nodes_df = pd.DataFrame({
    "Id": concepts,
    "Label": concepts,
    "Type": "concept"
})
nodes_df.to_csv(
    f"{OUTPUT_DIR}/nodes_concept_concept.csv",
    index=False
)

print(" Done !")
