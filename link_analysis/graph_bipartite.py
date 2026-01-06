"""
Bipartite Graph Construction Module
----------------------------------

This script builds a weighted bipartite graph representing the semantic
relationships between companies and CSR (Corporate Social Responsibility)
concepts extracted from textual data.

The graph is constructed from a TF-IDF matrix where:
- Rows correspond to companies
- Columns correspond to CSR-related concepts (unigrams, bigrams, trigrams)
- Cell values represent the importance of each concept in a company's discourse

Main objectives:
- Construct a bipartite company–concept graph
- Filter weak semantic links using a TF-IDF threshold
- Weight edges using a log-transformed TF-IDF score to reduce extreme values
- Export node and edge lists compatible with Gephi for link analysis and
  visualization

This module constitutes the foundational step of the network analysis pipeline.
"""

import pandas as pd
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------

# Path to the TF-IDF matrix (companies × concepts)
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv"

# Output directory for Gephi-compatible files
OUTPUT_DIR = "gephi_graph"

# Minimum TF-IDF score required to create an edge
TFIDF_THRESHOLD = 0.1

# Optional constraint on the number of concepts per company
#TOP_N_CONCEPTS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Load TF-IDF data
# -------------------------------

# Load the TF-IDF matrix
df_tfidf = pd.read_csv(TFIDF_PATH, index_col=0)

# Extract company and concept identifiers
companies = df_tfidf.index.tolist()
concepts = df_tfidf.columns.tolist()

print(
    f" TF-IDF data loaded: "
    f"{len(companies)} companies, {len(concepts)} concepts"
)

# -------------------------------
# 2. Build edges (company–concept)
# -------------------------------

edges = []

# Iterate over companies
for company in companies:

    # Retrieve TF-IDF values for the current company
    tfidf_values = df_tfidf.loc[company]

    # Select concepts exceeding the TF-IDF threshold
    selected_concepts = tfidf_values[tfidf_values >= TFIDF_THRESHOLD]

    # Skip companies with no significant concepts
    if selected_concepts.empty:
        continue

    """
    Optional alternative strategy:
    Select only the top-N most representative concepts per company
    (commented out to preserve flexibility in graph density control)
    selected_concepts = (
        selected_concepts
        .sort_values(ascending=False)
        .head(TOP_N_CONCEPTS)
    )
    """

    # Create edges between the company and each selected concept
    for concept, weight in selected_concepts.items():
        edges.append({
            "Source": company,
            "Target": concept,
            "Weight": round(np.log1p(weight), 4),
            # Explicitly defined as an undirected graph
            "Type": "Undirected"
        })

# Export edge list
edges_df = pd.DataFrame(edges)
edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges_df.to_csv(edges_path, index=False)

print(
    f" Edges CSV saved: {edges_path} "
    f"({len(edges_df)} edges)"
)

# -------------------------------
# 3. Build nodes
# -------------------------------

nodes = []

# Company nodes
for company in edges_df["Source"].unique():
    nodes.append({
        "Id": company,
        "Label": company,
        "Type": "company"
    })

# Concept nodes
for concept in edges_df["Target"].unique():

    # Identify n-gram type for descriptive purposes
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

# Export node list
nodes_df = pd.DataFrame(nodes)
nodes_path = os.path.join(OUTPUT_DIR, "nodes.csv")
nodes_df.to_csv(nodes_path, index=False)

print(
    f" Nodes CSV saved: {nodes_path} "
    f"({len(nodes_df)} nodes)"
)

print("\n Graph ready to import into Gephi!")
print(" No maximum degree constraint applied")
print(" TF-IDF threshold =", TFIDF_THRESHOLD)