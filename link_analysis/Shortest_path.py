"""
Company-to-Company Semantic Distance Module
------------------------------------------

This script computes semantic distances between companies based on the
bipartite company–concept CSR graph. The approach relies on shortest-path
analysis in a weighted graph where edge weights are interpreted as distances.

The original edges encode semantic similarity (e.g., TF-IDF-based weights).
To enable distance-based reasoning, edge weights are inverted so that:
- Strong semantic similarity → short distance
- Weak semantic similarity → long distance

Using this transformed graph, the script computes shortest paths between
all pairs of companies, capturing indirect semantic proximity mediated
by shared CSR concepts.

Main objectives:
- Quantify semantic distance between companies’ CSR discourses
- Identify companies that are thematically close despite no direct link
- Reveal intermediary concepts or firms acting as semantic bridges
- Produce a distance-based dataset for clustering or benchmarking

The resulting distances can be interpreted as a proxy for CSR discourse
similarity in a network space rather than a purely vectorial space.
"""

import pandas as pd
import networkx as nx
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

# Input files exported from the bipartite graph construction
NODES_PATH = "gephi_graph/nodes.csv"
EDGES_PATH = "gephi_graph/edges.csv"

# Output directory for results
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD NODES AND EDGES
# =========================

# Load node metadata (companies and concepts)
nodes = pd.read_csv(NODES_PATH)

# Load edges encoding company–concept semantic relationships
edges = pd.read_csv(EDGES_PATH)

# =========================
# BUILD DISTANCE GRAPH
# =========================

# Initialize an undirected graph
G = nx.Graph()

# Add nodes with their semantic type (company or concept)
for _, row in nodes.iterrows():
    G.add_node(row["Id"], node_type=row["Type"])

# Add edges with inverted weights to model semantic distance
# Higher similarity → lower distance
for _, row in edges.iterrows():
    if row["Weight"] > 0:
        weight = 1 / row["Weight"]
        G.add_edge(
            row["Source"],
            row["Target"],
            weight=weight
        )

print(
    f"Distance graph loaded: "
    f"{G.number_of_nodes()} nodes, "
    f"{G.number_of_edges()} edges."
)

# =========================
# SHORTEST PATHS BETWEEN COMPANIES
# =========================

# Extract company nodes only
companies = nodes[nodes["Type"] == "company"]["Id"].tolist()

results = []

# Compute pairwise shortest paths between companies
for i, c1 in enumerate(companies):
    for c2 in companies[i + 1:]:
        try:
            distance = nx.shortest_path_length(
                G,
                source=c1,
                target=c2,
                weight="weight"
            )
            results.append({
                "company_1": c1,
                "company_2": c2,
                "semantic_distance": round(distance, 4)
            })
        except nx.NetworkXNoPath:
            # If no semantic path exists between two companies, skip
            continue

df_results = pd.DataFrame(results)

# =========================
# SAVE RESULTS
# =========================

# Export computed semantic distances
output_path = os.path.join(
    OUTPUT_DIR,
    "shortest_paths_companies.csv"
)

df_results.sort_values(
    "semantic_distance"
).to_csv(
    output_path,
    index=False
)

print(" Shortest paths computed.")
print(f"Results saved to {output_path}")
