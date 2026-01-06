"""
Centrality Metrics Computation Module
------------------------------------

This script computes a set of classical network centrality measures on a
CSR bipartite graph (companies–concepts) constructed from TF-IDF–weighted
edges.

Two complementary graph representations are used:
- A similarity graph, where edge weights represent semantic proximity
  (TF-IDF scores), used for degree-based metrics and PageRank.
- A distance graph, where edge weights are inverted (1 / TF-IDF), used
  for path-based metrics such as betweenness and closeness centrality.

Main objectives:
- Compute degree, normalized degree and weighted degree
- Compute betweenness and closeness centrality using distance-aware paths
- Compute PageRank to capture recursive structural importance
- Consolidate all metrics into a single node-level table
- Export results for downstream statistical analysis and reporting

This module ensures methodological consistency between semantic similarity
and graph-theoretic distance interpretations.
"""

import pandas as pd
import networkx as nx
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

# Input files exported from the bipartite graph construction step
NODES_PATH = "gephi_graph/nodes.csv"
EDGES_PATH = "gephi_graph/edges.csv"

# Output directory for centrality metrics
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD NODES AND EDGES
# =========================

# Load node attributes (identifier and type: company or concept)
nodes = pd.read_csv(NODES_PATH)

# Load weighted edges (TF-IDF scores)
edges = pd.read_csv(EDGES_PATH)

# =========================
# GRAPH 1: Similarity graph
# (Degree & PageRank)
# =========================

# Undirected graph where edge weights represent semantic similarity
G = nx.Graph()

# Add nodes with their respective types
for _, row in nodes.iterrows():
    G.add_node(row["Id"], node_type=row["Type"])

# Add weighted edges (TF-IDF similarity)
for _, row in edges.iterrows():
    G.add_edge(
        row["Source"],
        row["Target"],
        weight=row["Weight"]
    )

print(
    f"Similarity graph loaded: "
    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
)

# =========================
# GRAPH 2: Distance graph
# (Betweenness & Closeness)
# =========================

# Graph dedicated to shortest-path computations
# Edge weights are inverted to represent semantic distance
G_dist = nx.Graph()

# Add nodes with the same attributes
for _, row in nodes.iterrows():
    G_dist.add_node(row["Id"], node_type=row["Type"])

# Add edges with inverted weights
for _, row in edges.iterrows():
    weight = 1 / row["Weight"] if row["Weight"] > 0 else 0
    G_dist.add_edge(
        row["Source"],
        row["Target"],
        weight=weight
    )

# =========================
# DEGREE CENTRALITY
# =========================

# Raw degree (number of connections)
df_degree = pd.DataFrame(
    G.degree(),
    columns=["node", "degree"]
)

# Normalized degree centrality
norm_degree_dict = nx.degree_centrality(G)
df_norm_degree = pd.DataFrame(
    norm_degree_dict.items(),
    columns=["node", "norm_degree"]
)

# Weighted degree (sum of TF-IDF weights)
df_weight_degree = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weight_degree"]
)

# =========================
# BETWEENNESS CENTRALITY
# =========================

# Betweenness centrality computed on the distance graph
# This captures the intermediary role of nodes on shortest semantic paths
betweenness = nx.betweenness_centrality(
    G_dist,
    weight="weight",
    normalized=True
)

df_betweenness = pd.DataFrame(
    betweenness.items(),
    columns=["node", "betweenness"]
)

# =========================
# CLOSENESS CENTRALITY
# =========================

# Closeness centrality based on weighted shortest paths
df_closeness = pd.DataFrame(
    nx.closeness_centrality(
        G_dist,
        distance="weight"
    ).items(),
    columns=["node", "closeness"]
)

# =========================
# PAGERANK
# =========================

# PageRank computed on the similarity graph
# Edge weights influence the probability of random walks
pagerank = nx.pagerank(
    G,
    weight="weight"
)

df_pagerank = pd.DataFrame(
    pagerank.items(),
    columns=["node", "pagerank"]
)

# =========================
# MERGE ALL METRICS
# =========================

# Merge all centrality metrics into a single DataFrame
df_results = (
    df_degree
    .merge(df_norm_degree, on="node")
    .merge(df_weight_degree, on="node")
    .merge(df_betweenness, on="node")
    .merge(df_closeness, on="node")
    .merge(df_pagerank, on="node")
)

# Add node type (company / concept)
df_results = df_results.merge(
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id",
    how="left"
).drop(columns="Id")

df_results.rename(
    columns={"Type": "node_type"},
    inplace=True
)

# =========================
# SAVE RESULTS
# =========================

# Export centrality metrics sorted by PageRank
output_path = os.path.join(
    OUTPUT_DIR,
    "centrality_metrics.csv"
)

df_results.sort_values(
    by="pagerank",
    ascending=False
).to_csv(output_path, index=False)

print(" Centrality metrics computed.")
print(f" Results saved to {output_path}")