import pandas as pd
import networkx as nx
import os
from Load_Graph_nondirect import load_graph

# -------------------------------
# CONFIGURATION
# -------------------------------

NODES_PATH = "gephi_graph/nodes.csv"
EDGES_PATH = "gephi_graph/edges.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD GRAPH
# =========================

# Load graph with raw TF-IDF (for degree & pagerank)
G, nodes = load_graph(
    NODES_PATH,
    EDGES_PATH,
    invert_weights=False
)

# Load graph with inverted weights (for betweenness)
G_dist, _ = load_graph(
    NODES_PATH,
    EDGES_PATH,
    invert_weights=True
)

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# =========================
# DEGREE CENTRALITY
# =========================

df_degree = pd.DataFrame(
    G.degree(),
    columns=["node", "degree"]
)

df_weighted_degree = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weighted_degree"]
)

# =========================
# BETWEENNESS CENTRALITY
# =========================

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
# PAGERANK
# =========================

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

df_results = df_degree \
    .merge(df_weighted_degree, on="node") \
    .merge(df_betweenness, on="node") \
    .merge(df_pagerank, on="node")

# Add node type
df_results = df_results.merge(
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id",
    how="left"
).drop(columns="Id")

df_results.rename(columns={"Type": "node_type"}, inplace=True)

# =========================
# SAVE RESULTS
# =========================

output_path = os.path.join(OUTPUT_DIR, "centrality_metrics.csv")
df_results.sort_values("pagerank", ascending=False).to_csv(output_path, index=False)

print("✅ Centrality metrics computed.")
print(f"Results saved to {output_path}")