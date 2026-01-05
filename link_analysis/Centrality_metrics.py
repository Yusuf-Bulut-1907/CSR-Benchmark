import pandas as pd
import networkx as nx
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

NODES_PATH = "gephi_graph/nodes.csv"
EDGES_PATH = "gephi_graph/edges.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD NODES AND EDGES
# =========================

nodes = pd.read_csv(NODES_PATH)
edges = pd.read_csv(EDGES_PATH)

# =========================
# GRAPH 1: Similarity graph
# (Degree & PageRank)
# =========================

G = nx.Graph()

# Add nodes
for _, row in nodes.iterrows():
    G.add_node(row["Id"], node_type=row["Type"])

# Add edges (TF-IDF similarity)
for _, row in edges.iterrows():
    G.add_edge(
        row["Source"],
        row["Target"],
        weight=row["Weight"]
    )

print(f"Similarity graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# =========================
# GRAPH 2: Distance graph
# (Betweenness)
# =========================

G_dist = nx.Graph()

# Add nodes
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

df_degree = pd.DataFrame(
    G.degree(),
    columns=["node", "degree"]
)

norm_degree_dict = nx.degree_centrality(G)
df_norm_degree = pd.DataFrame(
    norm_degree_dict.items(),
    columns=["node", "norm_degree"]
)

df_weight_degree = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weight_degree"]
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
# CLOSENESS CENTRALITY
# =========================

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
    .merge(df_norm_degree, on="node") \
    .merge(df_weight_degree, on="node") \
    .merge(df_betweenness, on="node") \
    .merge(df_closeness, on="node") \
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
df_results.sort_values( by="pagerank", ascending=False ).to_csv(output_path, index=False)

print("✅ Centrality metrics computed.")
print(f"Results saved to {output_path}")