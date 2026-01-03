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
# LOAD GRAPH (INVERT WEIGHTS)
# =========================

G, nodes = load_graph(
    NODES_PATH,
    EDGES_PATH,
    invert_weights=True
)

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# =========================
# BETWEENNESS CENTRALITY
# =========================

betweenness = nx.betweenness_centrality(
    G,
    weight="weight",
    normalized=True
)

# Convert to DataFrame
df_betweenness = pd.DataFrame(
    betweenness.items(),
    columns=["node", "betweenness"]
)

# Add node type
df_results = df_betweenness.merge(
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id",
    how="left"
).drop(columns="Id")

df_results.rename(columns={"Type": "node_type"}, inplace=True)

# =========================
# SAVE RESULTS
# =========================

output_path = os.path.join(OUTPUT_DIR, "betweenness_centrality.csv")
df_results.sort_values("betweenness", ascending=False).to_csv(output_path, index=False)

print("✅ – Betweenness centrality computed.")
print(f"Results saved to {output_path}")