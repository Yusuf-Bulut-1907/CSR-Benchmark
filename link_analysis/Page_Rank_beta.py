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
# LOAD GRAPH (RAW TF-IDF)
# =========================

G, nodes = load_graph(
    NODES_PATH,
    EDGES_PATH,
    invert_weights=False
)



# =========================
# PAGERANK
# =========================

pagerank = nx.pagerank(
    G,
    weight="weight"
)

# Convert to DataFrame
df_pagerank = pd.DataFrame(
    pagerank.items(),
    columns=["node", "pagerank"]
)

# Add node type
df_results = df_pagerank.merge(
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id",
    how="left"
).drop(columns="Id")

df_results.rename(columns={"Type": "node_type"}, inplace=True)

# =========================
# SAVE RESULTS
# =========================

output_path = os.path.join(OUTPUT_DIR, "pagerank.csv")
df_results.sort_values("pagerank", ascending=False).to_csv(output_path, index=False)

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

print("✅ PageRank computed.")

print(f"Results saved to {output_path}")