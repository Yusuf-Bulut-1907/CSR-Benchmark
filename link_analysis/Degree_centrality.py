from Load_Graph_nondirect import load_graph
import pandas as pd
import os

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

G, nodes = load_graph(
    "gephi_graph/nodes.csv",
    "gephi_graph/edges.csv",
    invert_weights=False  # pour shortest path, si true, weight = 1 / TF-IDF
)

df_degree = pd.DataFrame(G.degree(), columns=["node", "degree"])
df_weighted = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weighted_degree"]
)

df_results = df_degree.merge(df_weighted, on="node")
df_results = df_results.merge(
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id"
).drop(columns="Id")

df_results.rename(columns={"Type": "node_type"}, inplace=True)



output_path = os.path.join(OUTPUT_DIR, "degree_centrality.csv")
df_results.to_csv(output_path, index=False)

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

print("✅ Degree centrality computed.")

print(f"Degree centrality saved to {output_path}")