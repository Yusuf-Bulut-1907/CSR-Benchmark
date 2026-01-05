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
# BUILD DISTANCE GRAPH
# =========================

G = nx.Graph()

# Add nodes
for _, row in nodes.iterrows():
    G.add_node(row["Id"], node_type=row["Type"])

# Add edges with inverted weights (distance)
for _, row in edges.iterrows():
    if row["Weight"] > 0:
        weight = 1 / row["Weight"]
        G.add_edge(
            row["Source"],
            row["Target"],
            weight=weight
        )

print(f"Distance graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# =========================
# SHORTEST PATHS BETWEEN COMPANIES
# =========================

companies = nodes[nodes["Type"] == "company"]["Id"].tolist()

results = []

for i, c1 in enumerate(companies):
    for c2 in companies[i+1:]:
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
            continue

df_results = pd.DataFrame(results)

# =========================
# SAVE RESULTS
# =========================

output_path = os.path.join(OUTPUT_DIR, "shortest_paths_companies.csv")
df_results.sort_values("semantic_distance").to_csv(output_path, index=False)

print("✅  – Shortest paths computed.")
print(f"Results saved to {output_path}")