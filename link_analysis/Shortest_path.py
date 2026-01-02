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

print("✅ Step 6 – Shortest paths computed.")
print(f"Results saved to {output_path}")