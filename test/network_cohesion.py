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
# LOAD GRAPH (DISTANCES)
# =========================

G, nodes = load_graph(
    NODES_PATH,
    EDGES_PATH,
    invert_weights=True
)

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# =========================
# NETWORK DENSITY
# =========================

density = nx.density(G)

# =========================
# DIAMETER (LARGEST COMPONENT)
# =========================

# Extract largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_cc = G.subgraph(largest_cc)

# Compute weighted diameter
diameter = nx.diameter(G_cc, weight="weight")

# =========================
# INTERPRETATION
# =========================

if density > 0.05:
    density_interpretation = "High density: standardized RSE discourse"
else:
    density_interpretation = "Low density: diversified RSE discourse"

if diameter > 5:
    diameter_interpretation = "Large diameter: weak integration between themes"
else:
    diameter_interpretation = "Small diameter: integrated RSE strategy"

# =========================
# SAVE RESULTS
# =========================

df_results = pd.DataFrame([
    {"metric": "density", "value": round(density, 4), "interpretation": density_interpretation},
    {"metric": "diameter", "value": round(diameter, 4), "interpretation": diameter_interpretation}
])

output_path = os.path.join(OUTPUT_DIR, "network_cohesion.csv")
df_results.to_csv(output_path, index=False)

print("✅ Network cohesion metrics computed.")
print(f"Results saved to {output_path}")