import pandas as pd
import networkx as nx
import os
from networkx.algorithms.community import louvain_communities

# ============================
# CONFIGURATION
# ============================
COSINE_PATH = "results/cosine_similarity_tf_idf.csv"
OUTPUT_DIR = "results/link_analysis"
SIM_THRESHOLD = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# LOAD COSINE SIMILARITY
# ============================
cosine_df = pd.read_csv(COSINE_PATH, index_col=0)
companies = cosine_df.index.tolist()

# ============================
# BUILD GRAPH
# ============================
G = nx.Graph(name="CSR Semantic Similarity Graph")

for company in companies:
    G.add_node(company, label=company, type="company")

for i, c1 in enumerate(companies):
    for j, c2 in enumerate(companies):
        if j > i:
            sim = cosine_df.iloc[i, j]
            if sim >= SIM_THRESHOLD:
                G.add_edge(c1, c2, weight=round(float(sim), 4))

# ============================
# NETWORK METRICS
# ============================
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, weight="weight")
pagerank = nx.pagerank(G, weight="weight")

# Shortest path (average)
avg_shortest_path = (
    nx.average_shortest_path_length(G, weight="weight")
    if nx.is_connected(G)
    else None
)

# ============================
# COMMUNITY DETECTION (LOUVAIN)
# ============================
communities = louvain_communities(G, weight="weight", seed=42)
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

# ============================
# EXPORT NODES CSV
# ============================
nodes_df = pd.DataFrame({
    "Id": companies,
    "Label": companies,
    "DegreeCentrality": [degree.get(c, 0) for c in companies],
    "BetweennessCentrality": [betweenness.get(c, 0) for c in companies],
    "PageRank": [pagerank.get(c, 0) for c in companies],
    "Community": [community_map.get(c, -1) for c in companies]
})

nodes_df.to_csv(f"{OUTPUT_DIR}/nodes.csv", index=False)

# ============================
# EXPORT EDGES CSV
# ============================
edges_df = pd.DataFrame([
    {"Source": u, "Target": v, "Weight": d["weight"]}
    for u, v, d in G.edges(data=True)
])

edges_df.to_csv(f"{OUTPUT_DIR}/edges.csv", index=False)

# ============================
# EXPORT GEPHI FILE
# ============================
nx.write_gexf(G, f"{OUTPUT_DIR}/graph_similarity.gexf")

# ============================
# ANALYSIS REPORT
# ============================
with open(f"{OUTPUT_DIR}/analyse_link.txt", "w", encoding="utf-8") as f:
    f.write("=====================================\n")
    f.write("CSR LINK ANALYSIS REPORT\n")
    f.write("=====================================\n\n")

    f.write(f"Number of nodes: {G.number_of_nodes()}\n")
    f.write(f"Number of edges: {G.number_of_edges()}\n")
    f.write(f"Graph density: {nx.density(G):.4f}\n")
    f.write(f"Connected components: {nx.number_connected_components(G)}\n\n")

    if avg_shortest_path:
        f.write(f"Average shortest path length: {avg_shortest_path:.4f}\n\n")

    f.write("Top 5 Degree Centrality:\n")
    for n, v in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        f.write(f"- {n}: {v:.4f}\n")

    f.write("\nTop 5 Betweenness Centrality:\n")
    for n, v in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        f.write(f"- {n}: {v:.4f}\n")

    f.write("\nTop 5 PageRank:\n")
    for n, v in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]:
        f.write(f"- {n}: {v:.4f}\n")

    f.write(f"\nDetected communities (Louvain): {len(communities)}\n")

print("✅ Link analysis completed successfully")