import pandas as pd
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

INPUT_PATH = "results/centrality_metrics.csv"
OUTPUT_FILE = "results/centrality_analysis.xlsx"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_PATH)

# =========================
# METRICS
# =========================

metrics = [
    "degree",
    "norm_degree",
    "weight_degree",
    "betweenness",
    "closeness",
    "pagerank"
]

# =========================
# GLOBAL STATISTICS
# =========================

global_stats = df[metrics].describe().round(4)
average_values = df[metrics].mean().round(6)

avg_by_type = (
    df.groupby("node_type")[metrics]
    .mean()
    .round(6)
)

# =========================
# CORE NODES (Triple Condition)
# =========================

avg = average_values

# Un nœud est "Core" s'il est au-dessus de la moyenne sur les 3 piliers
df["core_node"] = (
    (df["weight_degree"] > avg["weight_degree"]) &
    (df["pagerank"] > avg["pagerank"]) &
    (df["closeness"] > avg["closeness"])
)

core_nodes = df[df["core_node"]].sort_values(
    "pagerank", ascending=False
)

# =========================
# SPLIT TYPES
# =========================

df_concepts = df[df["node_type"] == "concept"]
df_companies = df[df["node_type"] == "company"]

# =========================
# EXPORT EXCEL (SANS LIMITATION)
# =========================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    # 1. Onglet global
    df.to_excel(writer, sheet_name="Metrics_all_nodes", index=False)
    
    # 2. Onglets de statistiques
    global_stats.to_excel(writer, sheet_name="Global_statistics")
    average_values.to_frame("average").to_excel(
        writer, sheet_name="Average_values"
    )
    avg_by_type.to_excel(
        writer, sheet_name="Average_by_node_type"
    )
    
    # 3. Onglet des Core Nodes (les plus importants)
    core_nodes.to_excel(
        writer, sheet_name="Core_nodes", index=False
    )

    # 4. Onglets de classements complets par métrique
    for metric in metrics:

        # Classement intégral des concepts
        df_concepts.sort_values(metric, ascending=False).to_excel(
            writer,
            sheet_name=f"Rank_{metric}_concepts",
            index=False
        )

        # Classement intégral des entreprises
        df_companies.sort_values(metric, ascending=False).to_excel(
            writer,
            sheet_name=f"Rank_{metric}_companies",
            index=False
        )

print(" Analyse de centralité complète exportée.")
print(f" Fichier créé : {OUTPUT_FILE}")