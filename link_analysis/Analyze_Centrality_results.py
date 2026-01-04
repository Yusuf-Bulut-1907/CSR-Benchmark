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
# PREPARE ANALYSIS
# =========================

metrics = ["degree", "weighted_degree", "betweenness", "pagerank"]
TOP_N = 15

# Global statistics
global_stats = df[metrics].describe().round(4)

# Average values
average_values = df[metrics].mean().round(6)

# Average by node type
avg_by_type = (
    df.groupby("node_type")[metrics]
    .mean()
    .round(6)
)

# Core nodes (above average on weighted degree & pagerank)
avg = average_values
df["core_node"] = (
    (df["weighted_degree"] > avg["weighted_degree"]) &
    (df["pagerank"] > avg["pagerank"])
)

core_nodes = df[df["core_node"]].sort_values("pagerank", ascending=False)

# =========================
# SEPARATE CONCEPTS / COMPANIES
# =========================

df_concepts = df[df["node_type"] == "concept"]
df_companies = df[df["node_type"] == "company"]

# =========================
# WRITE SINGLE EXCEL FILE
# =========================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    # All nodes
    df.to_excel(writer, sheet_name="Metrics_all_nodes", index=False)

    # Global stats
    global_stats.to_excel(writer, sheet_name="Global_statistics")

    # Averages
    average_values.to_frame("average").to_excel(
        writer, sheet_name="Average_values"
    )

    avg_by_type.to_excel(
        writer, sheet_name="Average_by_node_type"
    )

    # Core nodes
    core_nodes.to_excel(
        writer, sheet_name="Core_nodes", index=False
    )

    # TOP nodes by metric, separated by type
    for metric in metrics:

        # Top concepts
        top_concepts = (
            df_concepts
            .sort_values(metric, ascending=False)
            .head(TOP_N)
        )

        top_concepts.to_excel(
            writer,
            sheet_name=f"Top_{metric}_concepts",
            index=False
        )

        # Top companies
        top_companies = (
            df_companies
            .sort_values(metric, ascending=False)
            .head(TOP_N)
        )

        top_companies.to_excel(
            writer,
            sheet_name=f"Top_{metric}_companies",
            index=False
        )

print("✅ Centrality analysis exported to a single Excel file.")
print(f"📁 File created: {OUTPUT_FILE}")