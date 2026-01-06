"""
Centrality Metrics Analysis and Reporting Module
------------------------------------------------

This script post-processes centrality metrics computed in Gephi and exported
as a CSV file. It performs a structured analytical summary of node-level
centrality indicators for a bipartite CSR network (companies–concepts).

Main objectives:
- Load and consolidate multiple centrality measures (degree, betweenness,
  closeness, PageRank, etc.)
- Produce descriptive statistics at the global level and by node type
  (company vs. concept)
- Identify structurally important "core nodes" based on a multi-criteria rule
- Generate exhaustive rankings of companies and concepts for each metric
- Export all results into a single multi-sheet Excel file for interpretation
  and reporting

This module is designed as a final analytical step bridging network computation
(Gephi) and qualitative interpretation (report writing).
"""

import pandas as pd
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

# Path to the CSV file containing centrality metrics exported from Gephi
INPUT_PATH = "results/centrality_metrics.csv"

# Output Excel file gathering all analytical results
OUTPUT_FILE = "results/centrality_analysis.xlsx"

# =========================
# LOAD DATA
# =========================

# Load node-level centrality metrics
# Each row corresponds to a node (company or concept)
df = pd.read_csv(INPUT_PATH)

# =========================
# METRICS
# =========================

# List of centrality measures considered in the analysis
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

# Descriptive statistics (count, mean, std, quartiles) for each metric
global_stats = df[metrics].describe().round(4)

# Average value of each centrality metric across all nodes
average_values = df[metrics].mean().round(6)

# Average centrality values computed separately for companies and concepts
avg_by_type = (
    df.groupby("node_type")[metrics]
    .mean()
    .round(6)
)

# =========================
# CORE NODES (Triple Condition)
# =========================

# Reference averages used as thresholds
avg = average_values

# A node is classified as "core" if it exceeds the global average on
# three complementary structural dimensions:
# - Weighted degree: intensity of semantic engagement
# - PageRank: structural importance within the network
# - Closeness: proximity to the rest of the network
df["core_node"] = (
    (df["weight_degree"] > avg["weight_degree"]) &
    (df["pagerank"] > avg["pagerank"]) &
    (df["closeness"] > avg["closeness"])
)

# Subset of core nodes, ranked by PageRank (descending)
core_nodes = df[df["core_node"]].sort_values(
    "pagerank", ascending=False
)

# =========================
# SPLIT TYPES
# =========================

# Separate datasets for concepts and companies
# This allows metric-specific rankings by node category
df_concepts = df[df["node_type"] == "concept"]
df_companies = df[df["node_type"] == "company"]

# =========================
# EXPORT EXCEL (SANS LIMITATION)
# =========================

# Export all analytical results into a multi-sheet Excel file
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    # 1. Complete dataset with all metrics
    df.to_excel(writer, sheet_name="Metrics_all_nodes", index=False)
    
    # 2. Global descriptive statistics
    global_stats.to_excel(writer, sheet_name="Global_statistics")
    
    average_values.to_frame("average").to_excel(
        writer, sheet_name="Average_values"
    )
    
    avg_by_type.to_excel(
        writer, sheet_name="Average_by_node_type"
    )
    
    # 3. Core nodes (structurally central actors)
    core_nodes.to_excel(
        writer, sheet_name="Core_nodes", index=False
    )

    # 4. Full rankings by metric and node type
    for metric in metrics:

        # Ranking of concepts for the given metric
        df_concepts.sort_values(metric, ascending=False).to_excel(
            writer,
            sheet_name=f"Rank_{metric}_concepts",
            index=False
        )

        # Ranking of companies for the given metric
        df_companies.sort_values(metric, ascending=False).to_excel(
            writer,
            sheet_name=f"Rank_{metric}_companies",
            index=False
        )

print(" Centrality analysis completed.")
print(f" File created : {OUTPUT_FILE}")