"""
Semantic Distance Analysis and Correlation Module
-------------------------------------------------

This script analyzes the semantic distances previously computed between companies
from the bipartite CSR graph and explores relationships with network centrality metrics.

Main objectives:
- Identify the closest pairs of companies in terms of CSR discourse
- Detect the most isolated companies with atypical RSE discourse
- Analyze intra- vs inter-sectoral semantic proximity
- Correlate semantic distances with network centrality metrics:
    PageRank, Betweenness, Closeness
- Visualize distribution patterns and sectoral differences
"""

import pandas as pd
import matplotlib.pyplot as plt

#==========================
# 1. LOAD SEMANTIC DISTANCES
#==========================

# Load previously computed shortest paths between companies
df_paths = pd.read_csv("results/shortest_paths_companies.csv")

# Rename columns for clarity and consistency
df_paths = df_paths.rename(columns={
    "company_1": "source",
    "company_2": "target",
    "semantic_distance": "distance"
})

# Remove self-loops (companies paired with themselves)
df_paths = df_paths[df_paths["source"] != df_paths["target"]]

#==========================
# 2. TOP 10 CLOSEST PAIRS
#==========================

# Sort pairs by ascending semantic distance and keep top 10 closest
top_pairs = (
    df_paths
    .sort_values("distance", ascending=True)
    .head(10)
)

# Display closest pairs
print("Top 10 closest company pairs:")
print(top_pairs)

#==========================
# 3. ENRICH WITH METADATA
#==========================

# Load company metadata (sector, country, etc.)
from text_mining_analytics.company_metadata import COMPANY_METADATA
df_meta = pd.DataFrame(COMPANY_METADATA)

# Merge metadata for both source and target companies
top_pairs_enriched = (
    top_pairs
    .merge(df_meta, left_on="source", right_on="company", how="left")
    .merge(
        df_meta,
        left_on="target", right_on="company",
        how="left",
        suffixes=("_src", "_tgt")
    )
)

# Display enriched top pairs with sectors and HQ countries
print(
    top_pairs_enriched[[
        "source", "target", "distance",
        "sector_src", "sector_tgt",
        "hq_country_src", "hq_country_tgt"
    ]]
)

#==========================
# 4. MOST ISOLATED COMPANIES
#==========================

# Reshape distances into long format for averaging
df_long = pd.concat([
    df_paths[["source", "distance"]].rename(columns={"source": "company"}),
    df_paths[["target", "distance"]].rename(columns={"target": "company"})
])

# Compute average semantic distance for each company
avg_distance = (
    df_long
    .groupby("company")["distance"]
    .mean()
    .sort_values(ascending=False)
)

# Top 10 most isolated companies (highest average semantic distance)
isolated_companies = avg_distance.head(10)
print("\nMost isolated companies (highest avg semantic distance):")
print(isolated_companies)

#==========================
# 5. DISTANCE DISTRIBUTION
#==========================

# Visualize the distribution of semantic distances
plt.figure()
plt.hist(df_paths["distance"], bins=30)
plt.xlabel("Semantic distance between companies")
plt.ylabel("Frequency")
plt.title("Distribution of semantic distances (Shortest Paths)")
plt.show()

#==========================
# 6. SECTORIAL ANALYSIS
#==========================

print("------Sectorial analysis------")

# Reload data and metadata for consistency
df_paths = pd.read_csv("results/shortest_paths_companies.csv")
df_meta = pd.DataFrame(COMPANY_METADATA)

# Merge sector metadata for both companies in each pair
df = (
    df_paths
    .merge(df_meta, left_on="company_1", right_on="company", how="left")
    .rename(columns={"sector": "sector_1"})
    .drop(columns="company")
    .merge(df_meta, left_on="company_2", right_on="company", how="left")
    .rename(columns={"sector": "sector_2"})
    .drop(columns="company")
)

# Classify pairs as intra-sector or inter-sector
df["pair_type"] = df.apply(
    lambda x: "Intra-sector" if x["sector_1"] == x["sector_2"] else "Inter-sector",
    axis=1
)

# Compute mean, median, and count of distances by pair type
summary = df.groupby("pair_type")["semantic_distance"].agg(
    mean="mean",
    median="median",
    count="count"
)

print(summary)

#==========================
# 7. CORRELATION WITH CENTRALITY METRICS
#==========================

# Define a helper function to compute correlation with average distance
def compute_correlation(metric_name):
    # Load centrality metric
    df_pr = pd.read_csv("results/centrality_metrics_enriched.csv")
    df_pr = df_pr[df_pr["node_type"] == "company"][["node", metric_name]]

    # Compute average semantic distance for each company
    df_long = pd.concat([
        df_paths[["company_1", "semantic_distance"]].rename(columns={"company_1": "company"}),
        df_paths[["company_2", "semantic_distance"]].rename(columns={"company_2": "company"})
    ])

    avg_dist = (
        df_long
        .groupby("company")["semantic_distance"]
        .mean()
        .reset_index()
    )

    # Merge distances with centrality metric
    df_corr = avg_dist.merge(
        df_pr, left_on="company", right_on="node", how="left"
    )

    # Compute and print correlation
    print(f"Correlation between {metric_name} and avg semantic distance:")
    print(df_corr[["semantic_distance", metric_name]].corr())
    print("\n")

# Compute correlations with PageRank, Betweenness, Closeness
print("--------- PageRank correlation ---------")
compute_correlation("pagerank")

print("--------- Betweenness correlation ---------")
compute_correlation("betweenness")

print("--------- Closeness correlation ---------")
compute_correlation("closeness")