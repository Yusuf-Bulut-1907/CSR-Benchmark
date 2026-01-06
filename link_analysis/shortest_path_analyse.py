import pandas as pd
import matplotlib.pyplot as plt

# --- 1. LOAD SHORTEST PATHS ---
df_paths = pd.read_csv("results/shortest_paths_companies.csv")

# Rename columns to semantic names (robust & readable)
df_paths = df_paths.rename(columns={
    "company_1": "source",
    "company_2": "target",
    "semantic_distance": "distance"
})

# Safety check (remove self-loops)
df_paths = df_paths[df_paths["source"] != df_paths["target"]]

# --- 2. TOP 10 CLOSEST PAIRS ---
top_pairs = (
    df_paths
    .sort_values("distance", ascending=True)
    .head(10)
)

print("Top 10 closest company pairs:")
print(top_pairs)

# --- 3. ENRICH WITH METADATA ---
from company_metadata import COMPANY_METADATA

df_meta = pd.DataFrame(COMPANY_METADATA)

top_pairs_enriched = (
    top_pairs
    .merge(df_meta, left_on="source", right_on="company", how="left")
    .merge(
        df_meta,
        left_on="target",
        right_on="company",
        how="left",
        suffixes=("_src", "_tgt")
    )
)

print(
    top_pairs_enriched[[
        "source", "target", "distance",
        "sector_src", "sector_tgt",
        "hq_country_src", "hq_country_tgt"
    ]]
)

# --- 4. MOST ISOLATED COMPANIES ---
df_long = pd.concat([
    df_paths[["source", "distance"]].rename(columns={"source": "company"}),
    df_paths[["target", "distance"]].rename(columns={"target": "company"})
])

avg_distance = (
    df_long
    .groupby("company")["distance"]
    .mean()
    .sort_values(ascending=False)
)

isolated_companies = avg_distance.head(10)
print("\nMost isolated companies (highest avg semantic distance):")
print(isolated_companies)

# --- 5. DISTANCE DISTRIBUTION ---
plt.figure()
plt.hist(df_paths["distance"], bins=30)
plt.xlabel("Semantic distance between companies")
plt.ylabel("Frequency")
plt.title("Distribution of semantic distances (Shortest Paths)")
plt.show()

print("------Sectorial analysis------")
# Load data
df_paths = pd.read_csv("results/shortest_paths_companies.csv")
df_meta = pd.DataFrame(COMPANY_METADATA)
# Merge metadata
df = (
    df_paths
    .merge(df_meta, left_on="company_1", right_on="company", how="left")
    .rename(columns={"sector": "sector_1"})
    .drop(columns="company")
    .merge(df_meta, left_on="company_2", right_on="company", how="left")
    .rename(columns={"sector": "sector_2"})
    .drop(columns="company")
)

# Intra vs Inter
df["pair_type"] = df.apply(
    lambda x: "Intra-sector" if x["sector_1"] == x["sector_2"] else "Inter-sector",
    axis=1
)

# Compare means
summary = df.groupby("pair_type")["semantic_distance"].agg(
    mean="mean",
    median="median",
    count="count"
)

print(summary)


print("--------- pagerank and avg distance correlation ---------")
# Load PageRank
df_pr = pd.read_csv("results/centrality_metrics_enriched.csv")  # ou ton fichier Gephi exporté
df_pr = df_pr[df_pr["node_type"] == "company"][["node", "pagerank"]]

# Avg distance
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

# Merge
df_corr = avg_dist.merge(
    df_pr, left_on="company", right_on="node", how="left"
)

print(df_corr[["semantic_distance", "pagerank"]].corr())

print("--------- Betweenness and avg distance correlation ---------")
# Load Betweenness
df_pr = pd.read_csv("results/centrality_metrics_enriched.csv")  # ou ton fichier Gephi exporté
df_pr = df_pr[df_pr["node_type"] == "company"][["node", "betweenness"]]

# Avg distance
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

# Merge
df_corr = avg_dist.merge(
    df_pr, left_on="company", right_on="node", how="left"
)

print(df_corr[["semantic_distance", "betweenness"]].corr())


print("--------- Closeness and avg distance correlation ---------")
# Closeness 
df_pr = pd.read_csv("results/centrality_metrics_enriched.csv")  # ou ton fichier Gephi exporté
df_pr = df_pr[df_pr["node_type"] == "company"][["node", "closeness"]]

# Avg distance
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

# Merge
df_corr = avg_dist.merge(
    df_pr, left_on="company", right_on="node", how="left"
)

print(df_corr[["semantic_distance", "closeness"]].corr())