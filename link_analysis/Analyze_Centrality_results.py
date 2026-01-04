import pandas as pd
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

INPUT_PATH = "results/centrality_metrics.csv"
OUTPUT_DIR = "results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD RESULTS
# =========================

df = pd.read_csv(INPUT_PATH)

# =========================
# GLOBAL STATISTICS
# =========================

global_stats = (
    df[["degree", "weighted_degree", "betweenness", "pagerank"]]
    .describe()
    .round(4)
)

global_stats.to_csv(
    os.path.join(OUTPUT_DIR, "global_statistics.csv")
)

# =========================
# AVERAGE VALUES
# =========================

average_values = (
    df[["degree", "weighted_degree", "betweenness", "pagerank"]]
    .mean()
    .round(6)
    .to_frame(name="average")
)

average_values.to_csv(
    os.path.join(OUTPUT_DIR, "average_values.csv")
)

# =========================
# TOP N NODES BY METRIC
# =========================

TOP_N = 15

metrics = [
    "degree",
    "weighted_degree",
    "betweenness",
    "pagerank"
]

for metric in metrics:
    top_nodes = (
        df.sort_values(metric, ascending=False)
        .head(TOP_N)
    )

    top_nodes.to_csv(
        os.path.join(OUTPUT_DIR, f"top_{metric}.csv"),
        index=False
    )

# =========================
# AVERAGE BY NODE TYPE
# =========================

avg_by_type = (
    df.groupby("node_type")[["degree", "weighted_degree", "betweenness", "pagerank"]]
    .mean()
    .round(6)
)

avg_by_type.to_csv(
    os.path.join(OUTPUT_DIR, "average_by_node_type.csv")
)

# =========================
# CORE NODES (ABOVE AVERAGE)
# =========================

avg = df[["degree", "weighted_degree", "betweenness", "pagerank"]].mean()

core_nodes = df[
    (df["weighted_degree"] > avg["weighted_degree"]) &
    (df["pagerank"] > avg["pagerank"])
].sort_values("pagerank", ascending=False)

core_nodes.to_csv(
    os.path.join(OUTPUT_DIR, "core_nodes.csv"),
    index=False
)

# =========================
# SUMMARY FILE (VERY IMPORTANT)
# =========================

summary = pd.DataFrame({
    "total_nodes": [len(df)],
    "core_nodes": [len(core_nodes)],
    "core_ratio": [round(len(core_nodes) / len(df), 3)]
})

summary.to_csv(
    os.path.join(OUTPUT_DIR, "network_summary.csv"),
    index=False
)

print("✅ Analysis files successfully generated.")
print(f"📁 Results available in: {OUTPUT_DIR}")