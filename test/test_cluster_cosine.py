"""
NMF-based Topic Modeling and CSR Clustering Pipeline
--------------------------------------------------

This script applies Non-negative Matrix Factorization (NMF) on a TF-IDF
representation (uni-, bi- and trigrams) to extract latent CSR topics.

The extracted topic intensities are then used as low-dimensional,
interpretable features for K-Means clustering.

This approach is particularly suitable for CSR benchmarking as it:
- Produces human-interpretable topics
- Reduces dimensionality while preserving semantic structure
- Enables a clear business-oriented interpretation of clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


# ============================================================
# 1. DATA LOADING
# ============================================================
# TF-IDF matrix built on unigrams, bigrams and trigrams
# Rows correspond to companies, columns to lexical features
XPATH = "data/TFIDF_unigram_bigram_trigram.csv"
XDF = pd.read_csv(XPATH, index_col=0)

# Convert to NumPy array for matrix factorization
X = XDF.values

# Number of latent CSR topics to extract
n_components = 6


# ============================================================
# 2. DIMENSIONALITY REDUCTION WITH NMF
# ============================================================
# Non-negative Matrix Factorization is well suited for CSR analysis:
# - It ensures additive (parts-based) representations
# - Each topic can be interpreted as a CSR dimension
# - Topic weights reflect the importance of each theme for a company
nmf = NMF(
    n_components=n_components,
    init='nndsvd',      # improves convergence and stability
    random_state=42
)

# W matrix: topic intensity per company
# Shape: (n_companies, n_topics)
W = nmf.fit_transform(X)


# ============================================================
# 3. NORMALIZATION AND CLUSTERING
# ============================================================
# L2 normalization ensures that clustering focuses on topic proportions
# rather than absolute magnitudes
W_norm = normalize(W, norm='l2')

# Optimal number of clusters selected via validation (silhouette, elbow, etc.)
n_clusters = 6

kmeans = KMeans(
    n_clusters=n_clusters,
    n_init=50,
    random_state=42
)

labels = kmeans.fit_predict(W_norm)

# Global clustering quality indicator
print(f"Silhouette Score : {silhouette_score(W_norm, labels):.4f}")


# ============================================================
# 4. CLUSTER INTERPRETATION (BUSINESS VALIDATION)
# ============================================================
# This block allows semantic interpretation of clusters
# by identifying dominant CSR topics within each group

# Build an analysis DataFrame with topic scores
df_analysis = pd.DataFrame(
    W,
    columns=[f'Topic_{i}' for i in range(n_components)],
    index=XDF.index
)

# Append cluster labels
df_analysis['Cluster'] = labels

# Compute average topic importance per cluster
cluster_profiles = df_analysis.groupby('Cluster').mean()

# Visual inspection of cluster-topic relationships
plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_profiles,
    annot=True,
    cmap='Blues'
)
plt.title("Topic Importance by Cluster (Business Validation)")
plt.ylabel("Cluster ID")
plt.xlabel("CSR Topics (NMF Components)")
plt.tight_layout()
plt.show()


# ============================================================
# 5. FINAL EXPORT FOR CSR BENCHMARKING
# ============================================================
# The final cluster label can now be used as:
# - A CSR typology of companies
# - An explanatory variable in downstream analyses
# - A benchmarking segmentation tool

XDF['Cluster_Final'] = labels
XDF[['Cluster_Final']].to_csv("benchmark_csr_final.csv")