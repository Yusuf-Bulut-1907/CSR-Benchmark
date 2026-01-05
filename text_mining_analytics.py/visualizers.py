"""
Clustering Visualization and Validation Tools
---------------------------------------------

This module provides visualization utilities for:
- Cluster composition analysis (heatmaps)
- Cluster number selection (Elbow method)
- Cluster quality assessment (Silhouette analysis)

These plots support the interpretability and validation
of clustering results in a text mining context.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_samples
import numpy as np


# ==========================
# HEATMAP: CLUSTER COMPOSITION
# ==========================

def plot_heatmap(df, row_col, cluster_col, title, filename=None):
    """
    Plot a normalized heatmap showing the distribution of clusters
    across a categorical variable (e.g. sector, country).

    Each row is normalized, allowing interpretation in terms of
    proportions rather than absolute counts.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing clustering results and metadata.
    row_col : str
        Categorical variable used for rows (e.g. sector).
    cluster_col : str
        Column containing cluster labels.
    title : str
        Title of the heatmap.
    filename : str, optional
        If provided, the plot is saved in the 'results/' folder.
    """

    # Number of companies per cluster
    counts = df[cluster_col].value_counts().sort_index()
    
    # Row-normalized contingency table
    heatmap_data = pd.crosstab(
        df[row_col], 
        df[cluster_col], 
        normalize='index'
    )
    
    # Rename columns to include cluster size
    heatmap_data.columns = [
        f"C{i} (n={counts[i]})" for i in counts.index
    ]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1%", 
        cmap="YlGnBu", 
        linewidths=.5
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"results/{filename}")
        print(f"✅ Graph saved : results/{filename}")
    plt.show()


# ==========================
# ELBOW METHOD
# ==========================

def plot_elbow_method(X, k_range=range(1, 10), filename="elbow_curve.png"):    
    """
    Plot the Elbow curve to help determine the optimal
    number of clusters.

    The method analyzes the evolution of within-cluster
    inertia as k increases.

    Parameters
    ----------
    X : array-like or sparse matrix
        Feature matrix (e.g. TF-IDF or NMF matrix).
    k_range : iterable
        Range of k values to evaluate.
    filename : str
        Name of the output file.
    """
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    # Normalize for cosine/angle-based clustering
    X_scaled = normalize(X)

    inertias = []
    for k in k_range:
        km = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10
        )
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    plt.figure()
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow method For Optimal k')

    plt.savefig(f"results/{filename}")
    print(f"✅ Elbow curve saved : results/{filename}")

    plt.show()

# ==========================
# SILHOUETTE ANALYSIS
# ==========================
def plot_silhouette(X, labels, n_clusters, filename="silhouette_plot.png"):
    """
    Generate a silhouette plot to evaluate cluster cohesion
    and separation.

    The silhouette coefficient measures how similar a document
    is to its own cluster compared to other clusters.

    Parameters
    ----------
    X : array-like or sparse matrix
        Feature matrix.
    labels : array-like
        Cluster labels.
    n_clusters : int
        Number of clusters.
    filename : str
        Name of the output file.
    """
    from sklearn.preprocessing import normalize

    # L2 normalization for angular distance
    X_scaled = normalize(X)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    y_lower = 10

    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    avg_score = np.mean(sample_silhouette_values)

    for i in range(n_clusters):
        # Extract silhouette values for cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper), 
            0, 
            ith_cluster_silhouette_values,
            facecolor=color, 
            edgecolor=color, 
            alpha=0.7
        )

        # Label cluster number
        ax1.text(
            -0.05, 
            y_lower + 0.5 * size_cluster_i, 
            str(i)
        )

        y_lower = y_upper + 10

    # Global average silhouette score
    ax1.axvline(
        x=avg_score, 
        color="red", 
        linestyle="--", 
        label=f"Average ({avg_score:.2f})"
    )
    
    ax1.set_title("Silhouette Plot by Cluster")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Clusters")
    ax1.legend()
    
    if filename:
        plt.savefig(f"results/{filename}")
    plt.show()