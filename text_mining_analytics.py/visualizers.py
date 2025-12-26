import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_samples
import numpy as np

def plot_heatmap(df, row_col, cluster_col, title, filename=None):
    # calculate counts for each cluster
    counts = df[cluster_col].value_counts().sort_index()
    
    heatmap_data = pd.crosstab(df[row_col], df[cluster_col], normalize='index')
    
    # Rename columns to show "N" (number of companies)
    heatmap_data.columns = [f"C{i} (n={counts[i]})" for i in counts.index]

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="YlGnBu", linewidths=.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"results/{filename}")
        print(f"✅ Graph saved : results/{filename}")
    plt.show()

def plot_elbow_method(X, k_range=range(1, 10), filename="elbow_curve.png"):
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    X_scaled = normalize(X)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
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

def plot_silhouette(X, labels, n_clusters, filename="silhouette_plot.png"):
    from sklearn.preprocessing import normalize
    X_scaled = normalize(X)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    y_lower = 10
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    avg_score = np.mean(sample_silhouette_values)

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.axvline(x=avg_score, color="red", linestyle="--", label=f"Average ({avg_score:.2f})")
    ax1.set_title("Silhouette Plot by Cluster")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Clusters")
    ax1.legend()
    
    if filename:
        plt.savefig(f"results/{filename}")
    plt.show()