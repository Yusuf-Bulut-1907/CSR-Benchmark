import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

def run_kmeans(X, n_clusters=5):
    X_scaled = normalize(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def run_nmf(X, terms, n_components=8):
    nmf = NMF(n_components=n_components, random_state=42, init='nndsvd')
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    topics = {}
    for i, topic in enumerate(H):
        top_indices = topic.argsort()[-10:][::-1]
        topics[f"Topic {i}"] = [terms[idx] for idx in top_indices]
    return W, H, topics

def get_cluster_keywords(X, clusters, terms, n_clusters, n_words=10):
    cluster_keywords = {}
    for i in range(n_clusters):
        indices = np.where(clusters == i)[0]
        mean_tfidf = X[indices].mean(axis=0)
        top_indices = np.argsort(mean_tfidf)[-n_words:][::-1]
        cluster_keywords[i] = [terms[idx] for idx in top_indices]
    return cluster_keywords

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

def get_silhouette_validation(X, labels):
    """Calcule le score de silhouette global."""
    X_scaled = normalize(X)
    score = silhouette_score(X_scaled, labels)
    return score