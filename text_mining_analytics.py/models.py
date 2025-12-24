import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score



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

def run_kmeans_on_nmf(W, n_clusters=5): # Clustering on NMF results (matrix W)

    # Important : Normalize W before clustering
    W_scaled = normalize(W)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000, random_state=42)
    clusters = kmeans.fit_predict(W_scaled)

    
    return kmeans, clusters

def run_kmeans_on_nmf_angular(W, n_clusters=5): # Clustering on NMF results (matrix W) with angular distance
    
    # Normalize L2 for angular distance
    W_angular = normalize(W, norm='l2') 

    # Increase n_init for better convergence
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=50, 
        max_iter=1000, 
        random_state=42
    )
    clusters = kmeans.fit_predict(W_angular)
    
    return kmeans, clusters

def run_kmeans_angular_tfidf(X, n_clusters=5): # Clustering on TF-IDF with angular distance

    # Normalize L2 for angular distance
    X_angular = normalize(X, norm='l2') 

    # Increase n_init for better convergence
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=50, 
        max_iter=1000, 
        random_state=42
    )
    clusters = kmeans.fit_predict(X_angular)
    
    return kmeans, clusters

def get_cluster_keywords(X, clusters, terms, n_clusters, n_words=10):
    cluster_keywords = {}
    for i in range(n_clusters):
        indices = np.where(clusters == i)[0]
        mean_tfidf = X[indices].mean(axis=0)
        top_indices = np.argsort(mean_tfidf)[-n_words:][::-1]
        cluster_keywords[i] = [terms[idx] for idx in top_indices]
    return cluster_keywords

def get_silhouette_validation(X, labels):
    """Calculate the global silhouette score."""
    X_scaled = normalize(X)
    score = silhouette_score(X_scaled, labels)
    return score

def get_detailed_silhouette(X, labels):
    """Calculate the average silhouette score for each cluster individually."""
    # Ensure using angular distance (L2 norm)
    from sklearn.preprocessing import normalize
    X_norm = normalize(X, norm='l2')
    
    sample_scores = silhouette_samples(X_norm, labels)
    df_scores = pd.DataFrame({'cluster': labels, 'silhouette': sample_scores})
    
    # Average per cluster
    cluster_stats = df_scores.groupby('cluster')['silhouette'].mean()
    return cluster_stats
