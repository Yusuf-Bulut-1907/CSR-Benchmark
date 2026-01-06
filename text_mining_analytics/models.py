"""
Clustering and Topic Modeling Utilities
---------------------------------------

This module provides reusable functions for:
- K-means clustering on TF-IDF and NMF representations
- Non-negative Matrix Factorization (NMF) for topic extraction
- Angular (cosine-based) clustering through L2 normalization
- Cluster interpretation via top keywords
- Internal validation using silhouette scores

These methods support exploratory analysis of CSR / ESG textual corpora
at the company level.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score



# ==========================
# K-MEANS ON TF-IDF
# ==========================

def run_kmeans(X, n_clusters=5):
    """
    Apply K-means clustering on a TF-IDF matrix.

    The matrix is L2-normalized prior to clustering,
    which approximates cosine (angular) distance.

    Parameters
    ----------
    X : array-like or sparse matrix
        TF-IDF representation of documents.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    kmeans : sklearn.cluster.KMeans
        Fitted K-means model.
    clusters : ndarray
        Cluster assignment for each document.
    """
    X_scaled = normalize(X)
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=10, max_iter=1000, 
        random_state=42
        )
    
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

# ==========================
# NON-NEGATIVE MATRIX FACTORIZATION
# ==========================

def run_nmf(X, terms, n_components=8):
    """
    Perform Non-negative Matrix Factorization (NMF)
    to extract latent semantic topics.

    Parameters
    ----------
    X : array-like or sparse matrix
        TF-IDF matrix.
    terms : list-like
        Vocabulary corresponding to columns of X.
    n_components : int
        Number of latent topics.

    Returns
    -------
    W : ndarray
        Document-topic matrix.
    H : ndarray
        Topic-term matrix.
    topics : dict
        Dictionary mapping each topic to its top representative terms.
    """

    nmf = NMF(
        n_components=n_components, 
        random_state=42, 
        init='nndsvd'
        )
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    # Extract top terms per topic for interpretability
    topics = {}
    for i, topic in enumerate(H):
        top_indices = topic.argsort()[-10:][::-1]
        topics[f"Topic {i}"] = [terms[idx] for idx in top_indices]
    return W, H, topics

# ==========================
# K-MEANS ON NMF REPRESENTATION
# ==========================

def run_kmeans_on_nmf(W, n_clusters=5): # Clustering on NMF results (matrix W)
    """
    Apply K-means clustering on the NMF document-topic matrix (W).

    Normalization ensures that clustering is driven by topic proportions
    rather than document length.
    """
    # Normalize to reduce scale effects
    W_scaled = normalize(W)
    
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=20, 
        max_iter=1000, 
        random_state=42
        )
    
    clusters = kmeans.fit_predict(W_scaled)
    return kmeans, clusters

def run_kmeans_on_nmf_angular(W, n_clusters=5): # Clustering on NMF results (matrix W) with angular distance
    """
    Apply K-means clustering on NMF outputs using angular distance.

    L2 normalization transforms Euclidean distance into
    cosine-based similarity.
    """
    # L2 normalization for angular distance
    W_angular = normalize(W, norm='l2') 

    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=50, 
        max_iter=1000, 
        random_state=42
    )
    clusters = kmeans.fit_predict(W_angular)
    return kmeans, clusters

# ==========================
# ANGULAR K-MEANS ON TF-IDF
# ==========================

def run_kmeans_angular_tfidf(X, n_clusters=5): # Clustering on TF-IDF with angular distance
    """
    Apply K-means clustering on TF-IDF using angular (cosine) distance.

    This configuration is standard for high-dimensional text data.
    """
    # L2 normalization for cosine similarity
    X_angular = normalize(X, norm='l2') 

    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=50, 
        max_iter=1000, 
        random_state=42
    )
    clusters = kmeans.fit_predict(X_angular)
    return kmeans, clusters



# ==========================
# CLUSTER INTERPRETATION
# ==========================

def get_cluster_keywords(X, clusters, terms, n_clusters, n_words=10):
    """
    Extract representative keywords for each cluster.

    The method computes the average TF-IDF vector per cluster
    and selects the highest-weighted terms.

    Parameters
    ----------
    X : array-like or sparse matrix
        TF-IDF matrix.
    clusters : ndarray
        Cluster labels.
    terms : list-like
        Vocabulary.
    n_clusters : int
        Number of clusters.
    n_words : int
        Number of top terms to extract per cluster.
    """
    cluster_keywords = {}

    for i in range(n_clusters):
        indices = np.where(clusters == i)[0]

      # Mean TF-IDF vector for the cluster
        mean_tfidf = X[indices].mean(axis=0)

        top_indices = np.argsort(mean_tfidf)[-n_words:][::-1]
        cluster_keywords[i] = [terms[idx] for idx in top_indices]
    return cluster_keywords


# ==========================
# CLUSTER VALIDATION
# ==========================

def get_silhouette_validation(X, labels):
    """
    Compute the global silhouette score.

    A higher score indicates better cluster separation.
    """
    X_scaled = normalize(X)
    score = silhouette_score(X_scaled, labels)
    return score


def get_detailed_silhouette(X, labels):
    """"
    Compute the average silhouette score for each cluster.

    This allows identification of weak or unstable clusters.
    """
    # L2 normalization ensures angular distance
    from sklearn.preprocessing import normalize

    X_norm = normalize(X, norm='l2')
    
    sample_scores = silhouette_samples(X_norm, labels)

    df_scores = pd.DataFrame({
        'cluster': labels,
        'silhouette': sample_scores
        })
    
    # Mean silhouette score per cluster
    cluster_stats = df_scores.groupby('cluster')['silhouette'].mean()
    return cluster_stats
