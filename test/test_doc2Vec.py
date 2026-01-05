"""
Doc2Vec-based Semantic Embedding and Clustering Pipeline
-------------------------------------------------------

This script implements a Doc2Vec (Distributed Memory) approach to learn
dense semantic representations of CSR-related documents.

The pipeline:
- Trains a Doc2Vec model on cleaned CSR documents
- Computes document-level embeddings
- Aggregates embeddings at the company level
- Applies K-Means clustering using angular (cosine) distance
- Evaluates clustering quality using silhouette metrics

This approach enables comparison with TF-IDF/NMF and BERT-based pipelines.
"""
import sys
import os

# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================
# Ensure the project root is accessible for absolute imports
project_root = "/Users/matteogalizia/Documents/GitHub/CSR-Benchmark"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# 2. STANDARD AND THIRD-PARTY IMPORTS
# ============================================================
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score

# ============================================================
# 3. INTERNAL PROJECT IMPORTS
# ============================================================
from text_mining.stats_and_cleaning import get_cleaned_corpus

def train_doc2vec(
    vector_size=300,
    window=10,
    min_count=5,
    epochs=40,
    seed=42
):
    """
    Train a Doc2Vec (Distributed Memory) model on the cleaned CSR corpus.

    Parameters
    ----------
    vector_size : int
        Dimensionality of the document embeddings.
    window : int
        Context window size.
    min_count : int
        Minimum word frequency threshold.
    epochs : int
        Number of training epochs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_doc : np.ndarray
        Document-level embeddings (n_documents × vector_size).
    X_company : np.ndarray
        Company-level embeddings after mean pooling.
    companies : pd.Index
        Company identifiers aligned with X_company.
    metadata : list
        Original document metadata.
    """

    # ============================================================
    # 1. LOAD CLEANED CORPUS
    # ============================================================
    documents, metadata = get_cleaned_corpus()

    # Simple tokenization only:
    # - No lemmatization
    # - No n-grams
    # This is intentional to preserve Doc2Vec's internal semantics.
    tagged_docs = [
        TaggedDocument(words=doc.lower().split(), tags=[str(i)])
        for i, doc in enumerate(documents)
    ]

    # ============================================================
    # 2. TRAIN DOC2VEC MODEL
    # ============================================================
    # Distributed Memory (dm=1) is preferred for document representation
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1,             
        seed=seed
    )

    model.build_vocab(tagged_docs)
    model.train(
        tagged_docs,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    # ============================================================
    # 3. DOCUMENT-LEVEL EMBEDDINGS
    # ============================================================
    # Extract learned document vectors
    X_doc = np.array([model.dv[str(i)] for i in range(len(tagged_docs))])

    # L2-normalization enables cosine / angular distance
    X_doc = normalize(X_doc)  

    # ============================================================
    # 4. COMPANY-LEVEL AGGREGATION (MEAN POOLING)
    # ============================================================
    # Each company is represented as the centroid of its documents
    df_doc = pd.DataFrame(X_doc)
    df_doc["company"] = [m.get("company") for m in metadata]

    df_company = df_doc.groupby("company").mean()

    X_company = normalize(df_company.values)
    companies = df_company.index

    return X_doc, X_company, companies, metadata

def run_kmeans_angular_doc2vec(X, n_clusters=5): 
    """
    Apply K-Means clustering using angular distance on Doc2Vec embeddings.

    Angular distance is approximated by applying K-Means on L2-normalized
    vectors, which corresponds to cosine similarity optimization.

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix (documents or companies).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    kmeans : KMeans
        Trained KMeans model.
    clusters : np.ndarray
        Cluster labels.
    """

    X_angular = normalize(X, norm='l2') 

    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=50, 
        max_iter=1000, 
        random_state=42
    )
    clusters = kmeans.fit_predict(X_angular)
    
    return kmeans, clusters

def get_silhouette_validation(X, labels):
    """
    Compute the global silhouette score for a clustering solution.

    Returns
    -------
    float
        Global silhouette score.
    """
    X_scaled = normalize(X)
    score = silhouette_score(X_scaled, labels)
    return score

def get_detailed_silhouette(X, labels):
    """
    Compute average silhouette scores per cluster.

    This provides a cluster-level confidence indicator and helps
    detect overlapping or poorly separated clusters.

    Returns
    -------
    pandas.Series
        Mean silhouette score per cluster.
    """
    X_norm = normalize(X, norm='l2')
    
    sample_scores = silhouette_samples(X_norm, labels)
    df_scores = pd.DataFrame({'cluster': labels, 'silhouette': sample_scores})
    
    cluster_stats = df_scores.groupby('cluster')['silhouette'].mean()
    return cluster_stats

# ============================================================
# MAIN – STANDALONE EXECUTION / DEBUGGING
# ============================================================
if __name__ == "__main__":

    # ============================================================
    # EMBEDDING COMPUTATION
    # ============================================================
    X_doc, X_company, companies, meta = train_doc2vec()

    # ============================================================
    # CLUSTERING AT COMPANY LEVEL
    # ============================================================
    print(X_company.shape)
    print("Running KMeans clustering on company-level Doc2Vec embeddings...")

    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")

        kmeans, clusters = run_kmeans_angular_doc2vec(
            X_company, 
            n_clusters=i
        )

        silhouette = get_silhouette_validation(X_company, clusters)
        detailed_silhouette = get_detailed_silhouette(X_company, clusters)

        score_global = get_silhouette_validation(
            normalize(X_company, norm="l2"),
            clusters
        )

        print(f"⭐ Global silhouette score (Doc2Vec): {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)
    
    # ============================================================
    # CLUSTERING AT DOCUMENT LEVEL
    # ============================================================
    print(X_doc.shape)
    print("Running KMeans clustering on document-level Doc2Vec embeddings...")
    
    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")
        
        kmeans, clusters = run_kmeans_angular_doc2vec(
            X_doc, 
            n_clusters=i
        )

        silhouette = get_silhouette_validation(X_doc, clusters)
        detailed_silhouette = get_detailed_silhouette(X_doc, clusters)

        score_global = get_silhouette_validation(
            normalize(X_doc, norm="l2"),
            clusters
        )

        print(f"⭐ Global silhouette score (Doc2Vec): {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)
