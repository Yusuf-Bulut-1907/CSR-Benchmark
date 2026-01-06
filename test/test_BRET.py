"""
BERT-based Semantic Embedding and Clustering Pipeline
-----------------------------------------------------

This script computes semantic document embeddings using a pretrained
Sentence-BERT model, aggregates them at the company level, and applies
K-Means clustering using angular (cosine) distance.

The objective is to compare semantic clustering (BERT embeddings)
with traditional TF-IDF-based approaches.
"""
import sys
import os

# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================
# Ensure that the project root directory is included in sys.path
# to allow absolute imports across the project structure.
project_root = ""
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# 2. STANDARD LIBRARY AND THIRD-PARTY IMPORTS
# ============================================================
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# ============================================================
# 3. INTERNAL PROJECT IMPORTS
# ============================================================
# Import the cleaned corpus (text + metadata)
from text_mining.stats_and_cleaning import get_cleaned_corpus

# Le reste de votre code...

def compute_bert_embeddings(
    model_name="all-MiniLM-L6-v2",
    batch_size=16
):
    """
    Compute Sentence-BERT embeddings for the cleaned corpus.

    The function generates:
    - Document-level embeddings
    - Company-level embeddings via mean pooling

    Parameters
    ----------
    model_name : str
        Name of the pretrained SentenceTransformer model.
    batch_size : int
        Batch size for embedding computation.

    Returns
    -------
    X_doc : np.ndarray
        Dense document-level embeddings.
    X_company : np.ndarray
        Company-level embeddings obtained by averaging documents.
    companies : pd.Index
        Company identifiers.
    metadata : list
        Original document metadata.
    """

    # ============================================================
    # 1. LOAD CLEANED CORPUS
    # ============================================================
    # No tokenization or linguistic preprocessing is applied here,
    # as BERT models handle raw text internally.
    documents, metadata = get_cleaned_corpus()

    # ============================================================
    # 2. LOAD PRETRAINED SENTENCE-BERT MODEL
    # ============================================================
    model = SentenceTransformer(model_name)

    # ============================================================
    # 3. DOCUMENT-LEVEL EMBEDDINGS
    # ============================================================
    # Embeddings are L2-normalized to enable cosine similarity
    # and angular distance computations.
    X_doc = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-ready
    )

    # ============================================================
    # 4. COMPANY-LEVEL AGGREGATION (MEAN POOLING)
    # ============================================================
    # Each company is represented by the centroid of its documents.
    df_doc = pd.DataFrame(X_doc)
    df_doc["company"] = [m.get("company") for m in metadata]

    df_company = df_doc.groupby("company").mean()

    # Final normalization for angular distance
    X_company = normalize(df_company.values)
    companies = df_company.index

    return X_doc, X_company, companies, metadata

def run_kmeans_angular_BERT(X, n_clusters=5): # Clustering on BERT with angular distance
    """
    Apply K-Means clustering using angular distance on BERT embeddings.

    Angular distance is approximated by applying K-Means on L2-normalized
    vectors, which is equivalent to cosine similarity optimization.

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

    # Normalize embeddings for angular distance
    X_angular = normalize(X, norm='l2') 

    # High n_init ensures better convergence in high-dimensional spaces
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

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    labels : array-like
        Cluster labels.

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
    Compute the average silhouette score per cluster.

    This provides a cluster-level confidence indicator and helps
    identify poorly separated clusters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    labels : array-like
        Cluster labels.

    Returns
    -------
    pandas.Series
        Mean silhouette score for each cluster.
    """
    X_norm = normalize(X, norm='l2')
    
    sample_scores = silhouette_samples(X_norm, labels)
    df_scores = pd.DataFrame({
        'cluster': labels, 
        'silhouette': sample_scores
    })
    
    cluster_stats = df_scores.groupby('cluster')['silhouette'].mean()
    return cluster_stats


# ============================================================
# MAIN – STANDALONE EXECUTION / DEBUGGING
# ============================================================
if __name__ == "__main__":

    # ============================================================
    # EMBEDDING COMPUTATION
    # ============================================================
    X_doc, X_company, companies, meta = compute_bert_embeddings()

    # ============================================================
    # CLUSTERING ON DOCUMENT-LEVEL EMBEDDINGS
    # ============================================================
    print(X_doc.shape)
    print("Running KMeans clustering on document-level BERT embeddings...")

    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")

        # Run K-Means clustering
        kmeans, clusters = run_kmeans_angular_BERT(
            X_doc,
            n_clusters=i
        )

        # ========================================================
        # CLUSTER VALIDATION
        # ========================================================
        silhouette = get_silhouette_validation(X_doc, clusters)
        detailed_silhouette = get_detailed_silhouette(X_doc, clusters)

        X_norm = normalize(X_doc, norm="l2")
        score_global = get_silhouette_validation(X_norm, clusters)

        print(f"⭐ Global silhouette score (BERT): {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)