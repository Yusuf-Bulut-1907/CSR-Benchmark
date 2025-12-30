import sys
import os

# On s'assure que la racine du projet est dans le path
project_root = "/Users/matteogalizia/Documents/GitHub/CSR-Benchmark"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
from text_mining.stats_and_cleaning import get_cleaned_corpus

def train_doc2vec(
    vector_size=300,
    window=10,
    min_count=5,
    epochs=40,
    seed=42
):
    """
    Train a Doc2Vec model on cleaned documents.

    Returns
    -------
    X_doc : np.ndarray
        Document-level embeddings (n_documents, vector_size)
    X_company : np.ndarray
        Company-level embeddings after mean pooling (n_companies, vector_size)
    companies : pd.Index
        Company labels aligned with X_company
    metadata : list
        Original metadata
    """

    # ============================================================
    # 1. LOAD CLEANED CORPUS
    # ============================================================
    documents, metadata = get_cleaned_corpus()

    # Simple tokenization ONLY (no lemmatization, no n-grams)
    tagged_docs = [
        TaggedDocument(words=doc.lower().split(), tags=[str(i)])
        for i, doc in enumerate(documents)
    ]

    # ============================================================
    # 2. TRAIN DOC2VEC MODEL
    # ============================================================
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1,              # Distributed Memory (recommended)
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
    X_doc = np.array([model.dv[str(i)] for i in range(len(tagged_docs))])
    X_doc = normalize(X_doc)  # angular / cosine-ready

    # ============================================================
    # 4. COMPANY-LEVEL AGGREGATION (MEAN POOLING)
    # ============================================================
    df_doc = pd.DataFrame(X_doc)
    df_doc["company"] = [m.get("company") for m in metadata]

    df_company = df_doc.groupby("company").mean()

    X_company = normalize(df_company.values)
    companies = df_company.index

    return X_doc, X_company, companies, metadata

def run_kmeans_angular_doc2vec(X, n_clusters=5): # Clustering on TF-IDF with angular distance

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

# ============================================================
# MAIN (DEBUG / STANDALONE RUN)
# ============================================================
if __name__ == "__main__":

    X_doc, X_company, companies, meta = train_doc2vec()

    # Run KMeans clustering on X_company
    print(X_company.shape)
    print("Running KMeans clustering X_company...")
    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")
         # Run KMeans clustering
        kmeans, clusters = run_kmeans_angular_doc2vec(X_company, n_clusters=i)

    # Evaluate clustering
        silhouette = get_silhouette_validation(X_company, clusters)
        detailed_silhouette = get_detailed_silhouette(X_company, clusters)
        X_norm = normalize(X_company, norm="l2")

        score_global = get_silhouette_validation(X_norm, clusters)
        print(f"⭐ Global silhouette score (Doc2Vec) : {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)
    
    # Run KMeans clustering on X_doc
    print(X_doc.shape)
    print("Running KMeans clustering X_doc...")
    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")
         # Run KMeans clustering
        kmeans, clusters = run_kmeans_angular_doc2vec(X_doc, n_clusters=i)
    # Evaluate clustering
        silhouette = get_silhouette_validation(X_doc, clusters)
        detailed_silhouette = get_detailed_silhouette(X_doc, clusters)
        X_norm = normalize(X_doc, norm="l2")

        score_global = get_silhouette_validation(X_norm, clusters)
        print(f"⭐ Global silhouette score (Doc2Vec) : {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)
