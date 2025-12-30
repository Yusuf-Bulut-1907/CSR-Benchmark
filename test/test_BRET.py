import sys
import os

# On s'assure que la racine du projet est dans le path
project_root = "/Users/matteogalizia/Documents/GitHub/CSR-Benchmark"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Imports standards
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 3. Import de votre fonction (maintenant que le path et l'import interne sont fixés)
from text_mining.stats_and_cleaning import get_cleaned_corpus

# Le reste de votre code...

def compute_bert_embeddings(
    model_name="all-MiniLM-L6-v2",
    batch_size=16
):
    """
    Compute BERT embeddings on cleaned documents.

    Returns
    -------
    X_doc : np.ndarray
        Document-level embeddings
    X_company : np.ndarray
        Company-level embeddings after mean pooling
    companies : pd.Index
        Company labels
    metadata : list
        Original metadata
    """

    # ============================================================
    # 1. LOAD CLEANED CORPUS (NO TOKENIZATION)
    # ============================================================
    documents, metadata = get_cleaned_corpus()

    # ============================================================
    # 2. LOAD PRETRAINED MODEL
    # ============================================================
    model = SentenceTransformer(model_name)

    # ============================================================
    # 3. DOCUMENT-LEVEL EMBEDDINGS
    # ============================================================
    X_doc = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-ready
    )

    # ============================================================
    # 4. COMPANY-LEVEL AGGREGATION (MEAN POOLING)
    # ============================================================
    df_doc = pd.DataFrame(X_doc)
    df_doc["company"] = [m.get("company") for m in metadata]

    df_company = df_doc.groupby("company").mean()

    X_company = normalize(df_company.values)
    companies = df_company.index

    return X_doc, X_company, companies, metadata
def run_kmeans_angular_BERT(X, n_clusters=5): # Clustering on BERT with angular distance

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

    X_doc, X_company, companies, meta = compute_bert_embeddings()

    '''# Run KMeans clustering on X_company
    print(X_company.shape)
    print("Running KMeans clustering X_company...")
    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")
         # Run KMeans clustering
        kmeans, clusters = run_kmeans_angular_BERT(X_company, n_clusters=i)

    # Evaluate clustering
        silhouette = get_silhouette_validation(X_company, clusters)
        detailed_silhouette = get_detailed_silhouette(X_company, clusters)
        X_norm = normalize(X_company, norm="l2")

        score_global = get_silhouette_validation(X_norm, clusters)
        print(f"⭐ Global silhouette score (BERT) : {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)'''
    
    # Run KMeans clustering on X_doc
    print(X_doc.shape)
    print("Running KMeans clustering X_doc...")
    for i in range(3, 6):
        print(f"\nNumber of clusters: {i}")
         # Run KMeans clustering
        kmeans, clusters = run_kmeans_angular_BERT(X_doc, n_clusters=i)
    # Evaluate clustering
        silhouette = get_silhouette_validation(X_doc, clusters)
        detailed_silhouette = get_detailed_silhouette(X_doc, clusters)
        X_norm = normalize(X_doc, norm="l2")

        score_global = get_silhouette_validation(X_norm, clusters)
        print(f"⭐ Global silhouette score (BERT) : {score_global:.3f}")
        print("Global Silhouette Score:", silhouette)
        print("Detailed Silhouette Scores:")
        print(detailed_silhouette)