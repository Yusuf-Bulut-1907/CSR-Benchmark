from loaders import load_tfidf_data, load_metadata, merge_results
from processing import compute_cosine_similarity, compute_cooccurrence, get_top_terms
from models import (
    run_kmeans, run_nmf, get_cluster_keywords, 
    get_silhouette_validation, run_kmeans_on_nmf, 
    run_kmeans_on_nmf_angular, run_kmeans_angular_tfidf,
    get_detailed_silhouette 
)
from visualizers import plot_heatmap, plot_elbow_method, plot_silhouette
from sklearn.preprocessing import normalize
import numpy as np
import os

def main_text_analysis():
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)
    # 1. Data Loading
    df_tfidf, terms, companies = load_tfidf_data("data/TFIDF_unigram_bigram_trigram.csv")
    df_uni, terms_uni, _ = load_tfidf_data("data/TFIDF_unigram.csv")
    df_meta = load_metadata("data/metadata.csv")

    # 2. Basic Text Analysis
    cosine_sim_df = compute_cosine_similarity(df_tfidf.values, companies)
    cosine_sim_df.to_csv("results/cosine_similarity_tf_idf.csv")
    
    cooc_df = compute_cooccurrence(df_uni.values, terms_uni)
    cooc_df.to_csv("results/cooccurrence_matrix_tf_idf.csv")

    # 3. Topic Modeling (NMF)
    n_topics = 6
    W, H, topics = run_nmf(df_tfidf.values, terms, n_components=n_topics)
    print(f"✅ NMF terminée : Données compressées à {n_topics} thèmes.")
    
    for t, words in topics.items():
        print(f"{t}: {', '.join(words)}")

    # 4. Clustering (Angular distance on W from NMF)
    # if clustering on W from NMF
    # euclidean distance
    '''
    km_model, clusters = run_kmeans_on_nmf(W, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)
    score = get_silhouette_validation(W, clusters)
    plot_silhouette(W, clusters, n_clusters, filename="silhouette_validation.png")
    plot_elbow_method(W, filename="elbow_optimization_nmf.png")
    '''
    # if clustering on TF-IDF directly
    # euclidean distance
    '''
    km_model, clusters = run_kmeans(df_tfidf.values, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)
    score = get_silhouette_validation(df_tfidf.values, clusters)
    plot_silhouette(df_tfidf.values, clusters, n_clusters, filename="silhouette_validation.png")
    plot_elbow_method(df_tfidf.values, filename="elbow_optimization.png")
    '''
    # angular distance
    '''
    km_model, clusters = run_kmeans_angular_tfidf(df_tfidf.values, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)
    X_normalized = normalize(df_tfidf.values, norm='l2')
    score = get_silhouette_validation(X_normalized, clusters)
    plot_silhouette(X_normalized, clusters, n_clusters, filename="silhouette_validation_normalized.png")
    plot_elbow_method(X_normalized, filename="elbow_optimization_normalized.png")'''

    # angular distance on W from NMF give better results
    n_clusters = 6
    km_model, clusters = run_kmeans_on_nmf_angular(W, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)

    # Clustering validation (on normalized W for angular distance)
    W_normalized = normalize(W, norm='l2')
    
    # Global Silhouette Score
    score_global = get_silhouette_validation(W_normalized, clusters)
    print(f"⭐ Mathematical Validation (Global Silhouette Score) : {score_global:.3f}")

    # Detailed scores by cluster
    detailed_scores = get_detailed_silhouette(W, clusters)

    # Visualizations (Using W_normalized for graphical consistency)
    plot_silhouette(W_normalized, clusters, n_clusters, filename="silhouette_validation.png")
    plot_elbow_method(W_normalized, filename="elbow_optimization_nmf.png")

    # 5. TEXT REPORT GENERATION
    with open("results/cluster_report.txt", "w", encoding="utf-8") as f:
        f.write("==========================================\n")
        f.write("===     CSR CLUSTER ANALYSIS REPORT    ===\n")
        f.write("==========================================\n")
        f.write(f"GLOBAL SILHOUETTE SCORE (Angular W): {score_global:.4f}\n")
        f.write("==========================================\n\n")

        # Keywords per cluster
        keywords = get_cluster_keywords(df_tfidf.values, clusters, terms, n_clusters)

        for i in range(n_clusters):
            sub = df_final[df_final["cluster"] == i]
            # Retrieve the specific cluster score (handle Pandas Series or Array)
            try:
                cluster_confidence = detailed_scores[i]
            except:
                cluster_confidence = detailed_scores.iloc[i]

            f.write(f"--- CLUSTER {i} ---\n")
            f.write(f"Confidence Score (Silhouette) : {cluster_confidence:.4f}\n")
            f.write(f"Number of Companies : {len(sub)}\n")
            
            if 'sector' in sub.columns:
                f.write("Top Sectors :\n")
                f.write(sub['sector'].value_counts(normalize=True).head(3).to_string() + "\n")
            
            f.write(f"Dominant Keywords : {', '.join(keywords[i])}\n")
            f.write("\n" + "-"*30 + "\n\n")

    print("✅ Report generated in results/cluster_report.txt")

    # 6. Visualizations and Plots
    plot_heatmap(df_final, 'sector', 'cluster', 
                 "CSR Profiling by Sector", filename="heatmap_secteurs.png")
    
    plot_heatmap(df_final, 'hq_country', 'cluster', 
                 "CSR Profiling by Country", filename="heatmap_pays.png")

if __name__ == "__main__":
    main_text_analysis()