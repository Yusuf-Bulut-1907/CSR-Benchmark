from loaders import load_tfidf_data, load_metadata, merge_results
from processing import compute_cosine_similarity, compute_cooccurrence, get_top_terms
from models import run_kmeans, run_nmf, get_cluster_keywords, get_silhouette_validation
from visualizers import plot_heatmap, plot_elbow_method, plot_silhouette

def main_text_analysis():
    # 1. data Loading
    df_tfidf, terms, companies = load_tfidf_data("data/TFIDF_unigram_bigram_trigram.csv")
    df_uni, terms_uni, _ = load_tfidf_data("data/TFIDF_unigram.csv")
    df_meta = load_metadata("data/metadata.csv")

    # 2. Basic Text Analysis
    cosine_sim_df = compute_cosine_similarity(df_tfidf.values, companies)
    cosine_sim_df.to_csv("results/cosine_similarity.csv")
    
    cooc_df = compute_cooccurrence(df_uni.values, terms_uni)
    cooc_df.to_csv("results/cooccurrence_matrix.csv")

    # 3. Topic Modeling (NMF)
    W, H, topics = run_nmf(df_tfidf.values, terms)
    for t, words in topics.items():
        print(f"{t}: {', '.join(words)}")

# 4. Clustering
    n_clusters = 5
    km_model, clusters = run_kmeans(df_tfidf.values, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)

    # --- Clustering validation ---

    # Silhouette Score
    score = get_silhouette_validation(df_tfidf.values, clusters)
    print(f"⭐ Validation mathématique (Silhouette Score) : {score:.3f}")

    # Graphical Silhouette Plot
    plot_silhouette(df_tfidf.values, clusters, n_clusters, filename="silhouette_validation.png")

    # Elbow Method Plot
    plot_elbow_method(df_tfidf.values, filename="elbow_optimization.png")
    
    # 5. TEXT REPORT GENERATION
    with open("results/cluster_report.txt", "w", encoding="utf-8") as f:
        f.write("=== CSR CLUSTER ANALYSIS REPORT ===\n\n")

        for i in range(n_clusters):
            sub = df_final[df_final["cluster"] == i]
            f.write(f"--- CLUSTER {i} ---\n")
            f.write(f"Number of Companies : {len(sub)}\n")
            
            if 'sector' in sub.columns:
                f.write("Top Sectors :\n")
                f.write(sub['sector'].value_counts(normalize=True).head(3).to_string() + "\n")
            
            # Ajout des mots-clés du cluster (venant de models.py)
            keywords = get_cluster_keywords(df_tfidf.values, clusters, terms, n_clusters)
            f.write(f"Dominant Keywords : {', '.join(keywords[i])}\n")
            f.write("\n" + "="*30 + "\n\n")
            
    print("✅ Report generated in results/cluster_report.txt")

    # 6. Visualizations and Plots
    plot_heatmap(df_final, 'sector', 'cluster', 
                 "CSR Profiling by Sector", filename="heatmap_secteurs.png")
    
    plot_heatmap(df_final, 'hq_country', 'cluster', 
                 "CSR Profiling by Country", filename="heatmap_pays.png")



if __name__ == "__main__":
    main_text_analysis()