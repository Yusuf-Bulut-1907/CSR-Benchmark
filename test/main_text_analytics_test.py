from loaders import load_tfidf_data, load_metadata, merge_results
from processing import compute_cosine_similarity, compute_cooccurrence
from models import (
    run_nmf, get_cluster_keywords, 
    get_silhouette_validation, 
    run_kmeans_on_nmf_angular,
    get_detailed_silhouette 
)
from visualizers import plot_heatmap, plot_elbow_method, plot_silhouette
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main_text_analysis():
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Chargement des données
    df_tfidf, terms, companies = load_tfidf_data("data/TFIDF_unigram_bigram_trigram.csv")
    df_uni, terms_uni, _ = load_tfidf_data("data/TFIDF_unigram.csv")
    df_meta = load_metadata("data/metadata.csv")

    # 2. Analyse de base (Cosine & Co-occurrence)
    cosine_sim_df = compute_cosine_similarity(df_tfidf.values, companies)
    cosine_sim_df.to_csv(f"{output_folder}/cosine_similarity_tf_idf.csv")
    
    cooc_df = compute_cooccurrence(df_uni.values, terms_uni)
    cooc_df.to_csv(f"{output_folder}/cooccurrence_matrix_tf_idf.csv")

# 3. Topic Modeling (NMF)
    n_topics = 7 
    W, H, topics = run_nmf(df_tfidf.values, terms, n_components=n_topics)
    print(f"✅ NMF terminée : {n_topics} thèmes identifiés.")

    # --- ÉTAPE CRUCIALE : Créer la variable AVANT de l'utiliser ---
    W_normalized = normalize(W, norm='l2') 

    # 4. Clustering (Validation et choix du K)
    print("Génération de l'Elbow Plot...")
    # Maintenant W_normalized existe bien !
    plot_elbow_method(W_normalized, k_range=range(2, 15), filename="elbow_optimization_nmf.png")
    
    n_clusters = 6 
    km_model, clusters = run_kmeans_on_nmf_angular(W, n_clusters=n_clusters)
    df_final = merge_results(companies, clusters, df_meta)

    # Validation mathématique
    W_normalized = normalize(W, norm='l2')
    score_global = get_silhouette_validation(W_normalized, clusters)
    print(f"⭐ Validation (Silhouette Score) : {score_global:.3f}")

    # --- ANALYSE AVANCÉE POUR LE BENCHMARK ---
    
    # A. Identification des Leaders (Entreprises les plus centrales par cluster)
    distances = km_model.transform(W_normalized)
    cluster_leaders = {}
    for i in range(n_clusters):
        leader_idx = np.argmin(distances[:, i])
        cluster_leaders[i] = companies[leader_idx]

    # B. Profil thématique des clusters (Lien Topics <-> Clusters)
    # On utilise "Topic 0", "Topic 1"... pour matcher exactement les clés du dictionnaire 'topics'
    topic_names = [f"Topic {i}" for i in range(n_topics)]
    df_w = pd.DataFrame(W, columns=topic_names)
    df_w['cluster'] = clusters
    cluster_topic_profile = df_w.groupby('cluster').mean()

    # 5. GÉNÉRATION DU RAPPORT TEXTE
    detailed_scores = get_detailed_silhouette(W, clusters)
    keywords = get_cluster_keywords(df_tfidf.values, clusters, terms, n_clusters)

    report_path = f"{output_folder}/cluster_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================\n")
        f.write("===     CSR CLUSTER ANALYSIS REPORT    ===\n")
        f.write("==========================================\n")
        f.write(f"GLOBAL SILHOUETTE SCORE: {score_global:.4f}\n")
        f.write("==========================================\n\n")

        for i in range(n_clusters):
            sub = df_final[df_final["cluster"] == i]
            
            # Identifier le topic NMF dominant pour ce cluster
            dominant_topic_name = cluster_topic_profile.iloc[i].idxmax()
            topic_words = topics[dominant_topic_name] # Plus de KeyError ici

            f.write(f"--- CLUSTER {i} : Focus sur {dominant_topic_name} ---\n")
            f.write(f"Strategic Leader (Central Co.) : {cluster_leaders[i]}\n")
            f.write(f"Confidence Score : {detailed_scores.iloc[i]:.4f}\n")
            f.write(f"Number of Companies : {len(sub)}\n")
            
            if 'sector' in sub.columns:
                f.write("Top Sectors :\n")
                f.write(sub['sector'].value_counts(normalize=True).head(3).to_string() + "\n")
            
            f.write(f"Dominant Keywords (TF-IDF): {', '.join(keywords[i])}\n")
            f.write(f"Core Topic Concept (NMF): {', '.join(topic_words)}\n")
            f.write("\n" + "-"*30 + "\n\n")

    print(f"✅ Rapport généré : {report_path}")

    # 6. VISUALISATIONS (Gestion des bugs plt.figure)
    
    # A. Silhouette Plot
    plot_silhouette(W_normalized, clusters, n_clusters, filename="silhouette_validation.png")
    
    # B. Heatmap Secteurs
    plot_heatmap(df_final, 'sector', 'cluster', 
                 "Répartition des profils CSR par Secteur", filename="heatmap_secteurs.png")

    # C. Heatmap des Topics (L'importance thématique par cluster)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_topic_profile, annot=True, cmap="Blues", ax=ax)
    ax.set_title("Importance des Topics NMF par Cluster")
    ax.set_ylabel("ID du Cluster")
    ax.set_xlabel("Thématiques identifiées (NMF)")
    plt.savefig(f"{output_folder}/topic_cluster_distribution.png", bbox_inches='tight')
    plt.close(fig) # Libère la mémoire
    print("✅ Heatmap des Topics sauvegardée.")

if __name__ == "__main__":
    main_text_analysis()