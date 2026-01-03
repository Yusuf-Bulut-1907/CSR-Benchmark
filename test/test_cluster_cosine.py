import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# --- 1. CHARGEMENT ---
XPATH = "data/TFIDF_unigram_bigram_trigram.csv"
XDF = pd.read_csv(XPATH, index_col=0)
X = XDF.values 
n_components = 6

# --- 2. RÉDUCTION PAR NMF (7 TOPICS) ---
# La NMF est excellente pour le benchmark RSE car elle identifie des thématiques claires
nmf = NMF(n_components= n_components, init='nndsvd', random_state=42)
W = nmf.fit_transform(X) # W contient le score de chaque entreprise pour les 7 thèmes

# --- 3. NORMALISATION ET CLUSTERING ---
# On normalise les scores NMF pour que le clustering soit équilibré
W_norm = normalize(W, norm='l2')

n_clusters = 6 # Ton réglage optimal
kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
labels = kmeans.fit_predict(W_norm)

print(f"Silhouette Score : {silhouette_score(W_norm, labels):.4f}")

# --- 4. ANALYSE DES CLUSTERS (Le bloc que tu as demandé) ---
# Ce bloc permet de "donner un nom" à tes clusters selon les thèmes CSR dominants
df_analysis = pd.DataFrame(W, columns=[f'Topic_{i}' for i in range(n_components)], index=XDF.index)
df_analysis['Cluster'] = labels

# Profil moyen : quel sujet définit quel groupe d'entreprises ?
cluster_profiles = df_analysis.groupby('Cluster').mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles, annot=True, cmap='Blues')
plt.title("Importance des Topics par Cluster (Validation Métier)")
plt.ylabel("Numéro du Cluster")
plt.xlabel("Thématiques (Topics NMF)")
plt.show()

# --- 5. EXPORT FINAL ---
# On ajoute le cluster à ton DataFrame original pour ton benchmark
XDF['Cluster_Final'] = labels
XDF[['Cluster_Final']].to_csv("benchmark_csr_final.csv")