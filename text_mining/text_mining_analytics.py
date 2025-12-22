import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import binarize
from sklearn.preprocessing import normalize

# =====================
# LOAD TF-IDF MATRIX
# =====================

tfidf_df = pd.read_csv("/Users/matteogalizia/Library/CloudStorage/OneDrive-UCL/Maste 1/Web mining/Web_mining_project/data/TFIDF_unigram_bigram_trigram.csv", index_col=0)

X_tfidf = tfidf_df.values
terms = tfidf_df.columns
companies = tfidf_df.index

unitfidf = pd.read_csv("/Users/matteogalizia/Library/CloudStorage/OneDrive-UCL/Maste 1/Web mining/Web_mining_project/data/TFIDF_unigram.csv", index_col=0)
X_uni_tfidf = unitfidf.values
terms_uni = unitfidf.columns
companies_uni = unitfidf.index

# =====================
# COSINE SIMILARITY
# =====================
cosine_sim = cosine_similarity(X_tfidf)
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=companies,
    columns=companies
)
cosine_sim_df.to_csv("results/cosine_similarity.csv")

# =====================
# TOP GLOBAL TERMS
# =====================
mean_tfidf = X_tfidf.mean(axis=0)
top_idx = np.argsort(mean_tfidf)[-30:][::-1]

top_terms = pd.DataFrame({
    "term": terms[top_idx],
    "mean_tfidf": mean_tfidf[top_idx]
})

print(top_terms)

# =====================
# COOCCURRENCE MATRIX 
# =====================

X_uni_bin = binarize(X_uni_tfidf, threshold=0)

cooccurrence = X_uni_bin.T @ X_uni_bin

cooc_df = pd.DataFrame(
    cooccurrence,
    index=terms_uni,
    columns=terms_uni
)

cooc_df.to_csv("results/cooccurrence_matrix.csv")

# =====================
# TOP COOCCURRING TERMS
# =====================
def get_top_cooccurring(term, top_n=10):
    if term not in cooc_df.index:
        print(f"Le terme '{term}' n'est pas dans le vocabulaire.")
        return None
    
    cooc_series = cooc_df.loc[term]
    top_coocs = cooc_series.sort_values(ascending=False).head(top_n + 1).iloc[1:]
    
    return pd.DataFrame({
        "term": top_coocs.index,
        "cooccurrence_count": top_coocs.values
    })

# =====================
# TOPIC MODELING (NMF)
# =====================
nmf = NMF(n_components=8, random_state=42)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_
for i, topic in enumerate(H):
    print(f"\nTopic {i}:")
    top_terms_idx = topic.argsort()[-10:][::-1]
    print(", ".join(terms[top_terms_idx]))

# =====================
# CLUSTERING (K-MEANS)
# =====================
kmeans = KMeans(n_clusters=6, random_state=42, n_init=20)
X_scaled = normalize(X_tfidf) #Normalize data, important for cosine-based clustering
clusters = kmeans.fit_predict(X_scaled)

cluster_df = pd.DataFrame({
    "company": companies,
    "cluster": clusters
})

cluster_df.to_csv("results/clusters.csv", index=False)
# faire une analyse des clusters avec des métadonnées externes ?