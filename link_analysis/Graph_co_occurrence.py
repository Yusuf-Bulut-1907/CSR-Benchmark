import pandas as pd
import numpy as np
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
TFIDF_PATH = "data/TFIDF_unigram_bigram_trigram.csv" # Utilise ton fichier TF-IDF global
OUTPUT_DIR = "gephi_concepts"
# On limite aux 100 concepts les plus importants pour la clarté du graphe
TOP_N_CONCEPTS = 100 
# Seuil de co-occurrence : combien d'entreprises doivent partager les deux mots pour créer un lien
LINK_MIN_STRENGTH = 2 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# 1. PRÉPARATION
# =====================
df_tfidf = pd.read_csv(TFIDF_PATH, index_col=0)

# On ne garde que les colonnes les plus riches en information
top_cols = df_tfidf.sum().sort_values(ascending=False).head(TOP_N_CONCEPTS).index
df_top = df_tfidf[top_cols]

# Transformer en matrice binaire (1 si le mot est présent, 0 sinon)
# On considère qu'un mot est présent si son TF-IDF > 0.05 (pour éviter le bruit)
df_binary = (df_top > 0.05).astype(int)

# =====================
# 2. CALCUL DE CO-OCCURRENCE
# =====================
# Produit matriciel : calcule le nombre de documents communs pour chaque paire de mots
co_matrix = df_binary.T.dot(df_binary)

# =====================
# 3. GÉNÉRATION DES EDGES (LIENS)
# =====================
edges = []
concepts = co_matrix.columns

for i in range(len(concepts)):
    for j in range(i + 1, len(concepts)):
        weight = co_matrix.iloc[i, j]
        if weight >= LINK_MIN_STRENGTH:
            edges.append({
                "Source": concepts[i],
                "Target": concepts[j],
                "Weight": weight
            })

df_edges = pd.DataFrame(edges)
df_edges.to_csv(os.path.join(OUTPUT_DIR, "concept_edges.csv"), index=False)

# =====================
# 4. GÉNÉRATION DES NODES (POINTS)
# =====================
# La taille du nœud dépendra de sa fréquence totale dans le corpus
node_importance = df_binary.sum()
nodes = []
for concept in concepts:
    nodes.append({
        "Id": concept,
        "Label": concept,
        "Size": node_importance[concept]
    })

df_nodes = pd.DataFrame(nodes)
df_nodes.to_csv(os.path.join(OUTPUT_DIR, "concept_nodes.csv"), index=False)

print(f"✅ Graphe de co-occurrence généré : {len(nodes)} concepts et {len(edges)} liens.")