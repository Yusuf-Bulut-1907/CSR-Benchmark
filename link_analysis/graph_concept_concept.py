import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

# Config
TFIDF_PATH = "data/TFIDF_RSE_unigram_bigram.csv"
OUTPUT_DIR = "gephi_graph"
COOC_THRESHOLD = 0.5  # Augmenté légèrement pour éviter un fichier Gephi trop lourd

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Chargement efficace
print("Chargement des données...")
df = pd.read_csv(TFIDF_PATH, index_col=0)
concepts = df.columns.tolist()

# 2. Conversion en matrice creuse (Sparse)
# Cela libère énormément de RAM
M = csr_matrix(df.values)

# 3. Calcul de co-occurrence via produit matriciel sparse
print("Calcul de la matrice de co-occurrence (Sparse)...")
cooc = (M.T @ M)
cooc.setdiag(0) # On enlève les auto-boucles
cooc.eliminate_zeros() # On retire les zéros pour gagner de la place

# 4. Extraction des liens (sans créer de DataFrame géant)
print("Extraction des arêtes...")
cooc_coo = cooc.tocoo() # Conversion au format Coordinate pour itérer

edges = []
for i, j, v in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):
    # Pour éviter les doublons (A-B et B-A) dans un graphe non-orienté
    if i < j and v >= COOC_THRESHOLD:
        edges.append({
            "Source": concepts[i],
            "Target": concepts[j],
            "Weight": round(float(v), 4),
            "Type": "Undirected"
        })

# 5. Sauvegarde
print(f"Création du CSV avec {len(edges)} arêtes...")
edges_df = pd.DataFrame(edges)
edges_df.to_csv(f"{OUTPUT_DIR}/edges_concept_concept.csv", index=False)

nodes_df = pd.DataFrame({"Id": concepts, "Label": concepts, "Type": "concept"})
nodes_df.to_csv(f"{OUTPUT_DIR}/nodes_concept_concept.csv", index=False)

print("✅ Terminé !")