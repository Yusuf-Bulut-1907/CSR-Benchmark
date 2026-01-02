import pandas as pd

# Chargement du fichier CSV
# Note : assurez-vous que le séparateur (sep) correspond à votre fichier (',' ou ';')
df = pd.read_csv('data/TFIDF_unigram_bigram_trigram.csv')

# Affichage de la forme (lignes, colonnes)
print(f"Shape de la matrice : {df.shape}")
print(f"Nombre de documents : {df.shape[0]}")
print(f"Nombre de mots (vocabulaire) : {df.shape[1]}")

# Aperçu des premières lignes
print(df.head())

import numpy as np

# 1. Calcul de la parcimonie (Sparsity)
non_zero_count = np.count_nonzero(df.values)
total_elements = df.size
sparsity = (1 - non_zero_count / total_elements) * 100
print(f"Parcimonie : {sparsity:.2f}% (pourcentage de zéros)")

# 2. Top 10 des mots les plus importants (moyenne TF-IDF)
top_words = df.mean().sort_values(ascending=False).head(10)
print("\nTop 10 des mots (score moyen le plus haut) :")
print(top_words)

# 3. Vérification des valeurs manquantes (NaN)
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")