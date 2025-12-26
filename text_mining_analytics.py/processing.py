import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize

def compute_cosine_similarity(X, index):
    sim = cosine_similarity(X)
    return pd.DataFrame(sim, index=index, columns=index)

def compute_cooccurrence(X_uni, terms_uni):
    X_bin = binarize(X_uni, threshold=0)
    cooc = X_bin.T @ X_bin
    return pd.DataFrame(cooc, index=terms_uni, columns=terms_uni)

def get_top_terms(X, terms, n_top=30):
    mean_tfidf = X.mean(axis=0)
    top_idx = mean_tfidf.argsort()[-n_top:][::-1]
    return pd.DataFrame({"term": terms[top_idx], "mean_tfidf": mean_tfidf[top_idx]})