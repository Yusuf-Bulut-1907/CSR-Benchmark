"""
Similarity and Term-Level Analysis Utilities
--------------------------------------------

This module provides functions to:
- Compute cosine similarity between documents (companies)
- Build term co-occurrence matrices from unigram representations
- Identify globally important terms using mean TF-IDF weights

These analyses support:
- Inter-company similarity assessment
- Lexical structure exploration
- Interpretability of TF-IDF-based models
"""

# ==========================
# COSINE SIMILARITY
# ==========================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize

def compute_cosine_similarity(X, index):
    """
    Compute the cosine similarity matrix between documents.

    Cosine similarity is particularly well-suited for TF-IDF
    representations, as it focuses on vector orientation rather
    than magnitude.

    Parameters
    ----------
    X : array-like or sparse matrix
        Document-term matrix (typically TF-IDF).
    index : list-like
        Document identifiers (e.g. company names).

    Returns
    -------
    pandas.DataFrame
        Square similarity matrix where each value represents
        the cosine similarity between two documents.
    """
    sim = cosine_similarity(X)
    return pd.DataFrame(sim, index=index, columns=index)


# ==========================
# TERM CO-OCCURRENCE MATRIX
# ==========================

def compute_cooccurrence(X_uni, terms_uni):
    """
    Compute a term co-occurrence matrix based on unigram presence.

    The matrix counts how many documents contain each pair of terms,
    ignoring frequency by binarizing the document-term matrix.

    Parameters
    ----------
    X_uni : array-like or sparse matrix
        Unigram document-term matrix.
    terms_uni : list-like
        Vocabulary corresponding to the columns of X_uni.

    Returns
    -------
    pandas.DataFrame
        Symmetric co-occurrence matrix where each cell represents
        the number of documents in which two terms co-occur.
    """
    # Convert counts to binary presence/absence
    X_bin = binarize(X_uni, threshold=0)

    # Term-term co-occurrence via matrix multiplication    
    cooc = X_bin.T @ X_bin

    return pd.DataFrame(cooc, index=terms_uni, columns=terms_uni)

def get_top_terms(X, terms, n_top=30):
    """
    Identify the most globally important terms in the corpus.

    Importance is measured using the mean TF-IDF weight
    across all documents.

    Parameters
    ----------
    X : array-like or sparse matrix
        TF-IDF document-term matrix.
    terms : list-like
        Vocabulary.
    n_top : int
        Number of top terms to return.

    Returns
    -------
    pandas.DataFrame
        Table containing the top terms and their average TF-IDF scores.
    """
    mean_tfidf = X.mean(axis=0)
    top_idx = mean_tfidf.argsort()[-n_top:][::-1]
    return pd.DataFrame({
        "term": terms[top_idx], 
        "mean_tfidf": mean_tfidf[top_idx]
        })