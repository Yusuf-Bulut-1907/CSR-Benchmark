"""
Data Loading and Result Integration Utilities
---------------------------------------------

This module provides lightweight helper functions to:
- Load TF-IDF matrices stored as CSV files
- Load external metadata describing companies
- Merge clustering results with metadata for interpretation

These utilities are typically used after unsupervised learning
(e.g., k-means, hierarchical clustering) to facilitate analysis
and reporting.
"""

import pandas as pd

def load_tfidf_data(path):
    """
    Load a TF-IDF matrix from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the TF-IDF matrix.
        Rows correspond to companies/documents,
        columns correspond to terms.

    Returns
    -------
    df : pandas.DataFrame
        The TF-IDF matrix.
    features : pandas.Index
        Column names representing the vocabulary.
    companies : pandas.Index
        Row index representing company identifiers.
    """
    df = pd.read_csv(path, index_col=0)
    return df, df.columns, df.index

def load_metadata(path): 
    """
    Load company-level metadata from a CSV file.

    The metadata file is expected to use a semicolon (';') as separator,
    which is common in European CSV exports.

    Parameters
    ----------
    path : str
        Path to the metadata CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing descriptive attributes
        (e.g., sector, country, size).
    """
    return pd.read_csv(
        path, 
        sep=";", 
        on_bad_lines='warn', 
        encoding='utf-8'
    )

def merge_results(companies, clusters, df_meta):
    """
    Merge clustering assignments with company metadata.

    This function aligns cluster labels with company identifiers
    and enriches them with external metadata, enabling
    qualitative interpretation of clusters.

    Parameters
    ----------
    companies : array-like
        List or index of company identifiers.
    clusters : array-like
        Cluster labels produced by a clustering algorithm.
    df_meta : pandas.DataFrame
        Metadata table containing a 'company' column.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame associating each company with:
        - its cluster assignment
        - its descriptive metadata
    """
    cluster_results = pd.DataFrame({"company": companies, "cluster": clusters})
    return pd.merge(cluster_results, df_meta, on="company", how="left")