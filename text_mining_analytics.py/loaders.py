import pandas as pd

def load_tfidf_data(path):
    df = pd.read_csv(path, index_col=0)
    return df, df.columns, df.index

def load_metadata(path):
    return pd.read_csv(
        path, 
        sep=";", 
        on_bad_lines='warn', 
        encoding='utf-8'
    )

def merge_results(companies, clusters, df_meta):
    cluster_results = pd.DataFrame({"company": companies, "cluster": clusters})
    return pd.merge(cluster_results, df_meta, on="company", how="left")