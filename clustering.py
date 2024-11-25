# clustering.py

import pandas as pd
from sklearn.cluster import KMeans

def perform_clustering_on_reduced(umap_embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(umap_embeddings)

def select_representative_stocks(company_desc_df, n_per_cluster=3):
    selected_stocks = []
    for _, group in company_desc_df.groupby('Cluster'):
        selected = group.sample(n=min(n_per_cluster, len(group)), random_state=42)
        selected_stocks.append(selected)
    return pd.concat(selected_stocks)