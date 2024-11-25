# visualization.py

import pandas as pd
import plotly.express as px
import streamlit as st

def plot_clusters_with_hover_labels(umap_embeddings, clusters, company_desc_df):
    plot_df = pd.DataFrame({
        'UMAP_1': umap_embeddings[:, 0],
        'UMAP_2': umap_embeddings[:, 1],
        'Cluster': clusters,
        'Ticker': company_desc_df['Ticker'],
        'Name': company_desc_df['CompanyName']
    })
    fig = px.scatter(
        plot_df,
        x='UMAP_1', y='UMAP_2', color='Cluster',
        hover_name='Ticker', hover_data=['Name'],
        title="Thematic Clusters"
    )
    st.plotly_chart(fig)