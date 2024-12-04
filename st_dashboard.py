# st_dashboard.py

import streamlit as st
from clustering import perform_clustering_on_reduced
from data_fetcher import fetch_data
from portfolio_optimization import portfolio_optimization, create_plotly_efficient_frontier
from visualization import plot_clusters_with_hover_labels
from preprocessing import calculate_returns

def create_streamlit_dashboard(company_desc_df, umap_embeddings):
    st.title("Thematic Investing Basket Creator")

    # Sidebar for dynamic number of clusters
    st.sidebar.header("User Inputs")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)

    # Perform Clustering on fixed UMAP embeddings
    clusters = perform_clustering_on_reduced(umap_embeddings, n_clusters)

    # Add cluster labels to DataFrame
    company_desc_df['Cluster'] = clusters

    # Plot Clusters with Hover Labels
    plot_clusters_with_hover_labels(umap_embeddings, clusters, company_desc_df)

    # Select a cluster for portfolio optimization
    selected_cluster = st.sidebar.selectbox("Select Cluster for Portfolio Optimization", range(n_clusters))
    st.write(f"Selected Cluster: {selected_cluster}")

    # Filter tickers in the selected cluster
    cluster_tickers = company_desc_df[company_desc_df['Cluster'] == selected_cluster]['Ticker'].tolist()

    # Fetch price data for the cluster tickers
    price_data = fetch_data(cluster_tickers)
    returns, avg_returns, volatility = calculate_returns(price_data)

    # Portfolio Optimization
    results, portfolio_weights = portfolio_optimization(avg_returns, returns)

    # Plot Efficient Frontier
    eff_frontier_plot = create_plotly_efficient_frontier(results, avg_returns, portfolio_weights)
    st.plotly_chart(eff_frontier_plot)
