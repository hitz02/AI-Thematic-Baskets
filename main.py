# main.py

import streamlit as st
import pandas as pd
from data_fetcher import fetch_company_names
from preprocessing import load_embedding_model, generate_embeddings, generate_umap_embeddings
from st_dashboard import create_streamlit_dashboard

def main():

    # Load pre-trained embedding model
    model = load_embedding_model()

    # Load company description data
    company_desc_df = pd.read_csv("snp500_company_description.csv")

    # Fetch company names using tickers and merge with descriptions DataFrame
    tickers = company_desc_df['Ticker'].tolist()
    company_names_df = fetch_company_names(tickers)
    company_desc_df = pd.merge(company_desc_df, company_names_df, on='Ticker', how='left')

    # Generate embeddings for company descriptions if not already cached
    if 'embeddings' not in st.session_state:
        descriptions = company_desc_df['company_description']
        st.session_state['embeddings'] = generate_embeddings(model, descriptions)
    
    embeddings = st.session_state['embeddings']

    # Generate UMAP embeddings for visualization
    if 'umap_embeddings' not in st.session_state:
        st.session_state['umap_embeddings'] = generate_umap_embeddings(embeddings)
    
    umap_embeddings = st.session_state['umap_embeddings']

    # Call the Streamlit dashboard function
    create_streamlit_dashboard(company_desc_df, umap_embeddings)

if __name__ == "__main__":
    main()