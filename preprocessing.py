# preprocessing.py

import umap
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

@st.cache_data
def generate_embeddings(_model, descriptions):
    embeddings = _model.encode(descriptions.tolist(), show_progress_bar=True)
    return np.array(embeddings)

@st.cache_data
def calculate_returns(prices):
    if prices.empty:
        return pd.DataFrame(), pd.Series(), pd.Series()
    returns = prices.pct_change().dropna()
    avg_returns = returns.mean().fillna(0) * 252
    volatility = returns.std().fillna(0) * np.sqrt(252)
    non_zero_volatility = volatility[volatility > 0].index
    return returns[non_zero_volatility], avg_returns[non_zero_volatility], volatility[non_zero_volatility]

@st.cache_data
def generate_umap_embeddings(embeddings):
    reducer = umap.UMAP(n_components=2)
    return reducer.fit_transform(embeddings)