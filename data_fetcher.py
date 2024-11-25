# data_fetcher.py

import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def get_sp500_tickers():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()

@st.cache_data
def fetch_company_names(tickers):
    company_info = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('shortName', ticker)
            company_info.append({'Ticker': ticker, 'CompanyName': company_name})
        except Exception as e:
            print(f"Error fetching name for {ticker}: {e}")
            company_info.append({'Ticker': ticker, 'CompanyName': ticker})
    return pd.DataFrame(company_info)

@st.cache_data
def fetch_data(tickers, start_date='2020-01-01', end_date='2024-01-01'):
    valid_tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.isalnum()]
    if not valid_tickers:
        return pd.DataFrame()

    data = yf.download(valid_tickers, start=start_date, end=end_date)
    prices = data['Adj Close'].dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill')
    valid_columns = prices.columns[prices.notna().sum() > (0.5 * len(prices))]
    return prices[valid_columns]