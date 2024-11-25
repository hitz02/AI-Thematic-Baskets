# Thematic Investing Basket Creator

A Streamlit web application that helps create themed investment portfolios using AI and NLP.

## Features

- Groups S&P 500 companies into themes using AI-powered text analysis
- Interactive visualization of company clusters
- Portfolio optimization for selected company groups
- Real-time stock data integration
- Efficient frontier visualization for risk-return analysis

## Requirements
```
streamlit
pandas
numpy
yfinance
plotly
scikit-learn
sentence-transformers
umap-learn
matplotlib
```

## Installation

1. Download the repository/folder

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Set up your OpenRouter API key as an environment variable (`Optional`):
    ```
    export OPENROUTER_API_KEY='your-api-key'
    ```

## Usage

1. Generate company descriptions (one-time setup):

    `File already provided, this step should be skipped unless you have Openrouter API key from step 3 above`

    ```
    python get_descriptions.py
    ```

2. Run the Streamlit app:
    ```
    streamlit run main.py
    ```
