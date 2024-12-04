import numpy as np
import pandas as pd
import plotly.express as px

def portfolio_optimization(avg_returns, returns, risk_free_rate=0.02, n_portfolios=50):
    if returns.empty or avg_returns.empty:
        return np.array([]), []

    cov_matrix = returns.cov()
    n = len(avg_returns)
    results = np.zeros((n_portfolios, 3))  # Columns for [Return, Volatility, Sharpe Ratio]
    portfolio_weights = []  # To store weights for each portfolio
    epsilon = 1e-6

    for i in range(n_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        port_return = np.dot(weights, avg_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / (port_volatility + epsilon)
        results[i] = [port_return, port_volatility, sharpe_ratio]

    return results, portfolio_weights


def create_plotly_efficient_frontier(results, avg_returns, portfolio_weights):
    # Extract data
    returns_vals = results[:, 0]
    volatilities = results[:, 1]
    sharpe_ratios = results[:, 2]

    # Prepare data for hover: Top 5 stocks in each portfolio
    hover_texts = []
    for weights in portfolio_weights:
        weighted_returns = pd.Series(weights * avg_returns.values, index=avg_returns.index)
        top_stocks = weighted_returns.nlargest(5).index.tolist()
        hover_texts.append(f": {', '.join(top_stocks)}")

    # Create DataFrame for Plotly
    frontier_df = pd.DataFrame({
        "Portfolio Return": returns_vals,
        "Portfolio Volatility": volatilities,
        "Sharpe Ratio": sharpe_ratios,
        "Top 5 Stocks": hover_texts
    })

    # Plot using Plotly
    fig = px.scatter(
        frontier_df,
        x="Portfolio Volatility",
        y="Portfolio Return",
        color="Sharpe Ratio",
        hover_data={"Sharpe Ratio": True, "Portfolio Volatility": True, "Portfolio Return": True, "Top 5 Stocks": True},
        title="Efficient Frontier of Selected Cluster Portfolio",
        labels={"Portfolio Volatility": "Volatility (Risk)", "Portfolio Return": "Expected Return"},
        color_continuous_scale=px.colors.sequential.Viridis
    )

    return fig
