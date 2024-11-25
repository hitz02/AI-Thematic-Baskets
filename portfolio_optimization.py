# portfolio_optimization.py

import numpy as np
import matplotlib.pyplot as plt

def portfolio_optimization(avg_returns, returns, risk_free_rate=0.02):
    if returns.empty or avg_returns.empty:
        return np.array([])

    cov_matrix = returns.cov()
    n = len(avg_returns)
    n_portfolios = 5000
    results = np.zeros((n_portfolios, 3))
    epsilon = 1e-6

    for i in range(n_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_return = np.dot(weights, avg_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / (port_volatility + epsilon)
        results[i] = [port_return, port_volatility, sharpe_ratio]

    return results

def plot_efficient_frontier(results, avg_returns, volatility):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='YlGnBu', marker='o', alpha=0.6)
    plt.colorbar(scatter, label='Sharpe Ratio')
    for stock in avg_returns.index:
        plt.scatter(volatility[stock], avg_returns[stock], color='red', marker='*', s=100)
        plt.text(volatility[stock], avg_returns[stock], stock, fontsize=9, ha='right')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier with Individual Stock Labels')
    plt.grid(True)
    return plt