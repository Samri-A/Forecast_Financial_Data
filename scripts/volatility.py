import numpy as np

def get_volatility(cov_matrix, weights) -> float:
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    portfolio_vol *= np.sqrt(252)

    return portfolio_vol
