import pandas as pd
import numpy as np

def getSharpe(returns , annual_rf = 0.01):         
    daily_rf  = annual_rf / 252
    excess_returns = returns - daily_rf 
    mean_excess = excess_returns.mean()       
    std_excess  = excess_returns.std(ddof=0)  
    
    daily_sharpe = mean_excess / std_excess
    annual_sharpe = np.sqrt(252) * daily_sharpe
    return annual_sharpe