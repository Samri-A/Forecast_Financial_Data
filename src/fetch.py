import yfinance as yf
#import pandas as pd

def fetch(ticker_symbol , periods , path):
    ticker = yf.Ticker(ticker_symbol)
    
    historical_data = ticker.history(period = periods)
    
    historical_data.to_csv(path)

# This will fetch the fiancial data of Tesla of 10 years 
if __name__ == "__main__":
    fetch("TSLA" ,"10y" , "data/TSLA_10years_data.csv" )
    fetch("BND"  ,"10y" , "data/BND_10years_data.csv" )
    fetch("SPY" ,"10y" , "data/SPY_10years_data.csv"  )

