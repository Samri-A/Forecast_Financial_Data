# Forecast Financial Data

This project provides a comprehensive pipeline for fetching, analyzing, forecasting, and backtesting financial time series data for major assets (Tesla, S&P 500 ETF, and Vanguard Total Bond Market ETF). It includes data collection, exploratory data analysis (EDA), risk/return metric computation, time series forecasting (ARIMA, LSTM), and portfolio backtesting.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Scripts Overview](#scripts-overview)
- [Notebooks Overview](#notebooks-overview)
- [Data](#data)
- [License](#license)

---

## Project Structure

```
Forecast_Financial_Data/
│
├── data/                # Raw and processed financial data (CSV)
│   ├── TSLA_10years_data.csv
│   ├── SPY_10years_data.csv
│   ├── BND_10years_data.csv
│   └── predicted/
│
├── model/               # Saved models (e.g., LSTM)
│   └── lstm_model.pkl
│
├── notebooks/           # Jupyter notebooks for EDA, forecasting, backtesting
│   ├── EDA.ipynb
│   ├── ARIMA.ipynb
│   ├── lstm.ipynb
│   ├── forcast.ipynb
│   ├── Backtest.ipynb
│   └── task4.ipynb
│
├── scripts/             # Python scripts for core logic
│   ├── fetch.py
│   ├── back_test.py
│   ├── forcast.py
│   ├── get_sharpe.py
│   └── volatility.py
│
├── src/
│   └── fetch.py         # Data fetching utility
│
├── requirement.txt      # Python dependencies
└── README.md
```

---

## Features

- **Automated Data Fetching:**  
	Fetches historical price data for TSLA, SPY, and BND using Yahoo Finance (`yfinance`) and saves as CSV.

- **Exploratory Data Analysis (EDA):**  
	Notebooks for data loading, normalization, visualization, outlier detection, and stationarity testing.

- **Time Series Forecasting:**  
	- **ARIMA:** Traditional statistical forecasting.
	- **LSTM:** Deep learning-based forecasting using a pre-trained model.

- **Portfolio Backtesting:**  
	- Customizable backtesting engine for portfolio strategies (max Sharpe, min volatility, buy-and-hold, monthly rebalance).
	- Performance metrics: annual return, volatility, Sharpe ratio, drawdown, etc.

- **Risk Metrics:**  
	- Sharpe ratio calculation.
	- Portfolio volatility computation.

---

## Setup Instructions

1. **Clone the repository:**
	 ```powershell
	 git clone <repo-url>
	 cd Forecast_Financial_Data
	 ```

2. **Install dependencies:**
	 - Install required packages:
		 ```powershell
		 pip install -r requirement.txt
		 pip install yfinance scikit-learn scipy statsmodels tensorflow backtrader
		 ```
	 - (Some dependencies are listed in `requirement.txt`, others are used in notebooks/scripts.)

3. **Fetch the data:**
	 - Run the data fetching script:
		 ```powershell
		 python src/fetch.py
		 ```
	 - This will create/update the CSV files in the `data/` directory.

4. **Run the notebooks:**
	 - Open any notebook in the `notebooks/` folder using Jupyter or VS Code and run the cells.

---

## Usage Guide

- **Data Fetching:**  
	Edit `src/fetch.py` to change ticker symbols or periods as needed. Run to update data.

- **EDA:**  
	Use `notebooks/EDA.ipynb` to explore and visualize the data, check for anomalies, and perform statistical tests.

- **Forecasting:**  
	- `notebooks/ARIMA.ipynb` for ARIMA-based forecasting.
	- `notebooks/lstm.ipynb` for LSTM model training.
	- `notebooks/forcast.ipynb` for using the trained LSTM model to predict future prices.

- **Backtesting:**  
	- `notebooks/Backtest.ipynb` demonstrates portfolio backtesting using the custom engine in `scripts/back_test.py`.

---

## Scripts Overview

- **src/fetch.py:**  
	Fetches historical data for specified tickers and periods using yfinance.

- **scripts/back_test.py:**  
	Contains the `PortfolioBacktester` class for simulating portfolio strategies, rebalancing, and calculating performance metrics.

- **scripts/forcast.py:**  
	Loads a pre-trained LSTM model and predicts future prices for a given asset.

- **scripts/get_sharpe.py:**  
	Computes the annualized Sharpe ratio for a return series.

- **scripts/volatility.py:**  
	Computes annualized portfolio volatility given a covariance matrix and weights.

---

## Notebooks Overview

- **EDA.ipynb:**  
	Data loading, normalization, visualization, outlier detection, and stationarity testing.

- **ARIMA.ipynb:**  
	ARIMA modeling and forecasting for time series data.

- **lstm.ipynb:**  
	LSTM model training for time series forecasting.

- **forcast.ipynb:**  
	Uses the trained LSTM model to forecast future prices.

- **Backtest.ipynb:**  
	Demonstrates portfolio backtesting and performance evaluation.

---

## Data

- All raw and processed data is stored in the `data/` directory.
- Data is fetched from Yahoo Finance using the yfinance API.

---

## License

This project is open source and available under the MIT License.
