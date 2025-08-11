# Forecast Financial Data

This project provides tools and notebooks for fetching, analyzing, and visualizing financial time series data for Tesla (TSLA), S&P 500 ETF (SPY), and Vanguard Total Bond Market ETF (BND) over the past 10 years. The workflow includes data collection, exploratory data analysis (EDA), and risk metric computation.

## Project Structure

```
Forecast_Financial_Data/
│
├── data/
│   ├── TSLA_10years_data.csv
│   ├── SPY_10years_data.csv
│   └── BND_10years_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   └── fetch.py
│
├── TSLA_10yars_data.csv
└── README.md
```

## Features

- **Automated Data Fetching:**
	- `src/fetch.py` uses [yfinance](https://github.com/ranaroussi/yfinance) to download historical data for TSLA, SPY, and BND for the last 10 years and saves them as CSV files in the `data/` directory.

- **Exploratory Data Analysis (EDA):**
	- The `notebooks/EDA.ipynb` notebook provides:
		- Data loading and basic statistics
		- Data type and missing value checks
		- Data normalization
		- Visualization of closing prices and daily percentage changes
		- Rolling mean and standard deviation analysis
		- Outlier detection using Z-scores
		- Stationarity tests (ADF test)
		- Value at Risk (VaR) calculation

## Setup Instructions

1. **Clone the repository:**
	 ```powershell
	 git clone <repo-url>
	 cd Forecast_Financial_Data
	 ```

2. **Install dependencies:**
	 - Required Python packages:
		 - pandas
		 - numpy
		 - matplotlib
		 - yfinance
		 - scikit-learn
		 - scipy
		 - statsmodels
	 - Install with pip:
		 ```powershell
		 pip install pandas numpy matplotlib yfinance scikit-learn scipy statsmodels
		 ```

3. **Fetch the data:**
	 - Run the data fetching script to download the latest 10-year data:
		 ```powershell
		 python src/fetch.py
		 ```
	 - This will create/update the CSV files in the `data/` directory.

4. **Run the EDA notebook:**
	 - Open `notebooks/EDA.ipynb` in Jupyter Notebook or VS Code and run the cells to perform exploratory data analysis and visualize results.

## Usage

- **Data Fetching:**
	- Modify `src/fetch.py` to change ticker symbols or time periods as needed.

- **Analysis:**
	- Use the EDA notebook to explore the data, check for anomalies, and compute risk metrics.

## Notes

- The project is designed for educational and research purposes.
- Data is fetched from Yahoo Finance using the yfinance API.
- Ensure you have a stable internet connection when running the data fetching script.

## License

This project is open source and available under the MIT License.
