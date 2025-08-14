import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import timedelta


class PortfolioBacktester:
    def __init__(self, price_data: pd.DataFrame, strategy_tickers, benchmark_tickers=["SPY", "BND"],
                 strategy_kind="max_sharpe", rebalance="monthly", lookback_days=256, risk_free=0.0):
        """
        price_data: DataFrame with datetime index and columns = tickers'  Close prices
        strategy_tickers: list of tickers in the strategy
        benchmark_tickers: list of tickers for benchmark (default SPY/BND)
        strategy_kind: 'max_sharpe' or 'min_vol'
        rebalance: 'buy_and_hold' or 'monthly'
        lookback_days: number of trading days for mu/cov estimation
        risk_free: annual risk-free rate
        """
        self.prices = price_data.copy()
        self.strategy_tickers = strategy_tickers
        self.benchmark_tickers = benchmark_tickers
        self.strategy_kind = strategy_kind
        self.rebalance = rebalance
        self.lookback_days = lookback_days
        self.risk_free = risk_free

        self.rets = self.prices.pct_change().dropna()
        self.bt_start = self.rets.index[-min(252, len(self.rets))]
        self.bt_end = self.rets.index[-1]

    
    @staticmethod
    def annualize_mean_cov(returns_df):
        mu = returns_df.mean() * 252
        cov = returns_df.cov() * 252
        return mu, cov

    @staticmethod
    def portfolio_perf(weights, mu, cov, rf=0.0):
        port_ret = float(np.dot(weights, mu))
        port_vol = float(np.sqrt(weights @ cov.values @ weights))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
        return port_ret, port_vol, sharpe

    @staticmethod
    def _min_vol_obj(w, cov):
        return np.sqrt(w @ cov.values @ w)

    @staticmethod
    def _neg_sharpe_obj(w, mu, cov, rf):
        ret, vol, _ = PortfolioBacktester.portfolio_perf(w, mu, cov, rf)
        return -((ret - rf) / vol) if vol > 0 else 1e9

    def optimize_weights(self, mu, cov):
        n = len(mu)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        if self.strategy_kind == "min_vol":
            res = minimize(self._min_vol_obj, x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
        else:
            res = minimize(self._neg_sharpe_obj, x0, args=(mu, cov, self.risk_free), method="SLSQP", bounds=bounds,
                           constraints=cons)
        if not res.success:
            return x0
        w = np.maximum(res.x, 0)
        return w / w.sum()

    @staticmethod
    def month_starts(index):
        df = pd.DataFrame(index=index)
        ms = df.groupby([index.year, index.month]).head(1).index
        return list(ms)

    @staticmethod
    def apply_weights_over_period(period_rets, weights, cols):
        w = pd.Series(weights, index=cols)
        return (period_rets[cols] * w).sum(axis=1)

    
    def backtest_strategy(self):
        rets_bt = self.rets.loc[self.bt_start:self.bt_end, self.strategy_tickers].dropna()
        wealth = pd.Series(index=rets_bt.index, dtype=float)
        self.weights_log = []

        if self.rebalance == "buy_and_hold":
            lookback_start_idx = self.rets.index.get_loc(rets_bt.index[0])
            lb_start = max(0, lookback_start_idx - self.lookback_days)
            est_window = self.rets.iloc[lb_start:lookback_start_idx][self.strategy_tickers]
            mu, cov = self.annualize_mean_cov(est_window)
            w = self.optimize_weights(mu, cov)
            self.weights_log.append((rets_bt.index[0], dict(zip(self.strategy_tickers, w))))
            wealth[:] = self.apply_weights_over_period(rets_bt, w, self.strategy_tickers)

        else:
            starts = self.month_starts(rets_bt.index)
            if starts[0] != rets_bt.index[0]:
                starts = [rets_bt.index[0]] + starts
            boundaries = starts + [rets_bt.index[-1] + timedelta(days=1)]

            for i in range(len(boundaries) - 1):
                seg_start = boundaries[i]
                seg_end = boundaries[i + 1] - timedelta(days=1)
                seg_idx = rets_bt.loc[seg_start:seg_end].index
                if len(seg_idx) == 0:
                    continue
                global_idx_start = self.rets.index.get_loc(seg_idx[0])
                lb_start = max(0, global_idx_start - self.lookback_days)
                est_window = self.rets.iloc[lb_start:global_idx_start][self.strategy_tickers]
                mu, cov = self.annualize_mean_cov(est_window)
                w = self.optimize_weights(mu, cov)
                self.weights_log.append((seg_idx[0], dict(zip(self.strategy_tickers, w))))
                wealth.loc[seg_idx] = self.apply_weights_over_period(rets_bt.loc[seg_idx], w, self.strategy_tickers)

        return wealth.dropna()

    def backtest_benchmark(self):
        cols = [c for c in self.benchmark_tickers if c in self.rets.columns]
        rets_bt_bm = self.rets.loc[self.bt_start:self.bt_end, cols].dropna()
        w_target = np.array([0.6, 0.4])

        if self.rebalance == "buy_and_hold":
            bm = (rets_bt_bm * w_target).sum(axis=1)
        else:
            starts = self.month_starts(rets_bt_bm.index)
            if starts[0] != rets_bt_bm.index[0]:
                starts = [rets_bt_bm.index[0]] + starts
            boundaries = starts + [rets_bt_bm.index[-1] + timedelta(days=1)]
            bm_parts = []
            for i in range(len(boundaries) - 1):
                seg = rets_bt_bm.loc[boundaries[i]:boundaries[i + 1] - timedelta(days=1)]
                bm_parts.append((seg * w_target).sum(axis=1))
            bm = pd.concat(bm_parts).sort_index()

        return bm

    def metrics(self, ret_series):
        rf_daily = (1 + self.risk_free) ** (1 / 252) - 1
        excess = ret_series - rf_daily
        ann_ret = (1 + ret_series).prod() ** (252 / len(ret_series)) - 1
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = (excess.mean() * 252) / (ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else np.nan
        wealth = (1 + ret_series).cumprod()
        dd = wealth / wealth.cummax() - 1
        max_dd = dd.min()
        return {
            "Annual Return": ann_ret,
            "Annual Volatility": ann_vol,
            "Sharpe": sharpe,
            "Cumulative Return": wealth.iloc[-1] - 1,
            "Max Drawdown": max_dd
        }
