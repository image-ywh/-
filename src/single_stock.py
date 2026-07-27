from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    paths: np.ndarray
    terminal_prices: np.ndarray
    current_price: float
    mu_annualized: float
    sigma_annualized: float
    lower_1pct_price: float
    expected_terminal_price: float
    upside_probability: float

    @property
    def var_amount_99(self) -> float:
        return max(0.0, self.current_price - self.lower_1pct_price)

    @property
    def var_pct_99(self) -> float:
        return self.var_amount_99 / self.current_price if self.current_price else np.nan


def prepare_single_stock(daily_prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = daily_prices.loc[daily_prices["Ticker"].eq(ticker)].copy()
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    df = df.sort_values("Date").reset_index(drop=True)
    df["DailyReturn"] = df["Close"].pct_change()
    for window in (20, 52, 252):
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
    df["LogReturn"] = np.log(df["Close"]).diff()
    df["Drawdown"] = df["Close"] / df["Close"].cummax() - 1
    return df


def summarize_single_stock(df: pd.DataFrame) -> dict[str, float | str]:
    returns = df["DailyReturn"].dropna()
    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    total_return = last_close / first_close - 1
    annualized_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
    return {
        "ticker": str(df["Ticker"].iloc[0]),
        "stock_name": str(df["StockNameCN"].iloc[0]),
        "start_date": str(df["Date"].min().date()),
        "end_date": str(df["Date"].max().date()),
        "latest_close": last_close,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(returns.std() * np.sqrt(252)),
        "max_drawdown": float(df["Drawdown"].min()),
        "mean_daily_return": float(returns.mean()),
        "latest_volume": float(df["Volume"].iloc[-1]),
    }


def simulate_gbm(
    df: pd.DataFrame,
    horizon_days: int = 252,
    n_paths: int = 5000,
    seed: int = 42,
) -> MonteCarloResult:
    returns = df["LogReturn"].dropna()
    mu_annualized = float(returns.mean() * 252)
    sigma_annualized = float(returns.std() * np.sqrt(252))
    dt = 1 / 252
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, horizon_days))
    increments = (
        (mu_annualized - 0.5 * sigma_annualized**2) * dt
        + sigma_annualized * np.sqrt(dt) * shocks
    )
    paths = np.empty((n_paths, horizon_days + 1), dtype=float)
    paths[:, 0] = float(df["Close"].iloc[-1])
    paths[:, 1:] = paths[:, [0]] * np.exp(np.cumsum(increments, axis=1))
    terminal = paths[:, -1]
    lower_1pct = float(np.quantile(terminal, 0.01))
    return MonteCarloResult(
        paths=paths,
        terminal_prices=terminal,
        current_price=float(df["Close"].iloc[-1]),
        mu_annualized=mu_annualized,
        sigma_annualized=sigma_annualized,
        lower_1pct_price=lower_1pct,
        expected_terminal_price=float(terminal.mean()),
        upside_probability=float((terminal > paths[:, 0]).mean()),
    )

