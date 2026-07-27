from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "IntradayReturn",
    "RangePct",
    "DailyReturn",
    "VolumeChange",
    "Return5",
    "Return20",
    "MA20Gap",
    "Volatility20Annualized",
    "IsST",
]


def build_ml_features(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs and a leakage-safe five-trading-day target."""
    df = daily_prices.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)
    grouped = df.groupby("Ticker", group_keys=False)

    df["IntradayReturn"] = df["Close"] / df["Open"] - 1
    df["RangePct"] = (df["High"] - df["Low"]) / df["Open"]
    df["DailyReturn"] = grouped["Close"].pct_change()
    df["VolumeChange"] = grouped["Volume"].pct_change()
    df["Return5"] = grouped["Close"].pct_change(5)
    df["Return20"] = grouped["Close"].pct_change(20)
    df["MA20"] = grouped["Close"].transform(lambda values: values.rolling(20).mean())
    df["MA52"] = grouped["Close"].transform(lambda values: values.rolling(52).mean())
    df["MA252"] = grouped["Close"].transform(lambda values: values.rolling(252).mean())
    df["MA20Gap"] = df["Close"] / df["MA20"] - 1
    df["Volatility20Annualized"] = (
        grouped["DailyReturn"].transform(lambda values: values.rolling(20).std())
        * np.sqrt(252)
    )
    future_close = grouped["Close"].shift(-5)
    df["Future5Return"] = future_close / df["Close"] - 1
    df["LabelUp5D"] = np.where(
        df["Future5Return"].notna(),
        (df["Future5Return"] > 0).astype(int),
        np.nan,
    )

    board = df["Board"].fillna("").astype(str)
    is_st = df["IsST"].fillna(0).astype(int).eq(1)
    limit_threshold = np.where(board.str.contains("ChiNext"), 0.195, 0.095)
    limit_threshold = np.where(is_st, 0.045, limit_threshold)
    df["LimitUpFlagApprox"] = (df["DailyReturn"] >= limit_threshold).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def training_frame(features: pd.DataFrame) -> pd.DataFrame:
    columns = MODEL_FEATURES + ["LabelUp5D"]
    return features.dropna(subset=columns).copy()
