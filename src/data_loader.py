from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "A_share_quant_project_dataset.xlsx"
)


@dataclass
class DatasetBundle:
    stock_info: pd.DataFrame
    daily_prices: pd.DataFrame
    single_sheet: pd.DataFrame
    provided_ml_features: pd.DataFrame
    data_dictionary: pd.DataFrame
    sources: pd.DataFrame
    quality: Dict[str, int]


def _read_sheet(path: Path, sheet_name: str, required_column: str) -> pd.DataFrame:
    """Read a sheet whose title rows are above the actual header row."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row: Optional[int] = None
    for index, row in raw.iterrows():
        values = {str(value).strip() for value in row.dropna().tolist()}
        if required_column in values:
            header_row = int(index)
            break
    if header_row is None:
        raise ValueError(
            f"Could not locate header containing {required_column!r} in {sheet_name!r}"
        )
    frame = raw.iloc[header_row + 1 :].copy()
    frame.columns = [str(value).strip() for value in raw.iloc[header_row].tolist()]
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def clean_daily_prices(
    daily_prices: pd.DataFrame,
    stock_info: pd.DataFrame,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize dates/numbers and remove duplicate or logically invalid OHLC rows."""
    df = daily_prices.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    _coerce_numeric(df, ["Open", "High", "Low", "Close", "AdjClose", "Volume"])

    missing_date = df["Date"].isna()
    duplicate_key = df.duplicated(["Ticker", "Date"], keep="first")
    invalid_ohlc = (
        (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
        | (df["High"] < df["Low"])
        | (df[["Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1))
        | (df["Volume"] < 0)
    )
    quality = {
        "input_rows": int(len(df)),
        "missing_date_rows": int(missing_date.sum()),
        "duplicate_ticker_date_rows": int(duplicate_key.sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
    }

    df["QC_OHLC_Invalid"] = invalid_ohlc.astype(int)
    if drop_invalid:
        df = df.loc[~missing_date & ~duplicate_key & ~invalid_ohlc].copy()

    info_columns = [
        "Ticker",
        "Market",
        "Board",
        "Industry",
        "Region",
        "IsST",
        "ListingDate",
    ]
    info = stock_info[[column for column in info_columns if column in stock_info.columns]].copy()
    info["IsST"] = pd.to_numeric(info["IsST"], errors="coerce").fillna(0).astype(int)
    info["ListingDate"] = pd.to_datetime(info["ListingDate"], errors="coerce")
    df = df.drop(
        columns=[column for column in info.columns if column != "Ticker" and column in df.columns],
        errors="ignore",
    )
    df = df.merge(info, on="Ticker", how="left", validate="many_to_one")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    quality["output_rows"] = int(len(df))
    quality["dropped_rows"] = quality["input_rows"] - quality["output_rows"]
    return df, quality


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> DatasetBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    stock_info = _read_sheet(path, "StockInfo", "Ticker")
    daily_raw = _read_sheet(path, "DailyPrices", "Ticker")
    single_sheet = _read_sheet(path, "Single_Moutai", "Date")
    provided_ml = _read_sheet(path, "ML_Features", "Ticker")
    data_dictionary = _read_sheet(path, "DataDictionary", "Sheet")
    sources = _read_sheet(path, "Sources", "SourceID")

    stock_info["ListingDate"] = pd.to_datetime(stock_info["ListingDate"], errors="coerce")
    stock_info["IsST"] = pd.to_numeric(stock_info["IsST"], errors="coerce").fillna(0).astype(int)
    daily_prices, quality = clean_daily_prices(daily_raw, stock_info)
    return DatasetBundle(
        stock_info=stock_info,
        daily_prices=daily_prices,
        single_sheet=single_sheet,
        provided_ml_features=provided_ml,
        data_dictionary=data_dictionary,
        sources=sources,
        quality=quality,
    )

