from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_DATA_PATH, load_dataset
from src.features import build_ml_features
from src.ml_selection import train_and_screen
from src.plots import (
    bar_distribution,
    monte_carlo_distribution_figure,
    monte_carlo_paths_figure,
    price_ma_figure,
    return_distribution_figure,
    return_time_figure,
    volume_figure,
)
from src.single_stock import prepare_single_stock, simulate_gbm, summarize_single_stock
from src.static_charts import save_static_charts


def parse_args():
    parser = argparse.ArgumentParser(description="Generate project analysis artifacts.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--ticker", default="600519.SS")
    parser.add_argument("--paths", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(args.data)
    single = prepare_single_stock(bundle.daily_prices, args.ticker)
    summary = summarize_single_stock(single)
    simulation = simulate_gbm(single, n_paths=args.paths)
    features = build_ml_features(bundle.daily_prices)
    model = train_and_screen(features, threshold=args.threshold)

    bundle.daily_prices.to_csv(output / "clean_daily_prices.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output / "ml_features_recomputed.csv", index=False, encoding="utf-8-sig")
    model.candidates.to_csv(output / "candidate_stocks.csv", index=False, encoding="utf-8-sig")
    single.to_csv(output / f"{args.ticker}_single_analysis.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "single_stock": summary,
        "monte_carlo": {
            "current_price": simulation.current_price,
            "mu_annualized": simulation.mu_annualized,
            "sigma_annualized": simulation.sigma_annualized,
            "expected_terminal_price": simulation.expected_terminal_price,
            "lower_1pct_price": simulation.lower_1pct_price,
            "var_amount_99": simulation.var_amount_99,
            "var_pct_99": simulation.var_pct_99,
            "upside_probability": simulation.upside_probability,
        },
        "data_quality": bundle.quality,
        "model": {
            "primary_name": model.primary_name,
            "metrics": model.metrics,
            "split_dates": model.split_dates,
            "feature_columns": model.feature_columns,
        },
    }
    with (output / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    figures = {
        "single_price_ma": price_ma_figure(single),
        "single_return_time": return_time_figure(single),
        "single_return_distribution": return_distribution_figure(single),
        "single_volume_distribution": volume_figure(single),
        "monte_carlo_paths": monte_carlo_paths_figure(
            simulation.paths, simulation.current_price
        ),
        "monte_carlo_distribution": monte_carlo_distribution_figure(
            simulation.terminal_prices,
            simulation.current_price,
            simulation.lower_1pct_price,
        ),
        "industry_distribution": bar_distribution(
            bundle.stock_info["Industry"], "行业分布", "Industry"
        ),
        "region_distribution": bar_distribution(
            bundle.stock_info["Region"], "地区分布", "Region"
        ),
    }
    for name, figure in figures.items():
        figure.write_html(output / f"{name}.html", include_plotlyjs="cdn")
    save_static_charts(output, single, simulation, bundle.stock_info, model)

    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
