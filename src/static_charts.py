from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .ml_selection import ModelResult
from .single_stock import MonteCarloResult


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_static_charts(
    output: str | Path,
    single: pd.DataFrame,
    simulation: MonteCarloResult,
    stock_info: pd.DataFrame,
    model: ModelResult,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(single["Date"], single["Close"], label="收盘价", linewidth=1.4)
    for column in ("MA20", "MA52", "MA252"):
        ax.plot(single["Date"], single[column], label=column, linewidth=1.0)
    ax.set_title("收盘价与多周期均线")
    ax.set_ylabel("价格")
    ax.legend(ncol=4)
    ax.grid(alpha=0.2)
    _save(fig, output, "01_close_and_moving_averages.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(single["Date"], single["DailyReturn"], linewidth=0.7, color="#1565C0")
    ax.axhline(0, color="#455A64", linewidth=0.8)
    ax.set_title("日收益率时序")
    ax.set_ylabel("日收益率")
    ax.grid(alpha=0.2)
    _save(fig, output, "02_daily_return_time_series.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(single["DailyReturn"].dropna(), bins=60, color="#42A5F5", edgecolor="white")
    ax.set_title("日收益率分布")
    ax.set_xlabel("日收益率")
    ax.set_ylabel("频数")
    _save(fig, output, "03_daily_return_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(single["Volume"].dropna(), bins=50, color="#66BB6A", edgecolor="white")
    ax.set_title("成交量分布")
    ax.set_xlabel("成交量")
    ax.set_ylabel("频数")
    _save(fig, output, "04_volume_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for path in simulation.paths[:80]:
        ax.plot(path, linewidth=0.5, alpha=0.2, color="#1E88E5")
    ax.axhline(simulation.current_price, linestyle="--", color="#263238")
    ax.set_title("蒙特卡洛模拟价格路径")
    ax.set_xlabel("未来交易日")
    ax.set_ylabel("模拟价格")
    _save(fig, output, "05_monte_carlo_paths.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        simulation.terminal_prices,
        bins=60,
        color="#7E57C2",
        edgecolor="white",
    )
    ax.axvline(
        simulation.lower_1pct_price,
        color="red",
        linewidth=2.2,
        label=f"99%风险分位：{simulation.lower_1pct_price:.2f}",
    )
    ax.axvline(simulation.current_price, color="#263238", linestyle="--", label="当前价")
    ax.set_title("未来一年终值价格分布")
    ax.set_xlabel("终值价格")
    ax.set_ylabel("路径数量")
    ax.legend()
    _save(fig, output, "06_monte_carlo_terminal_distribution.png")

    for column, filename, title in [
        ("Industry", "07_industry_distribution.png", "股票行业分布"),
        ("Region", "08_region_distribution.png", "股票地区分布"),
    ]:
        counts = stock_info[column].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        counts.plot.barh(ax=ax, color="#26A69A")
        ax.set_title(title)
        ax.set_xlabel("股票数量")
        ax.set_ylabel("")
        _save(fig, output, filename)

    counts = stock_info["Market"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    ax.set_title("市场类型占比")
    _save(fig, output, "09_market_share_donut.png")

    test_metrics = model.metrics[model.primary_name]["test"]
    keys = ["accuracy", "precision", "sensitivity_recall", "f1", "roc_auc"]
    labels = ["准确率", "精确率", "灵敏度", "F1", "ROC-AUC"]
    values = [test_metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F"])
    ax.set_ylim(0, 1)
    ax.set_title("机器学习模型测试集评价指标")
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    _save(fig, output, "10_model_metrics.png")

