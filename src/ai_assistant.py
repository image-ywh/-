from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


ASSISTANT_NAME = "Serene 小智"


@dataclass(frozen=True)
class StockAssistantContext:
    ticker: str
    stock_name: str
    latest_date: str
    latest_close: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    recent_return_20d: float
    ma20_gap: float
    simulation_expected_return: float
    simulation_upside_probability: float
    simulation_var_pct: float
    model_name: str
    model_test_accuracy: float
    model_test_roc_auc: float
    probability_up_5d: float | None
    expected_5d_return: float | None
    probability_threshold: float
    stock_pool: int
    effective_candidates: int
    top_candidates: tuple[tuple[str, str, float], ...]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def build_assistant_context(
    ticker: str,
    single: pd.DataFrame,
    single_summary: dict,
    simulation,
    features: pd.DataFrame,
    model_result,
) -> StockAssistantContext:
    """Build a data-grounded snapshot for the in-app assistant."""
    latest = single.iloc[-1]
    recent_return = single["Close"].pct_change(20).iloc[-1]
    ma20_gap = latest["Close"] / latest["MA20"] - 1 if pd.notna(latest["MA20"]) else np.nan

    probability: float | None = None
    expected_return: float | None = None
    model_features = model_result.feature_columns
    latest_rows = (
        features.dropna(subset=model_features)
        .sort_values(["Date", "Ticker"])
        .groupby("Ticker", as_index=False)
        .tail(1)
    )
    selected_row = latest_rows.loc[latest_rows["Ticker"].eq(ticker)]
    if not selected_row.empty:
        probability = _safe_float(
            model_result.primary_model.predict_proba(selected_row[model_features])[:, 1][0],
            default=np.nan,
        )
        if np.isfinite(probability):
            train_metrics = model_result.metrics[model_result.primary_name]
            train = train_metrics.get("train", {})
            # The model stores the class-conditional returns in the candidate
            # calculation; use the selected row when available and otherwise
            # leave expected return unknown rather than inventing a forecast.
            positive_candidates = model_result.candidates.loc[
                model_result.candidates["Ticker"].eq(ticker), "Expected5DReturn"
            ]
            if not positive_candidates.empty:
                expected_return = _safe_float(positive_candidates.iloc[0], np.nan)

    top_candidates: list[tuple[str, str, float]] = []
    if not model_result.candidates.empty:
        for row in model_result.candidates.head(5).itertuples():
            top_candidates.append(
                (
                    str(row.Ticker),
                    str(row.StockNameCN),
                    _safe_float(row.ProbabilityUp5D),
                )
            )

    model_test = model_result.metrics[model_result.primary_name]["test"]
    return StockAssistantContext(
        ticker=ticker,
        stock_name=str(single_summary["stock_name"]),
        latest_date=str(single_summary["end_date"]),
        latest_close=_safe_float(single_summary["latest_close"]),
        total_return=_safe_float(single_summary["total_return"]),
        annualized_return=_safe_float(single_summary["annualized_return"]),
        annualized_volatility=_safe_float(single_summary["annualized_volatility"]),
        max_drawdown=_safe_float(single_summary["max_drawdown"]),
        recent_return_20d=_safe_float(recent_return, np.nan),
        ma20_gap=_safe_float(ma20_gap, np.nan),
        simulation_expected_return=(
            _safe_float(simulation.expected_terminal_price / simulation.current_price - 1)
            if simulation.current_price
            else 0.0
        ),
        simulation_upside_probability=_safe_float(simulation.upside_probability),
        simulation_var_pct=_safe_float(simulation.var_pct_99),
        model_name=str(model_result.primary_name),
        model_test_accuracy=_safe_float(model_test["accuracy"]),
        model_test_roc_auc=_safe_float(model_test["roc_auc"]),
        probability_up_5d=probability,
        expected_5d_return=expected_return,
        probability_threshold=_safe_float(
            model_result.screening_summary["probability_threshold"], 0.55
        ),
        stock_pool=int(model_result.screening_summary["stock_pool"]),
        effective_candidates=int(model_result.screening_summary["effective_candidates"]),
        top_candidates=tuple(top_candidates),
    )


def _signal_label(probability: float | None, threshold: float) -> str:
    if probability is None:
        return "暂无有效模型信号"
    if probability >= threshold:
        return "模型信号偏积极"
    if probability <= 1 - threshold:
        return "模型信号偏谨慎"
    return "模型信号中性"


def risk_label(volatility: float, drawdown: float) -> str:
    if volatility >= 0.45 or drawdown <= -0.35:
        return "偏高"
    if volatility >= 0.25 or drawdown <= -0.20:
        return "中等"
    return "相对较低"


def answer_query(query: str, context: StockAssistantContext) -> str:
    """Answer common stock-analysis questions from the current project data.

    This deliberately stays data-grounded and does not fabricate news,
    fundamentals, or real-time quotes. It is usable without an API key.
    """
    question = re.sub(r"\s+", "", query.lower())
    if not question:
        return "请输入想了解的问题，例如“当前股票的风险如何？”或“随机森林给出的上涨概率是多少？”"

    if any(word in question for word in ("帮助", "能做什么", "怎么用", "功能")):
        return (
            "我可以基于当前页面数据分析：\n\n"
            "1. 当前价格、区间收益和近期趋势\n"
            "2. 波动率、最大回撤、VaR 和模拟上涨概率\n"
            "3. 随机森林模型的未来 5 日上涨概率与测试集表现\n"
            "4. 当前候选股票池及筛选数量\n\n"
            "例如可以问：`请分析当前股票风险`、`模型为什么不看好它？`、"
            "`当前选股池有哪些股票？`"
        )

    if any(word in question for word in ("候选", "股票池", "选股", "推荐", "有哪些")):
        if not context.top_candidates:
            return (
                f"当前股票池共有 {context.stock_pool} 只股票，经过阈值 "
                f"{context.probability_threshold:.0%} 筛选后暂无有效候选。"
            )
        names = "、".join(
            f"{ticker}（{name}，{probability:.1%}）"
            for ticker, name, probability in context.top_candidates
        )
        return (
            f"当前股票池 {context.stock_pool} 只，最终有效候选 {context.effective_candidates} 只。"
            f"按模型上涨概率排序，前几名为：{names}。\n\n"
            "这些是量化筛选结果，不等同于确定性买入建议。"
        )

    if any(word in question for word in ("风险", "波动", "回撤", "止损", "var")):
        return (
            f"{context.stock_name}（{context.ticker}）的年化波动率为 "
            f"{context.annualized_volatility:.1%}，最大回撤为 {context.max_drawdown:.1%}，"
            f"风险等级可视为“{risk_label(context.annualized_volatility, context.max_drawdown)}”。\n\n"
            f"蒙特卡洛模拟的 99% 下行风险约为 {context.simulation_var_pct:.1%}，"
            f"终值高于当前价的路径比例为 {context.simulation_upside_probability:.1%}。"
            "这只是历史波动外推，不是止损价或确定性预测。"
        )

    if any(word in question for word in ("模型", "随机森林", "概率", "预测", "准确", "泛化")):
        probability_text = (
            f"{context.probability_up_5d:.1%}"
            if context.probability_up_5d is not None
            else "暂无"
        )
        expected_text = (
            f"{context.expected_5d_return:.2%}"
            if context.expected_5d_return is not None
            else "暂无（当前股票未进入正向候选列表）"
        )
        return (
            f"当前主模型为 {context.model_name}。{context.stock_name} 的未来 5 日上涨概率为 "
            f"{probability_text}，{_signal_label(context.probability_up_5d, context.probability_threshold)}；"
            f"模型估计的 5 日预期收益为 {expected_text}。\n\n"
            f"模型测试集准确率为 {context.model_test_accuracy:.1%}，ROC-AUC 为 "
            f"{context.model_test_roc_auc:.1%}。该指标反映历史测试窗口表现，不能保证未来收益。"
        )

    if any(word in question for word in ("价格", "收盘", "行情", "趋势", "收益", "走势", "买")):
        ma_text = (
            f"收盘价较 MA20 高 {context.ma20_gap:.1%}"
            if np.isfinite(context.ma20_gap)
            else "MA20 尚未形成"
        )
        recent_text = (
            f"近 20 个交易日收益为 {context.recent_return_20d:.1%}"
            if np.isfinite(context.recent_return_20d)
            else "近 20 日收益暂无"
        )
        return (
            f"{context.stock_name}（{context.ticker}）截至 {context.latest_date} 的最新收盘价为 "
            f"{context.latest_close:.2f}。区间累计收益 {context.total_return:.1%}，"
            f"年化收益 {context.annualized_return:.1%}；{recent_text}，{ma_text}。\n\n"
            f"综合来看，当前价格趋势只能作为历史行情描述，是否买入还需要结合基本面、"
            "估值和实时市场信息。"
        )

    return (
        f"我已读取当前选中的 {context.stock_name}（{context.ticker}）。目前可确认："
        f"最新价 {context.latest_close:.2f}，年化波动率 {context.annualized_volatility:.1%}，"
        f"模型上涨概率 "
        f"{context.probability_up_5d:.1%}。"
        if context.probability_up_5d is not None
        else
        f"我已读取当前选中的 {context.stock_name}（{context.ticker}）。目前最新价为 "
        f"{context.latest_close:.2f}，但该股票暂无有效模型概率。"
    )
