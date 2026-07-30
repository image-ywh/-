from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plots import _base


DEFAULT_MODULE2_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "module2_results"
)

ALPHA158_FEATURE_LABELS = {
    "MAX5": "5日最高价指标",
    "KLOW2": "下影线长度占比",
    "IMIN5": "5日最低价位置",
    "RESI30": "30日趋势残差",
    "SUMD60": "60日涨跌强度差",
    "VMA20": "20日成交量均值比",
    "RESI60": "60日趋势残差",
    "RESI20": "20日趋势残差",
    "BETA20": "20日价格趋势斜率",
    "SUMN60": "60日下跌幅度占比",
    "ROC30": "30日价格变化率",
    "IMAX10": "10日最高价位置",
    "IMXD5": "5日高低点位置差",
    "ROC5": "5日价格变化率",
    "SUMN30": "30日下跌幅度占比",
}


@dataclass(frozen=True)
class Module2Results:
    metrics: dict
    live_summary: dict
    backtest_summary: dict
    candidates: pd.DataFrame
    return_curves: pd.DataFrame
    rank_ic: pd.DataFrame
    selection_metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_module2_results(
    root: str | Path = DEFAULT_MODULE2_ROOT,
) -> Module2Results:
    """Load the second module's Alpha158/XGBoost strategy artifacts."""
    root = Path(root)
    required = [
        root / "metrics.json",
        root / "live_screening_summary.json",
        root / "buy_candidates_latest.csv",
        root / "return_curves.csv",
        root / "daily_rank_ic.csv",
        root / "selection_metrics.csv",
        root / "feature_importance.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Module 2 result files are missing: " + ", ".join(missing)
        )

    candidates = pd.read_csv(
        root / "candidate_table_enriched.csv"
        if (root / "candidate_table_enriched.csv").exists()
        else root / "buy_candidates_latest.csv",
        encoding="utf-8-sig",
    )
    for column in ("datetime_x", "datetime_y"):
        if column in candidates.columns:
            candidates[column] = pd.to_datetime(
                candidates[column], errors="coerce"
            )

    return_curves = pd.read_csv(root / "return_curves.csv", encoding="utf-8-sig")
    return_curves["datetime"] = pd.to_datetime(
        return_curves["datetime"], errors="coerce"
    )
    rank_ic = pd.read_csv(root / "daily_rank_ic.csv", encoding="utf-8-sig")
    rank_ic["datetime"] = pd.to_datetime(rank_ic["datetime"], errors="coerce")
    selection_metrics = pd.read_csv(
        root / "selection_metrics.csv", encoding="utf-8-sig"
    )
    feature_importance = (
        pd.read_csv(root / "feature_importance.csv", encoding="utf-8-sig", index_col=0)
        .reset_index()
        .rename(columns={"index": "feature", "gain": "importance"})
    )
    backtest_path = root / "backtest" / "qlib_backtest_summary.json"
    return Module2Results(
        metrics=_read_json(root / "metrics.json"),
        live_summary=_read_json(root / "live_screening_summary.json"),
        backtest_summary=(
            _read_json(backtest_path) if backtest_path.exists() else {}
        ),
        candidates=candidates,
        return_curves=return_curves,
        rank_ic=rank_ic,
        selection_metrics=selection_metrics,
        feature_importance=feature_importance,
    )


def module2_return_figure(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = {
        "strategy_zero_cost": "Alpha158 XGBoost 策略",
        "SH000300_benchmark": "沪深300基准",
        "strategy_excess": "复合超额收益",
        "compounded_excess": "复合超额收益",
    }
    for column in (
        "strategy_zero_cost",
        "SH000300_benchmark",
        "strategy_excess",
        "compounded_excess",
    ):
        if column not in curves.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=curves["datetime"],
                y=curves[column],
                mode="lines",
                name=labels[column],
            )
        )
    fig.update_layout(
        title="Alpha158 + XGBoost 组合回测累计收益",
        yaxis_title="累计收益",
        xaxis_title="日期",
    )
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, 430)


def module2_rank_ic_figure(rank_ic: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "rank_ic" in rank_ic.columns:
        fig.add_trace(
            go.Scatter(
                x=rank_ic["datetime"],
                y=rank_ic["rank_ic"],
                mode="lines",
                name="日 RankIC",
                line=dict(color="#38bdf8", width=1),
            )
        )
    if "rolling_20d" in rank_ic.columns:
        fig.add_trace(
            go.Scatter(
                x=rank_ic["datetime"],
                y=rank_ic["rolling_20d"],
                mode="lines",
                name="20日滚动 RankIC",
                line=dict(color="#f59e0b", width=2),
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(title="横截面概率排序 RankIC", yaxis_title="RankIC")
    return _base(fig, 390)


def module2_selection_figure(selection: pd.DataFrame) -> go.Figure:
    frame = selection.sort_values("top_fraction").copy()
    frame["selection_label"] = frame["top_fraction"].map(
        lambda value: f"Top {value:.0%}"
    )
    frame["spread_label"] = frame["mean_return_spread"].map(
        lambda value: f"+{value * 100:.2f}pp"
    )
    frame["bar_label"] = [
        f"{value:.2%}<br>{spread}"
        for value, spread in zip(
            frame["selected_mean_forward_5d_return"],
            frame["spread_label"],
        )
    ]
    universe_return = frame["universe_mean_forward_5d_return"].iloc[0]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["selection_label"],
            y=frame["selected_mean_forward_5d_return"],
            name="高置信度组合",
            marker=dict(
                color=["#2dd4bf", "#38bdf8", "#818cf8"],
                line=dict(color="rgba(255,255,255,.18)", width=1),
            ),
            text=frame["bar_label"],
            textposition="outside",
            textfont=dict(color="#e7f8ff", size=13),
            customdata=frame[
                ["mean_return_spread", "selected_win_rate_precision", "selected_rows"]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "组合平均收益：%{y:.2%}<br>"
                "相对基准超额：%{customdata[0]:.2%}<br>"
                "上涨样本占比：%{customdata[1]:.1%}<br>"
                "样本数量：%{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["selection_label"],
            y=frame["universe_mean_forward_5d_return"],
            mode="lines+markers",
            name=f"全样本基准 {universe_return:.2%}",
            line=dict(color="#94a3b8", width=2, dash="dot"),
            marker=dict(
                color="#cbd5e1",
                size=7,
                line=dict(color="#0f172a", width=1),
            ),
            hovertemplate="全样本平均收益：%{y:.2%}<extra></extra>",
        )
    )
    upper_bound = (
        frame["selected_mean_forward_5d_return"].max() * 1.32
    )
    fig.update_layout(
        title="概率排序收益对比",
        barmode="overlay",
        bargap=0.56,
        yaxis_title="未来5日平均收益率",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        title_text="入选股票比例",
        fixedrange=True,
    )
    fig.update_yaxes(
        tickformat=".2%",
        range=[0, upper_bound],
        fixedrange=True,
        zeroline=True,
    )
    return _base(fig, 410)


def module2_feature_importance_figure(
    importance: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    frame = importance.sort_values("importance", ascending=False).head(top_n)
    frame = frame.sort_values("importance", ascending=True)
    frame["feature_label"] = frame["feature"].map(
        lambda feature: ALPHA158_FEATURE_LABELS.get(feature, feature)
    )
    fig = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=frame["feature_label"],
            orientation="h",
            marker_color="#818cf8",
            customdata=frame["feature"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Alpha158代码：%{customdata}<br>"
                "特征贡献度：%{x:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Alpha158 特征重要性前 {len(frame)} 名",
        xaxis_title="特征贡献度",
        yaxis_title="",
    )
    fig = _base(fig, 480)
    fig.update_layout(margin=dict(l=145, r=24, t=68, b=48))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def render_module2_panel(results: Module2Results) -> None:
    """Render the report-backed multi-stock strategy panel in Streamlit."""
    metrics = results.metrics
    test_metrics = metrics.get("test_metrics", {})
    returns = metrics.get("returns", {})
    live = results.live_summary
    st.markdown(
        """
        <section class="module2-hero">
            <div class="module2-kicker"><i></i>CROSS-SECTIONAL ALPHA ENGINE</div>
            <h2 class="module2-title">多股票智能策略</h2>
            <p class="module2-copy">
                融合 Alpha158 量价因子与 XGBoost 概率排序，以严格时间切分完成训练、
                验证和测试，并通过可交易性过滤、TopK 组合构建及五日轮动回测，
                将分类信号转化为可复核的多股票选股流程。
            </p>
            <div class="module2-tags">
                <span>158 维量价因子</span>
                <span>XGBoost 概率排序</span>
                <span>验证集阈值</span>
                <span>沪深300动态成分股</span>
                <span>五日调仓</span>
            </div>
        </section>
        <div class="module2-flow">
            <div class="module2-flow-step">
                <span>STEP 01</span><b>Alpha158 因子</b><small>多维量价信号提取</small>
            </div>
            <div class="module2-flow-step">
                <span>STEP 02</span><b>XGBoost 学习</b><small>非线性交互建模</small>
            </div>
            <div class="module2-flow-step">
                <span>STEP 03</span><b>概率横截面排序</b><small>优先保留强信号</small>
            </div>
            <div class="module2-flow-step">
                <span>STEP 04</span><b>可交易性过滤</b><small>剔除受限交易标的</small>
            </div>
            <div class="module2-flow-step">
                <span>STEP 05</span><b>TopK 五日轮动</b><small>组合回测与风险审计</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "测试 Accuracy", f"{test_metrics.get('accuracy', 0):.2%}"
    )
    metric_columns[1].metric(
        "测试 Precision", f"{test_metrics.get('precision', 0):.2%}"
    )
    metric_columns[2].metric(
        "测试 Recall",
        f"{test_metrics.get('selection_sensitivity_recall', 0):.2%}",
    )
    metric_columns[3].metric("测试 F1", f"{test_metrics.get('f1', 0):.2%}")
    metric_columns[4].metric(
        "测试 ROC-AUC", f"{test_metrics.get('roc_auc', 0):.2%}"
    )
    detail_columns = st.columns(4)
    detail_columns[0].metric("Alpha158 特征数", metrics.get("feature_count", 158))
    detail_columns[1].metric(
        "测试样本数", f"{test_metrics.get('rows', 0):,}"
    )
    top5 = (
        metrics.get("selection_metrics", [{}])[0]
        if metrics.get("selection_metrics")
        else {}
    )
    detail_columns[2].metric(
        "Top 5% 平均未来5日收益",
        f"{top5.get('selected_mean_forward_5d_return', 0):.2%}",
    )
    detail_columns[3].metric(
        "策略累计收益",
        f"{returns.get('strategy_total_return', 0):.2%}",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            module2_selection_figure(results.selection_metrics),
            width="stretch",
        )
        st.caption("柱顶第二行表示相对全样本基准的超额收益（百分点）。")
    with right:
        st.plotly_chart(
            module2_rank_ic_figure(results.rank_ic),
            width="stretch",
        )
    st.plotly_chart(module2_return_figure(results.return_curves), width="stretch")

    st.markdown(
        """
        <div class="module2-section-label">
            <strong>最新候选股票</strong>
            <span>LIVE SCREENING · PROBABILITY RANKING</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        f"筛选日期：{live.get('selection_date', '—')} · "
        f"候选数量：{live.get('effective_selection_count', len(results.candidates))} · "
        f"概率阈值：{live.get('probability_threshold', 0):.2%}"
    )
    candidate_columns = [
        column
        for column in [
            "rank",
            "instrument",
            "name",
            "industry",
            "market",
            "probability",
            "expected_5d_return",
            "daily_return",
        ]
        if column in results.candidates.columns
    ]
    candidate_view = results.candidates[candidate_columns].rename(
        columns={
            "rank": "排名",
            "instrument": "股票代码",
            "name": "股票名称",
            "industry": "行业",
            "market": "市场",
            "probability": "上涨概率",
            "expected_5d_return": "预期5日收益",
            "daily_return": "当日涨跌",
        }
    )
    st.dataframe(
        candidate_view.style.format(
            {
                "上涨概率": "{:.2%}",
                "预期5日收益": "{:.2%}",
                "当日涨跌": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "下载模块2候选股票",
        data=results.candidates.to_csv(index=False).encode("utf-8-sig"),
        file_name="module2_buy_candidates_latest.csv",
        mime="text/csv",
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(
            module2_feature_importance_figure(results.feature_importance),
            width="stretch",
        )
    with right:
        model = metrics.get("model", {})
        parameters = model.get("parameters", {})
        segments = metrics.get("segments", {})
        backtest = metrics.get("backtest", {})
        risk_rows = (
            (results.backtest_summary.get("risk_analysis") or [])
            if isinstance(results.backtest_summary, dict)
            else []
        )
        risk_summary = next(
            (
                item
                for item in risk_rows
                if item.get("series") in {"strategy_after_cost", "strategy_gross"}
            ),
            {},
        )
        train_segment = segments.get("train", ["—", "—"])
        valid_segment = segments.get("valid", ["—", "—"])
        test_segment = segments.get("test", ["—", "—"])
        annualized = risk_summary.get("compounded_annualized_return")
        max_drawdown = risk_summary.get("compounded_max_drawdown")
        annualized_text = f"{annualized:.2%}" if annualized is not None else "—"
        drawdown_text = f"{max_drawdown:.2%}" if max_drawdown is not None else "—"
        st.markdown(
            """
            <div class="module2-section-label">
                <strong>策略摘要</strong>
                <span>MODEL &amp; BACKTEST SNAPSHOT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="module2-param-card">
                <div class="module2-param-row">
                    <span>模型</span>
                    <strong>{model.get("type", "XGBoost")} · 深度 {parameters.get("max_depth", "—")} · η {parameters.get("eta", "—")}</strong>
                </div>
                <div class="module2-param-row">
                    <span>时间切分</span>
                    <strong>训练 {train_segment[0][:4]}–{train_segment[1][:4]} · 验证 {valid_segment[0][:4]}–{valid_segment[1][:4]} · 测试 {test_segment[0][:4]}–{test_segment[1][:4]}</strong>
                </div>
                <div class="module2-param-row">
                    <span>组合规则</span>
                    <strong>TopK {backtest.get("topk", 50)} · {backtest.get("holding_period", "五日")} · {backtest.get("benchmark", "SH000300")}</strong>
                </div>
                <div class="module2-param-row">
                    <span>风险表现</span>
                    <strong><em>年化 {annualized_text}</em> · 最大回撤 {drawdown_text}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("查看完整模型与回测参数", expanded=False):
            st.json(
                {
                    "model": model,
                    "segments": segments,
                    "backtest": backtest,
                }
            )
    st.info(
        "本页读取第二模块已完成的 Alpha158 + XGBoost 结果包；"
        "原始 Qlib 数据提供器、训练源码和模型文件仍保留在第二模块目录中。"
    )
