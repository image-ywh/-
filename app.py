from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import DEFAULT_DATA_PATH, load_dataset
from src.features import build_ml_features
from src.ml_selection import train_and_screen
from src.plots import (
    bar_distribution,
    donut_distribution,
    feature_importance_figure,
    model_metrics_figure,
    monte_carlo_distribution_figure,
    monte_carlo_paths_figure,
    price_ma_figure,
    return_distribution_figure,
    return_time_figure,
    volume_figure,
)
from src.single_stock import prepare_single_stock, simulate_gbm, summarize_single_stock


st.set_page_config(
    page_title="A股量化分析与机器学习选股平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --quant-bg: #030712;
        --quant-panel: rgba(10, 20, 38, 0.76);
        --quant-panel-strong: rgba(12, 25, 46, 0.92);
        --quant-border: rgba(96, 165, 250, 0.16);
        --quant-text: #e5eefc;
        --quant-muted: #8ea3bf;
        --quant-cyan: #22d3ee;
        --quant-blue: #3b82f6;
        --quant-violet: #8b5cf6;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"], .stApp {
        background: #030712 !important;
        color: var(--quant-text);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 8%, rgba(37, 99, 235, 0.17), transparent 28rem),
            radial-gradient(circle at 91% 19%, rgba(139, 92, 246, 0.14), transparent 24rem),
            radial-gradient(circle at 55% 76%, rgba(6, 182, 212, 0.07), transparent 34rem),
            linear-gradient(160deg, #030712 0%, #06101f 48%, #02050b 100%) !important;
        background-attachment: fixed !important;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: .26;
        background-image:
            linear-gradient(rgba(96, 165, 250, .035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(96, 165, 250, .035) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 90%);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding: 2.2rem 3.2rem 5rem;
        position: relative;
        z-index: 1;
    }

    header[data-testid="stHeader"] {
        background: rgba(3, 7, 18, .68) !important;
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(96, 165, 250, .08);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(9, 18, 35, .98), rgba(3, 7, 18, .98)) !important;
        border-right: 1px solid var(--quant-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #b8c8dc !important;
    }

    .sidebar-brand {
        padding: .55rem 0 .75rem;
    }

    .sidebar-brand strong {
        display: block;
        color: #f8fbff;
        font-size: 1.28rem;
        letter-spacing: -.02em;
    }

    .sidebar-brand span {
        display: block;
        margin-top: .35rem;
        color: var(--quant-muted);
        font-size: .82rem;
        line-height: 1.55;
    }

    .quant-hero {
        position: relative;
        overflow: hidden;
        padding: 2.35rem 2.5rem 2.15rem;
        margin: .3rem 0 1.25rem;
        border: 1px solid rgba(96, 165, 250, .2);
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(15, 33, 61, .93), rgba(7, 16, 31, .82)),
            radial-gradient(circle at 80% 20%, rgba(34, 211, 238, .2), transparent 35%);
        box-shadow:
            0 30px 80px rgba(0, 0, 0, .36),
            inset 0 1px rgba(255, 255, 255, .06);
        animation: quantFadeUp .75s cubic-bezier(.2,.8,.2,1) both;
    }

    .quant-hero::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        top: -210px;
        right: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59, 130, 246, .34), transparent 68%);
        animation: quantFloat 7s ease-in-out infinite;
    }

    .hero-kicker {
        display: flex;
        align-items: center;
        gap: .55rem;
        color: #8cecff;
        font-size: .74rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, .6);
        animation: quantPulse 2s infinite;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        margin: .72rem 0 .42rem;
        color: #f8fbff;
        font-size: clamp(2.1rem, 4vw, 3.75rem);
        font-weight: 820;
        line-height: 1.08;
        letter-spacing: -.055em;
        background: linear-gradient(100deg, #ffffff 6%, #bfdbfe 52%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0;
        color: #9fb2ca;
        font-size: 1rem;
        line-height: 1.75;
    }

    .hero-pills {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: .65rem;
        margin-top: 1.35rem;
    }

    .hero-pills span {
        padding: .48rem .78rem;
        border: 1px solid rgba(125, 211, 252, .14);
        border-radius: 999px;
        background: rgba(8, 25, 47, .7);
        color: #b9cce2;
        font-size: .78rem;
        backdrop-filter: blur(10px);
    }

    .hero-pills b {
        color: #f0f8ff;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        position: relative;
        min-height: 118px;
        padding: 1.2rem 1.25rem;
        overflow: hidden;
        border: 1px solid var(--quant-border);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(13, 29, 52, .82), rgba(7, 17, 32, .76));
        box-shadow: 0 12px 34px rgba(0, 0, 0, .2), inset 0 1px rgba(255,255,255,.04);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
        animation: quantFadeUp .65s cubic-bezier(.2,.8,.2,1) both;
    }

    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--quant-cyan), var(--quant-violet), transparent);
        opacity: .6;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(34, 211, 238, .38);
        box-shadow: 0 18px 42px rgba(0, 0, 0, .3), 0 0 28px rgba(34, 211, 238, .07);
    }

    [data-testid="stMetricLabel"] {
        color: #8fa6c0 !important;
        font-size: .78rem !important;
        letter-spacing: .04em;
    }

    [data-testid="stMetricValue"] {
        color: #f4f9ff !important;
        font-size: 1.72rem !important;
        font-weight: 760 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        margin: 1.6rem 0 .9rem;
        padding: .38rem;
        width: fit-content;
        max-width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(96, 165, 250, .12);
        border-radius: 14px;
        background: rgba(7, 16, 30, .74);
        backdrop-filter: blur(18px);
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.65rem;
        padding: 0 1rem;
        border-radius: 10px;
        color: #8ea3bd;
        font-weight: 650;
        transition: color .22s ease, background .22s ease, transform .22s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #dff7ff;
        background: rgba(59, 130, 246, .1);
        transform: translateY(-1px);
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(120deg, rgba(37, 99, 235, .85), rgba(8, 145, 178, .8)) !important;
        box-shadow: 0 8px 22px rgba(14, 116, 144, .22);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stJson"] {
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, .13);
        border-radius: 18px;
        background: rgba(7, 17, 32, .62);
        box-shadow: 0 14px 38px rgba(0, 0, 0, .18);
        animation: quantFadeUp .6s cubic-bezier(.2,.8,.2,1) both;
        transition: border-color .25s ease, box-shadow .25s ease;
    }

    [data-testid="stPlotlyChart"]:hover,
    [data-testid="stDataFrame"]:hover {
        border-color: rgba(34, 211, 238, .27);
        box-shadow: 0 18px 48px rgba(0, 0, 0, .28);
    }

    h1, h2, h3 {
        color: #edf6ff !important;
        letter-spacing: -.025em;
    }

    h2, h3 {
        margin-top: 1.4rem !important;
    }

    [data-testid="stCaptionContainer"], .stCaption {
        color: #849ab5 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #b5c4d7;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.7rem;
        border: 1px solid rgba(34, 211, 238, .22) !important;
        border-radius: 11px !important;
        background: linear-gradient(120deg, #2563eb, #0891b2) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 24px rgba(8, 145, 178, .17);
        transition: transform .22s ease, box-shadow .22s ease, filter .22s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.12);
        box-shadow: 0 15px 32px rgba(8, 145, 178, .27);
    }

    [data-baseweb="select"] > div,
    [data-testid="stSlider"] [data-baseweb="slider"] {
        border-color: rgba(96, 165, 250, .18) !important;
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(96, 165, 250, .16);
        border-radius: 14px;
        background: rgba(11, 27, 48, .76);
    }

    footer {
        background: #030712 !important;
    }

    @keyframes quantFadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes quantPulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, .55); }
        70% { box-shadow: 0 0 0 9px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    @keyframes quantFloat {
        0%, 100% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(-18px, 16px, 0); }
    }

    @media (max-width: 820px) {
        [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
        .quant-hero { padding: 1.65rem 1.35rem; border-radius: 20px; }
        .hero-pills { gap: .45rem; }
        .stTabs [data-baseweb="tab"] { padding: 0 .7rem; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_bundle(data_path: str):
    return load_dataset(data_path)


@st.cache_data(show_spinner=False)
def get_features(data_path: str):
    return build_ml_features(get_bundle(data_path).daily_prices)


@st.cache_resource(show_spinner=False)
def get_model(data_path: str, threshold: float):
    return train_and_screen(get_features(data_path), threshold=threshold)


data_path = Path(DEFAULT_DATA_PATH)
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <strong>量化控制台</strong>
            <span>调整分析标的、模拟规模与机器学习筛选阈值</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not data_path.exists():
        st.error(f"数据文件不存在：{data_path}")
        st.stop()
    ticker_options = get_bundle(str(data_path)).stock_info["Ticker"].tolist()
    ticker = st.selectbox(
        "单股分析标的",
        ticker_options,
        index=ticker_options.index("600519.SS") if "600519.SS" in ticker_options else 0,
        format_func=lambda value: f"{value} | {get_bundle(str(data_path)).stock_info.loc[get_bundle(str(data_path)).stock_info['Ticker'].eq(value), 'StockNameCN'].iloc[0]}",
    )
    simulation_paths = st.slider("蒙特卡洛路径数", 1000, 10000, 5000, step=1000)
    probability_threshold = st.slider("选股概率阈值", 0.50, 0.80, 0.55, step=0.01)

bundle = get_bundle(str(data_path))
features = get_features(str(data_path))
single = prepare_single_stock(bundle.daily_prices, ticker)
single_summary = summarize_single_stock(single)
simulation = simulate_gbm(single, n_paths=simulation_paths)

st.markdown(
    f"""
    <section class="quant-hero">
        <div class="hero-kicker">
            <span class="status-dot"></span>
            QUANT INTELLIGENCE CONSOLE
        </div>
        <h1 class="hero-title">A股量化分析平台</h1>
        <p class="hero-subtitle">
            从历史行情、风险模拟到机器学习选股，以一套可交互的分析流程洞察市场信号。
        </p>
        <div class="hero-pills">
            <span>数据区间 <b>2021-01-04 — 2026-07-24</b></span>
            <span>覆盖股票 <b>{bundle.stock_info["Ticker"].nunique()} 只</b></span>
            <span>有效行情 <b>{len(bundle.daily_prices):,} 条</b></span>
            <span>当前标的 <b>{ticker}</b></span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

quality_cols = st.columns(4)
quality_cols[0].metric("最新收盘价", f"{single_summary['latest_close']:.2f}")
quality_cols[1].metric("区间累计收益", f"{single_summary['total_return']:.1%}")
quality_cols[2].metric("年化波动率", f"{single_summary['annualized_volatility']:.1%}")
quality_cols[3].metric("99%风险分位价", f"{simulation.lower_1pct_price:.2f}")

tabs = st.tabs(["单只股票量化分析", "多股票统计", "机器学习选股", "数据说明"])

with tabs[0]:
    st.subheader(f"{single_summary['stock_name']}（{ticker}）量化分析")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(price_ma_figure(single), width="stretch")
        st.plotly_chart(return_time_figure(single), width="stretch")
    with right:
        st.plotly_chart(volume_figure(single), width="stretch")
        st.plotly_chart(return_distribution_figure(single), width="stretch")

    st.subheader("蒙特卡洛模拟（几何布朗运动）")
    st.caption(
        "风险分位采用终值价格的下侧1%分位数，对应99%置信水平的下行风险；"
        "模拟参数由历史对数收益率估计。"
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            monte_carlo_paths_figure(simulation.paths, simulation.current_price),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            monte_carlo_distribution_figure(
                simulation.terminal_prices,
                simulation.current_price,
                simulation.lower_1pct_price,
            ),
            width="stretch",
        )
    mc_metrics = st.columns(4)
    mc_metrics[0].metric("年化漂移", f"{simulation.mu_annualized:.1%}")
    mc_metrics[1].metric("年化波动率", f"{simulation.sigma_annualized:.1%}")
    mc_metrics[2].metric("99% VaR金额", f"{simulation.var_amount_99:.2f}")
    mc_metrics[3].metric("终值高于当前价概率", f"{simulation.upside_probability:.1%}")

with tabs[1]:
    st.subheader("股票行业、地区与市场分布")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            bar_distribution(bundle.stock_info["Industry"], "行业分布", "Industry"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            bar_distribution(bundle.stock_info["Region"], "地区分布", "Region"),
            width="stretch",
        )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            donut_distribution(bundle.stock_info["Market"], "市场类型占比", "Market"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            donut_distribution(bundle.stock_info["Board"], "板块类型占比", "Board"),
            width="stretch",
        )

with tabs[2]:
    st.subheader("机器学习二分类模型与实盘筛选")
    with st.spinner("正在训练模型并生成候选股票清单..."):
        model_result = get_model(str(data_path), probability_threshold)
    metric = model_result.metrics[model_result.primary_name]["test"]
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(model_metrics_figure(metric), width="stretch")
    with right:
        st.write("时间切分")
        st.json(model_result.split_dates)
        st.write("模型特征")
        st.code(", ".join(model_result.feature_columns))
    st.markdown(
        f"主模型：**{model_result.primary_name}** · "
        f"测试集灵敏度：**{metric['sensitivity_recall']:.1%}** · "
        f"候选数量：**{len(model_result.candidates)}**"
    )
    st.dataframe(
        model_result.candidates.style.format(
            {"ProbabilityUp5D": "{:.1%}", "Expected5DReturn": "{:.1%}", "Close": "{:.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "下载候选股票清单",
        data=model_result.candidates.to_csv(index=False).encode("utf-8-sig"),
        file_name="candidate_stocks.csv",
        mime="text/csv",
    )
    st.subheader("特征重要性")
    st.plotly_chart(
        feature_importance_figure(model_result.feature_importance),
        width="stretch",
    )

with tabs[3]:
    st.subheader("数据质量与来源")
    st.json(bundle.quality)
    st.write("数据字典")
    st.dataframe(bundle.data_dictionary, width="stretch", hide_index=True)
    st.write("来源与审计记录")
    st.dataframe(bundle.sources, width="stretch", hide_index=True)
    st.info(
        "当前数据集中没有ST股票；模型代码保留了ST过滤逻辑。"
        "数据清洗会剔除重复键、缺失关键OHLCV和OHLC逻辑异常记录。"
    )
