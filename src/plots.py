from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CATEGORY_TRANSLATIONS = {
    "Industry": {
        "Automobiles": "汽车",
        "Banking": "银行",
        "Electrical Equipment / Batteries": "电力设备 / 电池",
        "Electrical Equipment / Solar": "电力设备 / 光伏",
        "Electronics": "电子",
        "Food & Beverage": "食品饮料",
        "Home Appliances": "家用电器",
        "Non-bank Finance": "非银金融",
        "Non-bank Finance / Financial IT": "非银金融 / 金融科技",
        "Pharmaceuticals": "医药生物",
        "Utilities": "公用事业",
    },
    "Region": {
        "Beijing": "北京",
        "Fujian": "福建",
        "Guangdong": "广东",
        "Guizhou": "贵州",
        "Jiangsu": "江苏",
        "Shaanxi": "陕西",
        "Shanghai": "上海",
        "Sichuan": "四川",
        "Zhejiang": "浙江",
    },
    "Market": {
        "Shanghai": "上海证券交易所",
        "Shenzhen": "深圳证券交易所",
    },
    "Board": {
        "Main Board": "主板",
        "ChiNext": "创业板",
    },
}

CATEGORY_LABELS = {
    "Industry": "行业",
    "Region": "地区",
    "Market": "市场",
    "Board": "板块",
}

FEATURE_LABELS = {
    "IntradayReturn": "日内收益率",
    "RangePct": "日内振幅",
    "DailyReturn": "日收益率",
    "VolumeChange": "成交量变化",
    "Return5": "5日收益率",
    "Return20": "20日收益率",
    "MA20Gap": "价格偏离20日均线",
    "Volatility20Annualized": "年化20日波动率",
    "IsST": "ST标记",
}


def _localize_category(series: pd.Series, label: str) -> tuple[pd.Series, str]:
    translations = CATEGORY_TRANSLATIONS.get(label, {})
    localized = series.astype("string").replace(translations)
    return localized, CATEGORY_LABELS.get(label, label)


def _base(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=38, r=24, t=68, b=42),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,12,24,0.30)",
        font=dict(color="#b9c9dd", family="Inter, Microsoft YaHei, sans-serif"),
        title=dict(font=dict(color="#f0f7ff", size=18), x=0.025, xanchor="left"),
        colorway=["#38bdf8", "#818cf8", "#2dd4bf", "#f59e0b", "#fb7185", "#a78bfa"],
        legend=dict(
            bgcolor="rgba(8,20,38,0.68)",
            bordercolor="rgba(125,211,252,0.12)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#0b1930",
            bordercolor="rgba(125,211,252,0.25)",
            font=dict(color="#edf7ff"),
        ),
        transition=dict(duration=420, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.14)",
        linecolor="rgba(148,163,184,0.16)",
        tickfont=dict(color="#8da3bd"),
        title_font=dict(color="#9fb3ca"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.14)",
        linecolor="rgba(148,163,184,0.16)",
        tickfont=dict(color="#8da3bd"),
        title_font=dict(color="#9fb3ca"),
    )
    return fig


def price_ma_figure(df: pd.DataFrame) -> go.Figure:
    columns = ["Close", "MA20", "MA52", "MA252"]
    labels = {"Close": "收盘价", "MA20": "MA20", "MA52": "MA52", "MA252": "MA252"}
    fig = go.Figure()
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[column],
                name=labels[column],
                mode="lines",
                line=dict(width=2 if column == "Close" else 1.3),
            )
        )
    fig.update_layout(title="收盘价与多周期均线", yaxis_title="价格")
    return _base(fig)


def return_time_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="Date", y="DailyReturn", title="日收益率时序")
    fig.update_yaxes(tickformat=".1%")
    return _base(fig)


def return_distribution_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=["DailyReturn"]),
        x="DailyReturn",
        nbins=60,
        title="日收益率分布",
        marginal="box",
    )
    fig.update_xaxes(tickformat=".1%")
    return _base(fig)


def volume_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Volume", nbins=50, title="成交量分布")
    return _base(fig)


def monte_carlo_paths_figure(paths, current_price: float, max_paths: int = 80) -> go.Figure:
    fig = go.Figure()
    for path in paths[:max_paths]:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(path))),
                y=path,
                mode="lines",
                line=dict(width=0.7, color="rgba(56, 189, 248, 0.19)"),
                showlegend=False,
            )
        )
    fig.add_hline(y=current_price, line_dash="dash", line_color="#f8fafc")
    fig.update_layout(title="蒙特卡洛模拟价格路径", xaxis_title="未来交易日", yaxis_title="模拟价格")
    return _base(fig, 460)


def monte_carlo_distribution_figure(
    terminal_prices, current_price: float, lower_1pct_price: float
) -> go.Figure:
    fig = px.histogram(
        x=terminal_prices,
        nbins=60,
        title="未来一年终值价格分布（99%下行风险分位线）",
        labels={"x": "终值价格", "y": "路径数量"},
    )
    fig.add_vline(
        x=lower_1pct_price,
        line_color="#fb7185",
        line_width=3,
        annotation_text=f"99%风险分位：{lower_1pct_price:.2f}",
        annotation_position="top left",
    )
    fig.add_vline(x=current_price, line_dash="dash", line_color="#f8fafc")
    return _base(fig)


def bar_distribution(series: pd.Series, title: str, label: str) -> go.Figure:
    localized, localized_label = _localize_category(series, label)
    values = (
        localized.value_counts()
        .rename_axis(localized_label)
        .reset_index(name="数量")
    )
    fig = px.bar(
        values,
        x=localized_label,
        y="数量",
        title=title,
        text_auto=True,
        labels={localized_label: localized_label, "数量": "股票数量"},
    )
    fig.update_traces(
        hovertemplate=f"{localized_label}：%{{x}}<br>股票数量：%{{y}}<extra></extra>"
    )
    return _base(fig, 360)


def donut_distribution(series: pd.Series, title: str, label: str) -> go.Figure:
    localized, localized_label = _localize_category(series, label)
    values = (
        localized.value_counts()
        .rename_axis(localized_label)
        .reset_index(name="数量")
    )
    fig = px.pie(
        values,
        names=localized_label,
        values="数量",
        hole=0.48,
        title=title,
    )
    fig.update_traces(
        texttemplate="%{percent:.0%}",
        hovertemplate=(
            f"{localized_label}：%{{label}}<br>"
            "股票数量：%{value}<br>占比：%{percent:.1%}<extra></extra>"
        ),
    )
    fig.update_layout(legend_title_text=localized_label)
    return _base(fig, 360)


def model_metrics_figure(metrics: dict) -> go.Figure:
    names = ["accuracy", "precision", "sensitivity_recall", "f1", "roc_auc"]
    labels = {
        "accuracy": "准确率",
        "precision": "精确率",
        "sensitivity_recall": "灵敏度/召回率",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    fig = go.Figure(
        go.Bar(
            x=[labels[name] for name in names],
            y=[metrics[name] for name in names],
            text=[f"{metrics[name]:.1%}" for name in names],
            textposition="auto",
            marker_color=["#38bdf8", "#2dd4bf", "#f59e0b", "#a78bfa", "#818cf8"],
        )
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title="模型测试集评价指标")
    return _base(fig, 380)


def feature_importance_figure(importance: pd.DataFrame) -> go.Figure:
    data = importance.sort_values("Importance", ascending=True)
    feature_labels = data["Feature"].map(FEATURE_LABELS).fillna(data["Feature"])
    fig = go.Figure(
        go.Bar(
            x=data["Importance"],
            y=feature_labels,
            orientation="h",
            marker=dict(
                color=data["Importance"],
                colorscale=[[0, "#2563eb"], [0.55, "#06b6d4"], [1, "#67e8f9"]],
                line=dict(color="rgba(255,255,255,0.08)", width=1),
            ),
        )
    )
    fig.update_layout(title="随机森林特征重要性", xaxis_title="重要性", yaxis_title="")
    return _base(fig, 420)
