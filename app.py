from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import DEFAULT_DATA_PATH, load_dataset
from src.features import build_ml_features
from src.ai_assistant import (
    ASSISTANT_NAME,
    answer_query,
    build_assistant_context,
    risk_label,
)
from src.ml_selection import train_and_screen
from src.plots import (
    CATEGORY_TRANSLATIONS,
    FEATURE_LABELS,
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


# Bump this whenever model code or hyperparameters change so Streamlit does
# not reuse a model resource trained with an older implementation.
MODEL_CACHE_VERSION = "screening-summary-v8-original-rf"


st.set_page_config(
    page_title="A股量化分析与机器学习选股平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@500;600;700&family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --quant-bg: #0a0608;
        --quant-panel: rgba(12, 22, 38, 0.72);
        --quant-panel-strong: rgba(15, 28, 46, 0.9);
        --quant-border: rgba(160, 213, 226, 0.17);
        --quant-text: #e5eefc;
        --quant-muted: #8ea3bf;
        --quant-cyan: #70d8e5;
        --quant-blue: #5b8ee8;
        --quant-violet: #9b83d8;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"], .stApp {
        background: var(--quant-bg) !important;
        color: var(--quant-text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 2%, rgba(117, 75, 101, 0.2), transparent 28rem),
            radial-gradient(circle at 92% 15%, rgba(48, 136, 161, 0.2), transparent 28rem),
            radial-gradient(circle at 50% 80%, rgba(30, 93, 128, 0.13), transparent 34rem),
            linear-gradient(160deg, #0a0608 0%, #071421 48%, #04111a 100%) !important;
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
            linear-gradient(180deg, rgba(16, 10, 19, .98), rgba(4, 14, 24, .98)) !important;
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
        font-family: 'Dancing Script', cursive !important;
        font-size: 1.72rem;
        font-weight: 600;
        letter-spacing: -.02em;
    }

    .sidebar-brand span {
        display: block;
        margin-top: .35rem;
        color: var(--quant-muted);
        font-size: .82rem;
        line-height: 1.55;
    }

    /* Sidebar command deck: compact glass panel with a subtle data-grid texture. */
    [data-testid="stSidebar"] {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at 18% 6%, rgba(91, 142, 232, .23), transparent 15rem),
            radial-gradient(circle at 86% 72%, rgba(112, 216, 229, .11), transparent 18rem),
            linear-gradient(180deg, rgba(10, 20, 42, .99), rgba(4, 12, 27, .99)) !important;
        box-shadow: 18px 0 55px rgba(0, 0, 0, .18);
    }

    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: .34;
        background-image:
            linear-gradient(rgba(112, 216, 229, .055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(112, 216, 229, .055) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: linear-gradient(to bottom, black 0%, transparent 78%);
    }

    [data-testid="stSidebar"] > div:first-child {
        position: relative;
        z-index: 1;
    }

    .sidebar-brand {
        position: relative;
        padding: .8rem .2rem 1.05rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(160, 213, 226, .15);
    }

    .sidebar-brand::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: -1px;
        width: 62px;
        height: 2px;
        border-radius: 999px;
        background: linear-gradient(90deg, #e45a84, #70d8e5);
        box-shadow: 0 0 16px rgba(112, 216, 229, .35);
    }

    .sidebar-brand strong {
        font-family: "STKaiti", "KaiTi", "FZKai-Z03", "Kaiti SC", serif !important;
        font-size: 1.92rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-shadow: 0 0 24px rgba(140, 228, 232, .22);
        filter: drop-shadow(0 2px 0 rgba(255, 255, 255, .05));
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #d6e6f5 !important;
        font-size: .77rem !important;
        font-weight: 700 !important;
        letter-spacing: .04em;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border: 1px solid rgba(112, 216, 229, .2) !important;
        border-radius: 14px !important;
        background: rgba(9, 27, 52, .76) !important;
        box-shadow: inset 0 1px rgba(255, 255, 255, .05), 0 8px 24px rgba(0, 0, 0, .12);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
        transform: translateY(-1px);
        border-color: rgba(112, 216, 229, .54) !important;
        box-shadow: 0 0 26px rgba(112, 216, 229, .11);
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: .7rem .78rem .82rem;
        margin: .35rem 0 .9rem;
        border: 1px solid rgba(125, 211, 252, .12);
        border-radius: 15px;
        background: rgba(9, 24, 42, .56);
        box-shadow: inset 0 1px rgba(255, 255, 255, .035);
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        border: 2px solid #ffe7ef !important;
        background: #e45a84 !important;
        box-shadow: 0 0 0 5px rgba(228, 90, 132, .12), 0 0 22px rgba(228, 90, 132, .38);
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
        background: rgba(128, 161, 195, .28) !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] {
        color: #718aa8 !important;
    }

    .sidebar-footer {
        margin-top: 2.1rem;
        padding: 1rem;
        border: 1px solid rgba(112, 216, 229, .16);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(14, 39, 62, .76), rgba(8, 20, 38, .68));
        box-shadow: inset 0 1px rgba(255, 255, 255, .05), 0 14px 34px rgba(0, 0, 0, .18);
    }

    .sidebar-status {
        display: flex;
        align-items: center;
        gap: .45rem;
        color: #9cf3d1;
        font-size: .64rem;
        font-weight: 800;
        letter-spacing: .14em;
    }

    .sidebar-status i {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, .6);
        animation: quantPulse 2s infinite;
    }

    .sidebar-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .55rem;
        margin-top: .8rem;
    }

    .sidebar-mini {
        padding: .62rem .65rem;
        border: 1px solid rgba(125, 211, 252, .1);
        border-radius: 11px;
        background: rgba(6, 18, 34, .55);
    }

    .sidebar-mini span,
    .sidebar-footer p {
        color: #8197b2 !important;
        font-size: .67rem;
        line-height: 1.55;
    }

    .sidebar-mini b {
        display: block;
        margin-top: .18rem;
        color: #e8f5ff;
        font-size: .92rem;
    }

    .sidebar-footer p {
        margin: .75rem 0 0 !important;
        font-size: .72rem;
    }

    /* Refined hero treatment: editorial typography, layered light and a quiet orbit. */
    .quant-hero {
        isolation: isolate;
        border-color: rgba(196, 232, 237, .24);
        background:
            linear-gradient(135deg, rgba(27, 13, 28, .92), rgba(8, 27, 46, .84) 54%, rgba(9, 58, 70, .74)),
            radial-gradient(circle at 12% 8%, rgba(255, 218, 203, .12), transparent 28%),
            radial-gradient(circle at 84% 18%, rgba(112, 216, 229, .24), transparent 38%);
        box-shadow:
            0 35px 100px rgba(0, 0, 0, .46),
            0 0 0 1px rgba(112, 216, 229, .04),
            inset 0 1px rgba(255, 255, 255, .09);
    }

    .quant-hero::before {
        width: 72%;
        left: 34%;
        opacity: .9;
        filter: blur(34px);
    }

    .quant-hero::after {
        width: 520px;
        height: 520px;
        top: -300px;
        right: -150px;
        background: radial-gradient(circle, rgba(91, 142, 232, .38), transparent 68%);
    }

    .hero-brand {
        font-size: clamp(1.9rem, 3.1vw, 2.65rem);
        text-shadow: 0 8px 28px rgba(255, 196, 181, .16);
    }

    .hero-kicker {
        width: fit-content;
        padding: .36rem .7rem;
        border: 1px solid rgba(112, 216, 229, .18);
        border-radius: 999px;
        background: rgba(9, 34, 48, .42);
        box-shadow: inset 0 1px rgba(255, 255, 255, .06);
    }

    .hero-title {
        margin-top: .9rem;
        font-family: "STKaiti", "KaiTi", "FZKai-Z03", "Kaiti SC", "Instrument Serif", serif !important;
        font-size: clamp(3.45rem, 7.6vw, 7.35rem);
        font-weight: 700;
        letter-spacing: .01em;
        line-height: 1.02;
        text-shadow: 0 0 42px rgba(255, 255, 255, .2), 0 0 90px rgba(112, 216, 229, .16);
    }

    .hero-subtitle {
        max-width: 790px;
        font-size: 1.04rem;
        letter-spacing: .02em;
    }

    .hero-pills span {
        position: relative;
        overflow: hidden;
    }

    .hero-pills span::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 25%, rgba(255,255,255,.14) 48%, transparent 70%);
        transform: translateX(-120%);
        transition: transform .7s ease;
    }

    .hero-pills span:hover::after {
        transform: translateX(120%);
    }

    .hero-orbit {
        position: absolute;
        z-index: 0;
        right: 9%;
        top: 22%;
        width: 190px;
        height: 190px;
        border: 1px solid rgba(140, 228, 232, .16);
        border-radius: 50%;
        transform: rotate(-24deg);
        animation: heroOrbit 16s linear infinite;
        pointer-events: none;
    }

    .hero-orbit::before,
    .hero-orbit::after {
        content: "";
        position: absolute;
        border-radius: 50%;
    }

    .hero-orbit::before {
        inset: 23px;
        border: 1px dashed rgba(255, 196, 181, .2);
    }

    .hero-orbit::after {
        width: 7px;
        height: 7px;
        top: -3px;
        left: 50%;
        background: #8cecff;
        box-shadow: 0 0 18px rgba(140, 236, 255, .8);
    }

    .hero-shooting-star {
        position: absolute;
        z-index: 1;
        width: 150px;
        height: 2px;
        border-radius: 999px;
        background: linear-gradient(90deg, transparent 0%, rgba(140, 236, 255, .18) 36%, #d8fbff 80%, transparent 100%);
        box-shadow: 0 0 12px rgba(140, 236, 255, .72);
        opacity: 0;
        transform: rotate(-35deg);
        pointer-events: none;
        animation: shootingStar 8.5s cubic-bezier(.25, .1, .2, 1) infinite;
    }

    .hero-shooting-star::after {
        content: "";
        position: absolute;
        left: 13px;
        top: 50%;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #efffff;
        box-shadow: 0 0 14px 3px rgba(140, 236, 255, .78);
        transform: translateY(-50%);
    }

    .hero-shooting-star.star-one {
        top: 18%;
        left: 72%;
        animation-delay: -1.2s;
    }

    .hero-shooting-star.star-two {
        top: 35%;
        left: 82%;
        width: 112px;
        animation-delay: -5.1s;
        animation-duration: 10.5s;
    }

    .hero-shooting-star.star-three {
        top: 12%;
        left: 49%;
        width: 92px;
        animation-delay: -7.4s;
        animation-duration: 12s;
        opacity: .7;
    }

    .quant-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        min-height: min(64vh, 680px);
        flex-direction: column;
        justify-content: center;
        padding: 4.3rem 5.5rem 4rem;
        margin: .3rem 0 1.25rem;
        border: 1px solid rgba(196, 232, 237, .18);
        border-radius: 32px;
        background:
            linear-gradient(135deg, rgba(25, 12, 24, .92), rgba(8, 25, 39, .8) 55%, rgba(10, 55, 67, .72)),
            radial-gradient(circle at 80% 20%, rgba(112, 216, 229, .2), transparent 35%);
        box-shadow:
            0 35px 100px rgba(0, 0, 0, .43),
            inset 0 1px rgba(255, 255, 255, .06);
        animation: quantFadeUp .75s cubic-bezier(.2,.8,.2,1) both;
    }

    .quant-hero::before {
        content: "";
        position: absolute;
        width: 60%;
        height: 130%;
        left: 42%;
        top: -32%;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(255, 196, 181, .13), rgba(66, 193, 205, .16), transparent 65%);
        filter: blur(30px);
        transform: rotate(-22deg);
        animation: quantAurora 13s ease-in-out infinite alternate;
        pointer-events: none;
    }

    .quant-hero::after {
        content: "";
        position: absolute;
        width: 430px;
        height: 430px;
        top: -250px;
        right: -120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(91, 142, 232, .32), transparent 68%);
        animation: quantFloat 9s ease-in-out infinite;
        pointer-events: none;
    }

    .hero-brand {
        position: relative;
        z-index: 1;
        margin-bottom: .45rem;
        color: rgba(255, 244, 237, .95);
        font-family: 'Dancing Script', cursive !important;
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        font-weight: 600;
        letter-spacing: .01em;
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
        max-width: 980px;
        margin: .72rem 0 .62rem;
        color: #f8fbff;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: clamp(3.25rem, 7.4vw, 7.2rem);
        font-weight: 400;
        line-height: .93;
        letter-spacing: -.045em;
        text-shadow: 0 0 40px rgba(255, 255, 255, .25), 0 0 90px rgba(112, 216, 229, .12);
        background: linear-gradient(100deg, #fff8f4 4%, #d9e6f2 53%, #8ce4e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0;
        color: rgba(227, 238, 245, .72);
        font-size: 1.02rem;
        font-weight: 300;
        line-height: 1.8;
    }

    .hero-quote {
        position: relative;
        z-index: 1;
        max-width: 700px;
        margin: 1.1rem 0 0;
        color: rgba(255, 241, 234, .82);
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: clamp(1.15rem, 2vw, 1.55rem);
        font-style: italic;
        line-height: 1.25;
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
        box-shadow: inset 0 1px rgba(255, 255, 255, .08);
        transition: transform .3s ease, background .3s ease, border-color .3s ease;
    }

    .hero-pills span:hover {
        transform: translateY(-3px);
        border-color: rgba(176, 229, 232, .34);
        background: rgba(28, 61, 74, .65);
    }

    .hero-pills b {
        color: #f0f8ff;
        font-weight: 700;
    }

    .hero-sound {
        position: absolute;
        z-index: 1;
        right: 2.4rem;
        bottom: 2rem;
        display: flex;
        align-items: center;
        gap: .65rem;
        color: rgba(227, 238, 245, .55);
        font-size: .62rem;
        font-weight: 600;
        letter-spacing: .14em;
        line-height: 1.45;
        text-transform: uppercase;
    }

    .hero-sound-bars {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 30px;
        padding: 0 .6rem;
        border: 1px solid rgba(255, 255, 255, .22);
        border-radius: 999px;
        background: rgba(255, 255, 255, .03);
    }

    .hero-sound-bars i {
        display: block;
        width: 2px;
        border-radius: 999px;
        background: #9be4e7;
        animation: soundBeat 1.35s ease-in-out infinite alternate;
    }

    .hero-sound-bars i:nth-child(1) { height: 8px; animation-delay: -.2s; }
    .hero-sound-bars i:nth-child(2) { height: 16px; animation-delay: -.45s; }
    .hero-sound-bars i:nth-child(3) { height: 11px; animation-delay: -.75s; }
    .hero-sound-bars i:nth-child(4) { height: 20px; animation-delay: -.95s; }

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

    .candidate-table-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .8rem;
        margin: 1.15rem 0 .55rem;
        color: #d9e9f8;
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .candidate-table-label span {
        display: inline-flex;
        align-items: center;
        gap: .42rem;
        color: #91a8c4;
        font-size: .7rem;
        font-weight: 600;
        letter-spacing: .02em;
        text-transform: none;
    }

    .candidate-table-label i {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px rgba(52, 211, 153, .55);
    }

    [data-testid="stDataFrame"] {
        background:
            linear-gradient(145deg, rgba(12, 30, 51, .84), rgba(7, 16, 30, .84)),
            radial-gradient(circle at 0 0, rgba(112, 216, 229, .08), transparent 30%);
    }

    [data-testid="stDataFrame"] .stDataFrameGlideDataEditor {
        border-radius: 14px;
    }

    .result-analysis {
        position: relative;
        overflow: hidden;
        margin: 1.35rem 0 .4rem;
        padding: 1.45rem;
        border: 1px solid rgba(45, 212, 191, .2);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(9, 31, 48, .91), rgba(8, 18, 34, .9)),
            radial-gradient(circle at 100% 0, rgba(34, 211, 238, .16), transparent 38%);
        box-shadow: 0 16px 42px rgba(0, 0, 0, .24), inset 0 1px rgba(255, 255, 255, .04);
        animation: quantFadeUp .68s cubic-bezier(.2,.8,.2,1) both;
    }

    .result-analysis::before {
        content: "";
        position: absolute;
        top: 0;
        left: 8%;
        right: 8%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(103, 232, 249, .72), transparent);
    }

    .analysis-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.1rem;
    }

    .analysis-eyebrow {
        display: block;
        color: #5eead4;
        font-size: .7rem;
        font-weight: 760;
        letter-spacing: .15em;
    }

    .analysis-header h4 {
        margin: .26rem 0 0;
        color: #f0f9ff;
        font-size: 1.24rem;
    }

    .analysis-badge {
        flex: 0 0 auto;
        padding: .42rem .72rem;
        border: 1px solid rgba(94, 234, 212, .22);
        border-radius: 999px;
        color: #a7f3d0;
        background: rgba(13, 148, 136, .12);
        font-size: .76rem;
        font-weight: 700;
    }

    .analysis-badge.cautious {
        color: #fde68a;
        border-color: rgba(251, 191, 36, .22);
        background: rgba(217, 119, 6, .12);
    }

    .analysis-badge.weak {
        color: #fecdd3;
        border-color: rgba(251, 113, 133, .24);
        background: rgba(225, 29, 72, .12);
    }

    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .75rem;
    }

    .analysis-item {
        padding: .9rem 1rem;
        border: 1px solid rgba(125, 211, 252, .1);
        border-radius: 14px;
        background: rgba(4, 14, 28, .56);
        transition: transform .22s ease, border-color .22s ease;
    }

    .analysis-item:hover {
        transform: translateY(-2px);
        border-color: rgba(45, 212, 191, .28);
    }

    .analysis-item span,
    .analysis-item small {
        display: block;
        color: #8298b2;
        font-size: .74rem;
    }

    .analysis-item strong {
        display: block;
        margin: .22rem 0 .18rem;
        color: #f0f9ff;
        font-size: 1.22rem;
    }

    .analysis-narrative {
        margin: 1rem .1rem .75rem !important;
        color: #b9c9dc !important;
        font-size: .9rem;
        line-height: 1.75;
    }

    .risk-note {
        padding: .76rem .9rem;
        border-left: 3px solid #fb7185;
        border-radius: 0 10px 10px 0;
        background: rgba(136, 19, 55, .1);
        color: #d4b6c1;
        font-size: .78rem;
        line-height: 1.6;
    }

    .screening-console {
        margin: .75rem 0 1.35rem;
        padding: 1.3rem;
        border: 1px solid rgba(96, 165, 250, .18);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(10, 25, 46, .9), rgba(6, 16, 31, .9)),
            radial-gradient(circle at 0 0, rgba(37, 99, 235, .14), transparent 34%);
        box-shadow: 0 16px 42px rgba(0, 0, 0, .22);
        animation: quantFadeUp .65s cubic-bezier(.2,.8,.2,1) both;
    }

    .screening-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .screening-head h4 {
        margin: .22rem 0 0;
        color: #f0f7ff;
        font-size: 1.2rem;
    }

    .screening-date {
        padding: .38rem .66rem;
        border: 1px solid rgba(125, 211, 252, .14);
        border-radius: 999px;
        color: #9fb5ce;
        background: rgba(5, 15, 29, .55);
        font-size: .74rem;
    }

    .screening-flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .65rem;
    }

    .screening-step {
        position: relative;
        min-height: 104px;
        padding: .82rem .9rem;
        border: 1px solid rgba(125, 211, 252, .1);
        border-radius: 13px;
        background: rgba(3, 12, 25, .58);
    }

    .screening-step:not(:last-child)::after {
        content: "›";
        position: absolute;
        top: 34%;
        right: -.52rem;
        z-index: 2;
        color: #38bdf8;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .screening-step span,
    .screening-step small {
        display: block;
        color: #8197b1;
        font-size: .71rem;
        line-height: 1.5;
    }

    .screening-step b {
        display: block;
        margin: .18rem 0;
        color: #edf7ff;
        font-size: 1.38rem;
    }

    .screening-step.final {
        border-color: rgba(52, 211, 153, .25);
        background: rgba(6, 78, 59, .11);
    }

    .screening-summary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .65rem;
        margin-top: .8rem;
    }

    .screening-summary div {
        padding: .72rem .88rem;
        border-radius: 11px;
        background: rgba(15, 31, 51, .7);
        color: #8fa4bc;
        font-size: .74rem;
    }

    .screening-summary strong {
        display: block;
        margin-top: .16rem;
        color: #dffaf4;
        font-size: 1rem;
    }

    .screening-rule {
        margin: .8rem .05rem 0 !important;
        color: #8ea3bc !important;
        font-size: .76rem;
        line-height: 1.65;
    }

    .feature-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: .58rem;
        min-height: 112px;
        padding: 1rem;
        border: 1px solid rgba(125, 211, 252, .13);
        border-radius: 16px;
        background:
            linear-gradient(145deg, rgba(10, 26, 46, .86), rgba(6, 15, 28, .78)),
            radial-gradient(circle at 100% 0, rgba(112, 216, 229, .14), transparent 42%);
        box-shadow: inset 0 1px rgba(255, 255, 255, .05);
    }

    .feature-chip {
        display: inline-flex;
        align-items: center;
        gap: .46rem;
        padding: .48rem .7rem .48rem .48rem;
        border: 1px solid rgba(155, 131, 216, .18);
        border-radius: 999px;
        background: rgba(19, 31, 53, .72);
        color: #dbeafe;
        font-size: .78rem;
        line-height: 1;
        box-shadow: inset 0 1px rgba(255, 255, 255, .05);
        transition: transform .24s ease, border-color .24s ease, background .24s ease;
        animation: quantFadeUp .45s cubic-bezier(.2,.8,.2,1) both;
    }

    .feature-chip:hover {
        transform: translateY(-3px) scale(1.02);
        border-color: rgba(112, 216, 229, .5);
        background: rgba(27, 62, 76, .8);
    }

    .feature-chip-index {
        display: inline-grid;
        width: 1.3rem;
        height: 1.3rem;
        place-items: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #6386e8, #70d8e5);
        color: #07111e;
        font-size: .64rem;
        font-weight: 800;
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

    @keyframes quantAurora {
        0% { transform: translate3d(-5%, 2%, 0) rotate(-22deg) scale(1); opacity: .64; }
        100% { transform: translate3d(11%, -4%, 0) rotate(-14deg) scale(1.12); opacity: .95; }
    }

    @keyframes soundBeat {
        0% { transform: scaleY(.6); opacity: .55; }
        100% { transform: scaleY(1.15); opacity: 1; }
    }

    @keyframes heroOrbit {
        from { transform: rotate(-24deg) translate3d(0, 0, 0); }
        50% { transform: rotate(156deg) translate3d(0, -8px, 0); }
        to { transform: rotate(336deg) translate3d(0, 0, 0); }
    }

    @keyframes shootingStar {
        0%, 58% {
            opacity: 0;
            transform: translate3d(210px, -130px, 0) rotate(-35deg) scaleX(.38);
        }
        63% {
            opacity: .9;
        }
        72% {
            opacity: 0;
            transform: translate3d(-220px, 180px, 0) rotate(-35deg) scaleX(1);
        }
        100% {
            opacity: 0;
            transform: translate3d(-220px, 180px, 0) rotate(-35deg) scaleX(1);
        }
    }

    @media (max-width: 820px) {
        [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
        .sidebar-brand strong {
            white-space: nowrap;
            font-size: 1.45rem;
            letter-spacing: .03em;
        }
        .sidebar-brand span { font-size: .74rem; }
        .hero-orbit { right: -5%; top: 18%; opacity: .7; transform: scale(.78); }
        .quant-hero { min-height: 58vh; padding: 2.6rem 1.35rem 3.4rem; border-radius: 22px; }
        .hero-title { font-size: clamp(3rem, 14vw, 5rem); }
        .hero-pills { gap: .45rem; }
        .hero-sound { right: 1.4rem; bottom: 1.35rem; }
        .stTabs [data-baseweb="tab"] { padding: 0 .7rem; }
        .analysis-grid { grid-template-columns: 1fr; }
        .analysis-header { align-items: center; }
        .screening-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .screening-step:nth-child(2)::after { display: none; }
        .screening-summary { grid-template-columns: 1fr; }
    }

    /* Serene 小智: compact desktop-style floating assistant. */
    [data-testid="stPopover"] {
        position: fixed !important;
        right: 1.4rem;
        bottom: 1.35rem;
        width: 4.2rem !important;
        z-index: 1000;
    }

    [data-testid="stPopover"] [data-testid="stTooltipHoverTarget"] {
        width: 4.2rem !important;
        justify-content: center !important;
    }

    [data-testid="stPopoverButton"] {
        width: 4.2rem !important;
        height: 4.2rem !important;
        padding: 0 !important;
        border: 1px solid rgba(182, 247, 241, .72) !important;
        border-radius: 999px !important;
        color: #06222b !important;
        background:
            radial-gradient(circle at 32% 22%, #f5ffef 0%, #9cf3d1 24%, #70d8e5 58%, #6582e8 100%) !important;
        box-shadow:
            0 12px 34px rgba(0, 0, 0, .32),
            0 0 0 7px rgba(112, 216, 229, .10),
            0 0 30px rgba(112, 216, 229, .30) !important;
        font-size: 1.55rem !important;
        font-weight: 900 !important;
        transition: transform .22s ease, box-shadow .22s ease !important;
        cursor: grab !important;
        touch-action: none;
    }

    [data-testid="stPopoverButton"]:hover {
        transform: translateY(-3px) scale(1.04);
        box-shadow:
            0 16px 40px rgba(0, 0, 0, .38),
            0 0 0 9px rgba(112, 216, 229, .14),
            0 0 38px rgba(112, 216, 229, .42) !important;
    }

    [data-testid="stPopoverButton"].assistant-dragging {
        cursor: grabbing !important;
        transform: scale(1.03);
        box-shadow:
            0 18px 42px rgba(0, 0, 0, .42),
            0 0 0 10px rgba(156, 243, 209, .16),
            0 0 42px rgba(112, 216, 229, .44) !important;
    }

    [data-testid="stPopoverBody"] {
        width: min(420px, calc(100vw - 2rem)) !important;
        max-height: min(690px, calc(100vh - 7rem));
        overflow-y: auto;
        padding: 0 !important;
        border: 1px solid rgba(170, 230, 235, .23) !important;
        border-radius: 24px !important;
        background:
            linear-gradient(155deg, rgba(10, 30, 49, .97), rgba(7, 14, 29, .98)) !important;
        box-shadow:
            0 26px 80px rgba(0, 0, 0, .52),
            inset 0 1px rgba(255, 255, 255, .08) !important;
        backdrop-filter: blur(20px);
    }

    [data-testid="stPopoverBody"] > div > [data-testid="stVerticalBlock"] {
        padding: 0 .9rem 1rem !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
        gap: .6rem;
    }

    .assistant-popover-header {
        position: relative;
        overflow: hidden;
        margin: -.1rem -.9rem .5rem;
        padding: 1.1rem 1.15rem 1rem;
        border-bottom: 1px solid rgba(170, 230, 235, .15);
        background:
            radial-gradient(circle at 88% 12%, rgba(112, 216, 229, .26), transparent 9rem),
            linear-gradient(135deg, rgba(38, 24, 54, .72), rgba(7, 55, 67, .58));
    }

    .assistant-popover-header::after {
        content: "";
        position: absolute;
        width: 130px;
        height: 130px;
        right: -42px;
        bottom: -78px;
        border: 1px solid rgba(156, 243, 209, .25);
        border-radius: 50%;
        box-shadow: 0 0 0 12px rgba(156, 243, 209, .06), 0 0 0 24px rgba(156, 243, 209, .035);
    }

    .assistant-popover-kicker {
        display: flex;
        align-items: center;
        color: #9cf3d1;
        font-size: .64rem;
        font-weight: 800;
        letter-spacing: .15em;
    }

    .assistant-popover-kicker::before {
        content: "";
        display: inline-block;
        width: .42rem;
        height: .42rem;
        margin-right: .36rem;
        border-radius: 50%;
        background: #9cf3d1;
        box-shadow: 0 0 0 4px rgba(156, 243, 209, .12), 0 0 14px rgba(156, 243, 209, .56);
    }

    .assistant-popover-kicker i {
        display: inline-block;
        width: .42rem;
        height: .42rem;
        margin-right: .36rem;
        border-radius: 50%;
        background: #9cf3d1;
        box-shadow: 0 0 0 4px rgba(156, 243, 209, .12), 0 0 14px rgba(156, 243, 209, .56);
        vertical-align: .04rem;
    }

    .assistant-popover-title {
        margin-top: .28rem;
        color: #f3fbff;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: .02em;
    }

    .assistant-popover-subtitle {
        margin-top: .22rem;
        color: #9eb4cc;
        font-size: .74rem;
    }

    [data-testid="stPopoverBody"] [data-testid="stMetric"] {
        min-height: 0;
        padding: .58rem .62rem;
        border: 1px solid rgba(125, 211, 252, .12);
        border-radius: 14px;
        background: rgba(8, 24, 42, .62);
    }

    [data-testid="stPopoverBody"] [data-testid="stMetricLabel"] {
        color: #8fa9c2 !important;
        font-size: .64rem !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stMetricValue"] {
        color: #e9fbff !important;
        font-size: 1.02rem !important;
    }

    .assistant-quick-title {
        margin: .38rem 0 .02rem !important;
        padding-top: .18rem;
        color: #90a8c1;
        font-size: .61rem;
        font-weight: 700;
        line-height: 1.25;
        letter-spacing: .14em;
    }

    [data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(.assistant-quick-title) {
        min-height: 1.55rem !important;
        overflow: visible !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stChatMessage"] {
        margin: .12rem 0;
        padding: .62rem .7rem;
        border: 1px solid rgba(125, 211, 252, .10);
        border-radius: 15px;
        background: rgba(8, 22, 39, .60);
    }

    [data-testid="stPopoverBody"] [data-testid="stChatMessage"] p {
        color: #d9e9f5;
        font-size: .78rem;
        line-height: 1.6;
    }

    [data-testid="stPopoverBody"] [data-testid="stTextInput"] input {
        min-height: 2.6rem;
        border: 1px solid rgba(112, 216, 229, .22) !important;
        border-radius: 13px !important;
        color: #ecfbff !important;
        background: rgba(5, 17, 31, .72) !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stButton"] button {
        min-height: 2rem;
        padding: .18rem .42rem !important;
        border: 1px solid rgba(112, 216, 229, .17);
        border-radius: 11px;
        color: #d9f9f2;
        background:
            linear-gradient(135deg, rgba(35, 105, 224, .84), rgba(11, 151, 181, .86));
        font-size: .70rem;
        font-weight: 700;
        letter-spacing: .04em;
    }

    [data-testid="stPopoverBody"] [data-testid="stButton"] button:hover {
        border-color: rgba(156, 243, 209, .54);
        color: #ffffff;
        background: rgba(23, 77, 84, .75);
    }

    [data-testid="stPopoverBody"] [data-testid="stButton"] button:focus:not(:active) {
        border-color: rgba(112, 216, 229, .28);
        box-shadow: 0 0 0 2px rgba(112, 216, 229, .12);
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
def get_model(data_path: str, threshold: float, cache_version: str):
    del cache_version
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
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="sidebar-status"><i></i> LIVE ANALYTICS</div>
            <div class="sidebar-mini-grid">
                <div class="sidebar-mini"><span>路径规模</span><b>{simulation_paths:,}</b></div>
                <div class="sidebar-mini"><span>筛选阈值</span><b>{probability_threshold:.0%}</b></div>
            </div>
            <p>调整左侧参数，图表与模型结论会同步更新。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

bundle = get_bundle(str(data_path))
features = get_features(str(data_path))
single = prepare_single_stock(bundle.daily_prices, ticker)
single_summary = summarize_single_stock(single)
simulation = simulate_gbm(single, n_paths=simulation_paths)

st.markdown(
    f"""
    <section class="quant-hero">
        <div class="hero-orbit" aria-hidden="true"></div>
        <div class="hero-shooting-star star-one" aria-hidden="true"></div>
        <div class="hero-shooting-star star-two" aria-hidden="true"></div>
        <div class="hero-shooting-star star-three" aria-hidden="true"></div>
        <div class="hero-brand">Serene Quant</div>
        <div class="hero-kicker">
            <span class="status-dot"></span>
            QUANT INTELLIGENCE CONSOLE
        </div>
        <h1 class="hero-title">A股量化分析平台</h1>
        <p class="hero-subtitle">
            从历史行情、风险模拟到机器学习选股，以一套可交互的分析流程洞察市场信号。
        </p>
        <p class="hero-quote">“让数据保持克制，让判断更接近真实。”</p>
        <div class="hero-pills">
            <span>数据区间 <b>2021-01-04 — 2026-07-24</b></span>
            <span>覆盖股票 <b>{bundle.stock_info["Ticker"].nunique()} 只</b></span>
            <span>有效行情 <b>{len(bundle.daily_prices):,} 条</b></span>
            <span>当前标的 <b>{ticker}</b></span>
        </div>
        <div class="hero-sound">
            <span class="hero-sound-bars"><i></i><i></i><i></i><i></i></span>
            <span>Market<br>in motion</span>
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

    expected_return = (
        simulation.expected_terminal_price / simulation.current_price - 1
    )
    if expected_return >= 0.05 and simulation.upside_probability >= 0.55:
        outlook_label = "预期偏正向"
        outlook_class = ""
        outlook_text = (
            "模拟终值的均值高于当前价格，且多数路径收于当前价之上，"
            "模型分布整体呈现偏正向预期。"
        )
    elif expected_return >= 0:
        outlook_label = "中性偏谨慎"
        outlook_class = "cautious"
        outlook_text = (
            "模拟均值略高于当前价格，但上涨优势不强，"
            "预期收益与价格波动之间仍需保持谨慎权衡。"
        )
    else:
        outlook_label = "预期偏弱"
        outlook_class = "weak"
        outlook_text = (
            "模拟终值的均值低于当前价格，模型分布反映出偏弱预期，"
            "应重点关注价格继续下行的可能性。"
        )

    st.markdown(
        f"""
        <section class="result-analysis">
            <div class="analysis-header">
                <div>
                    <span class="analysis-eyebrow">SIMULATION INSIGHT</span>
                    <h4>模拟结果分析</h4>
                </div>
                <span class="analysis-badge {outlook_class}">{outlook_label}</span>
            </div>
            <div class="analysis-grid">
                <div class="analysis-item">
                    <span>一年后预期价格</span>
                    <strong>{simulation.expected_terminal_price:.2f}</strong>
                    <small>相对当前价预期收益 {expected_return:+.1%}</small>
                </div>
                <div class="analysis-item">
                    <span>上涨路径占比</span>
                    <strong>{simulation.upside_probability:.1%}</strong>
                    <small>终值高于当前价的模拟概率</small>
                </div>
                <div class="analysis-item">
                    <span>99%下行风险分位</span>
                    <strong>{simulation.lower_1pct_price:.2f}</strong>
                    <small>较当前价可能下跌 {simulation.var_pct_99:.1%}</small>
                </div>
            </div>
            <p class="analysis-narrative">
                基于 {simulation_paths:,} 条几何布朗运动模拟路径，{outlook_text}
                在极端下行情景下，终值价格的下侧 1% 分位为
                {simulation.lower_1pct_price:.2f}，对应每股较当前价格减少
                {simulation.var_amount_99:.2f}。
            </p>
            <div class="risk-note">
                风险提示：模拟结果依赖历史收益与波动率，并不代表确定预测；
                99%风险分位表示约有1%的模拟终值可能低于该价格，实际市场风险可能更高。
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

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
        model_result = get_model(
            str(data_path),
            probability_threshold,
            MODEL_CACHE_VERSION,
        )
    metric = model_result.metrics[model_result.primary_name]["test"]
    screening = model_result.screening_summary
    average_expected = screening["average_expected_5d_return"]
    average_probability = screening["average_probability_up_5d"]
    maximum_expected = screening["maximum_expected_5d_return"]
    average_expected_text = (
        f"{average_expected:.2%}" if average_expected is not None else "暂无"
    )
    average_probability_text = (
        f"{average_probability:.1%}" if average_probability is not None else "暂无"
    )
    maximum_expected_text = (
        f"{maximum_expected:.2%}" if maximum_expected is not None else "暂无"
    )
    st.markdown(
        f"""
        <section class="screening-console">
            <div class="screening-head">
                <div>
                    <span class="analysis-eyebrow">LIVE SCREENING PIPELINE</span>
                    <h4>实盘筛选逻辑</h4>
                </div>
                <span class="screening-date">筛选日期 {screening["selection_date"]}</span>
            </div>
            <div class="screening-flow">
                <div class="screening-step">
                    <span>STEP 01 · 最新可用行情</span>
                    <b>{screening["stock_pool"]}</b>
                    <small>进入当日筛选股票池</small>
                </div>
                <div class="screening-step">
                    <span>STEP 02 · 模型概率达标</span>
                    <b>{screening["probability_pass"]}</b>
                    <small>上涨概率 ≥ {screening["probability_threshold"]:.0%}</small>
                </div>
                <div class="screening-step">
                    <span>STEP 03 · 排除当日涨停</span>
                    <b>{screening["limit_up_filtered"]}</b>
                    <small>本次剔除的涨停标的数量</small>
                </div>
                <div class="screening-step final">
                    <span>STEP 04 · 有效选股</span>
                    <b>{screening["effective_candidates"]}</b>
                    <small>输出可买入候选清单</small>
                </div>
            </div>
            <div class="screening-summary">
                <div>候选平均上涨概率<strong>{average_probability_text}</strong></div>
                <div>候选平均预期5日收益<strong>{average_expected_text}</strong></div>
                <div>最高预期5日收益<strong>{maximum_expected_text}</strong></div>
            </div>
            <p class="screening-rule">
                过滤规则：主板日涨幅达到约 9.5%、创业板达到约 19.5%、
                ST 标的达到约 4.5% 时标记为涨停并排除；阈值采用近似值，
                用于处理价格四舍五入误差。
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(model_metrics_figure(metric), width="stretch")
    with right:
        st.write("时间切分")
        st.json(model_result.split_dates)
        st.write("模型特征")
        feature_chips = "".join(
            f'<span class="feature-chip" style="animation-delay:{index * 55}ms">'
            f'<span class="feature-chip-index">{index + 1:02d}</span>'
            f'{escape(FEATURE_LABELS.get(name, name))}</span>'
            for index, name in enumerate(model_result.feature_columns)
        )
        st.markdown(
            f'<div class="feature-cloud">{feature_chips}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"主模型：**{model_result.primary_name}** · "
        f"测试集灵敏度：**{metric['sensitivity_recall']:.1%}** · "
        f"有效候选数量：**{len(model_result.candidates)}**"
    )
    candidate_display = model_result.candidates.copy()
    candidate_display["交易状态"] = "可买入"
    candidate_display["Market"] = candidate_display["Market"].replace(
        {"Shanghai": "上海证券交易所", "Shenzhen": "深圳证券交易所"}
    )
    candidate_display["Board"] = candidate_display["Board"].replace(
        {"Main Board": "主板", "ChiNext": "创业板"}
    )
    candidate_display["Industry"] = candidate_display["Industry"].replace(
        CATEGORY_TRANSLATIONS["Industry"]
    )
    candidate_display = candidate_display.rename(
        columns={
            "Ticker": "股票代码",
            "StockNameCN": "股票名称",
            "Date": "筛选日期",
            "Market": "交易所",
            "Board": "板块",
            "Industry": "行业",
            "Close": "收盘价",
            "ProbabilityUp5D": "未来5日上涨概率",
            "Expected5DReturn": "预期5日收益",
            "LimitUpFlagApprox": "涨停标记",
        }
    )
    download_candidates = candidate_display.copy()
    table_display = candidate_display.copy()
    table_display["未来5日上涨概率"] = table_display["未来5日上涨概率"] * 100
    table_display["预期5日收益"] = table_display["预期5日收益"] * 100
    st.markdown(
        """
        <div class="candidate-table-label">
            <span><i></i>可买入候选清单</span>
            <span>模型概率与预期收益</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        table_display.style.format(
            {
                "未来5日上涨概率": "{:.1f}%",
                "预期5日收益": "{:.1f}%",
                "收盘价": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "股票代码": st.column_config.TextColumn("股票代码", width="small"),
            "股票名称": st.column_config.TextColumn("股票名称", width="small"),
            "筛选日期": st.column_config.DatetimeColumn("筛选日期", format="YYYY-MM-DD"),
            "收盘价": st.column_config.NumberColumn("收盘价", format="%.2f"),
            "未来5日上涨概率": st.column_config.ProgressColumn(
                "未来5日上涨概率",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "预期5日收益": st.column_config.NumberColumn("预期5日收益", format="%.1f%%"),
        },
    )
    st.download_button(
        "下载候选股票清单",
        data=download_candidates.to_csv(index=False).encode("utf-8-sig"),
        file_name="candidate_stocks.csv",
        mime="text/csv",
    )
    st.subheader("特征重要性")
    st.plotly_chart(
        feature_importance_figure(model_result.feature_importance),
        width="stretch",
    )

_LEGACY_INLINE_ASSISTANT = r"""
Legacy inline assistant layout retained as a comment so the floating popover
below is the only visible assistant surface.
st.divider()
st.subheader(f"{ASSISTANT_NAME} · AI 股票分析助手")
st.caption(
    "助手只使用当前项目中的历史行情、蒙特卡洛模拟和随机森林结果，"
    "不调用实时新闻，也不构成投资建议。"
)

assistant_context = build_assistant_context(
    ticker=ticker,
    single=single,
    single_summary=single_summary,
    simulation=simulation,
    features=features,
    model_result=model_result,
)

assistant_metrics = st.columns(4)
assistant_metrics[0].metric("当前标的", assistant_context.ticker)
assistant_metrics[1].metric("最新收盘价", f"{assistant_context.latest_close:.2f}")
assistant_metrics[2].metric("模型上涨概率", (
    f"{assistant_context.probability_up_5d:.1%}"
    if assistant_context.probability_up_5d is not None
    else "暂无"
))
assistant_metrics[3].metric("风险等级", risk_label(
    assistant_context.annualized_volatility,
    assistant_context.max_drawdown,
))

if st.session_state.get("stock_assistant_ticker") != ticker:
    st.session_state.stock_assistant_ticker = ticker
    st.session_state.stock_assistant_messages = []

if not st.session_state.stock_assistant_messages:
    st.session_state.stock_assistant_messages = [
        {
            "role": "assistant",
            "content": (
                f"你好，我是 {ASSISTANT_NAME}，正在为你分析 {assistant_context.stock_name}。"
                "你可以问我当前趋势、风险、随机森林概率或候选股票池。"
            ),
        }
    ]

for message in st.session_state.stock_assistant_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

assistant_prompt = st.chat_input(
    "例如：请分析当前股票风险，或解释随机森林的预测结果",
    key="stock_assistant_prompt",
)
if assistant_prompt:
    st.session_state.stock_assistant_messages.append(
        {"role": "user", "content": assistant_prompt}
    )
    assistant_reply = answer_query(assistant_prompt, assistant_context)
    st.session_state.stock_assistant_messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    st.rerun()

"""

assistant_context = build_assistant_context(
    ticker=ticker,
    single=single,
    single_summary=single_summary,
    simulation=simulation,
    features=features,
    model_result=model_result,
)

if st.session_state.get("stock_assistant_ticker") != ticker:
    st.session_state.stock_assistant_ticker = ticker
    st.session_state.stock_assistant_messages = []

if not st.session_state.stock_assistant_messages:
    st.session_state.stock_assistant_messages = [
        {
            "role": "assistant",
            "content": (
                f"你好，我是 {ASSISTANT_NAME}，正在为你分析 {assistant_context.stock_name}。"
                "你可以问我当前趋势、风险、随机森林概率或候选股票池。"
            ),
        }
    ]

with st.popover("✦", help=f"打开 {ASSISTANT_NAME}", use_container_width=False):
    st.markdown(
        f"""
        <div class="assistant-popover-header">
            <div class="assistant-popover-kicker">QUANT COMPANION · LIVE</div>
            <div class="assistant-popover-title">{ASSISTANT_NAME}</div>
            <div class="assistant-popover-subtitle">
                {assistant_context.stock_name} · {assistant_context.ticker} · 数据截至 {assistant_context.latest_date}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    assistant_metrics = st.columns(2)
    assistant_metrics[0].metric("最新收盘", f"{assistant_context.latest_close:.2f}")
    assistant_metrics[1].metric(
        "上涨概率",
        (
            f"{assistant_context.probability_up_5d:.1%}"
            if assistant_context.probability_up_5d is not None
            else "暂无"
        ),
    )
    st.markdown('<div class="assistant-quick-title">QUICK PROMPTS</div>', unsafe_allow_html=True)
    quick_columns = st.columns(2)
    quick_questions = [
        ("风险扫描", "请分析当前股票风险"),
        ("模型解读", "随机森林预测结果如何？"),
        ("趋势概览", "请分析当前股票走势"),
        ("候选股票", "当前股票池有哪些候选股票？"),
    ]
    for column, (label, question) in zip(quick_columns * 2, quick_questions):
        if column.button(label, key=f"assistant_quick_{label}", use_container_width=True):
            st.session_state.stock_assistant_messages.append(
                {"role": "user", "content": question}
            )
            st.session_state.stock_assistant_messages.append(
                {"role": "assistant", "content": answer_query(question, assistant_context)}
            )
            st.rerun()

    for message in st.session_state.stock_assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("stock_assistant_form", clear_on_submit=True):
        assistant_prompt = st.text_input(
            "向助手提问",
            placeholder="例如：这只股票的风险如何？",
            label_visibility="collapsed",
        )
        assistant_submitted = st.form_submit_button("发送", use_container_width=True)
    if assistant_submitted and assistant_prompt.strip():
        st.session_state.stock_assistant_messages.append(
            {"role": "user", "content": assistant_prompt.strip()}
        )
        st.session_state.stock_assistant_messages.append(
            {"role": "assistant", "content": answer_query(assistant_prompt, assistant_context)}
        )
        st.rerun()

components.html(
    """
    <script>
    (() => {
      const STORAGE_KEY = "serene-quant-assistant-position-v1";
      const hostWindow = window.parent;
      const hostDocument = hostWindow.document;
      let dragState = null;
      let suppressClick = false;

      const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

      const applySavedPosition = (shell) => {
        try {
          const saved = JSON.parse(hostWindow.localStorage.getItem(STORAGE_KEY) || "null");
          if (!saved || !Number.isFinite(saved.left) || !Number.isFinite(saved.top)) return;
          const maxLeft = Math.max(8, hostWindow.innerWidth - shell.offsetWidth - 8);
          const maxTop = Math.max(8, hostWindow.innerHeight - shell.offsetHeight - 8);
          shell.style.setProperty("left", `${clamp(saved.left, 8, maxLeft)}px`, "important");
          shell.style.setProperty("top", `${clamp(saved.top, 8, maxTop)}px`, "important");
          shell.style.setProperty("right", "auto", "important");
          shell.style.setProperty("bottom", "auto", "important");
        } catch (_) {}
      };

      const bind = () => {
        const shell = hostDocument.querySelector('[data-testid="stPopover"]');
        const button = hostDocument.querySelector('[data-testid="stPopoverButton"]');
        if (!shell || !button) return;
        applySavedPosition(shell);
        if (button.dataset.dragBound === "true") return;
        button.dataset.dragBound = "true";

        button.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          const rect = shell.getBoundingClientRect();
          dragState = {
            startX: event.clientX,
            startY: event.clientY,
            left: rect.left,
            top: rect.top,
            moved: false
          };
          button.setPointerCapture?.(event.pointerId);
          event.preventDefault();
        });

        button.addEventListener("pointermove", (event) => {
          if (!dragState) return;
          const dx = event.clientX - dragState.startX;
          const dy = event.clientY - dragState.startY;
          if (Math.abs(dx) + Math.abs(dy) > 4) {
            dragState.moved = true;
            button.classList.add("assistant-dragging");
            shell.style.setProperty("left", `${clamp(
              dragState.left + dx, 8, hostWindow.innerWidth - shell.offsetWidth - 8
            )}px`, "important");
            shell.style.setProperty("top", `${clamp(
              dragState.top + dy, 8, hostWindow.innerHeight - shell.offsetHeight - 8
            )}px`, "important");
            shell.style.setProperty("right", "auto", "important");
            shell.style.setProperty("bottom", "auto", "important");
          }
        });

        const finishDrag = () => {
          if (!dragState) return;
          const rect = shell.getBoundingClientRect();
          if (dragState.moved) {
            suppressClick = true;
            hostWindow.setTimeout(() => { suppressClick = false; }, 220);
            try {
              hostWindow.localStorage.setItem(
                STORAGE_KEY, JSON.stringify({left: rect.left, top: rect.top})
              );
            } catch (_) {}
          }
          dragState = null;
          button.classList.remove("assistant-dragging");
        };

        button.addEventListener("pointerup", finishDrag);
        button.addEventListener("pointercancel", finishDrag);
        button.addEventListener("click", (event) => {
          if (!suppressClick) return;
          event.preventDefault();
          event.stopImmediatePropagation();
        }, true);
      };

      new hostWindow.MutationObserver(bind).observe(hostDocument.body, {
        childList: true,
        subtree: true
      });
      bind();
    })();
    </script>
    """,
    height=0,
    scrolling=False,
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
