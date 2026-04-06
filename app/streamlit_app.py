#!/usr/bin/env python3
"""
🏪 Rossmann Sales Forecasting Dashboard
Production-grade Streamlit app with real data loading and interactive charts.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# ── Add src/ to path for Config import ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import Config

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1e1e35;
    }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* ── Main background ── */
    .stApp { background: #0b0b16; }

    /* ── Metric cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #13132a 0%, #1a1a35 100%);
        border: 1px solid #2a2a50;
        border-radius: 16px;
        padding: 22px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 6px 0 2px 0;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #888aaa !important;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #555780 !important;
        margin-top: 4px;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #c5c8f0;
        border-left: 3px solid #667eea;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #13132a 0%, #1e1040 50%, #13132a 100%);
        border: 1px solid #2a2a55;
        border-radius: 20px;
        padding: 42px 40px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8890c0;
        margin-bottom: 0;
    }
    .hero-accent { color: #667eea; font-weight: 600; }

    /* ── Prediction result box ── */
    .pred-box {
        background: linear-gradient(135deg, #1a1060 0%, #2a1580 100%);
        border: 1px solid #4040a0;
        border-radius: 20px;
        padding: 36px 30px;
        text-align: center;
    }
    .pred-amount {
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        margin: 8px 0;
    }
    .pred-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.55);
    }
    .pred-ci {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.4);
        margin-top: 6px;
        font-family: 'DM Mono', monospace;
    }

    /* ── Insight chips ── */
    .insight-chip {
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #c5c8f0;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        background: rgba(46,204,113,0.15);
        border: 1px solid rgba(46,204,113,0.4);
        color: #2ecc71;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #13132a;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #888aaa;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b0b16; }
    ::-webkit-scrollbar-thumb { background: #2a2a50; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CHART DEFAULTS
# ============================================================
CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,13,30,0)",
    plot_bgcolor="rgba(13,13,30,0)",
    font=dict(family="DM Sans, sans-serif", color="#c5c8f0"),
    margin=dict(l=20, r=20, t=50, b=20),
)

ACCENT = "#667eea"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
ORANGE = "#f39c12"

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_config():
    config = Config()
    model  = joblib.load(config.XGB_MODEL)
    return model, config


@st.cache_data(show_spinner=False)
def load_processed_data(_config):
    X_test = pd.read_csv(_config.X_TEST)
    y_test = pd.read_csv(_config.Y_TEST).squeeze()
    return X_test, y_test


@st.cache_data(show_spinner=False)
def load_raw_data(_config):
    df = pd.read_csv(_config.CLEANED_DATA, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def compute_predictions(_model, X_test):
    return _model.predict(X_test)


# ── Bootstrap ────────────────────────────────────────────────
try:
    model, config = load_model_and_config()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(
        f"❌ Model not found. Run `python main.py --stage all` first.\n\n`{e}`"
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:24px 0 8px 0; text-align:center;">
            <div style="font-size:2.2rem;">🏪</div>
            <div style="font-size:1.15rem; font-weight:700; color:#e0e0f0; margin-top:4px;">Rossmann</div>
            <div style="font-size:0.78rem; color:#555780; letter-spacing:1px; text-transform:uppercase;">Sales Forecasting</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MODEL_OK:
        st.markdown(
            '<div style="text-align:center; margin-bottom:20px;">'
            '<span class="status-badge">🟢 XGBoost Active</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Predict", "📊  Analytics", "ℹ️   Model Info"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="padding:14px; background:#13132a; border-radius:12px; border:1px solid #1e1e35;">
            <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; color:#555780; margin-bottom:8px;">Model Metrics</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#888aaa;">RMSE</span>
                <span style="font-size:0.8rem; font-weight:600; color:#667eea;">€ 1,866</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#888aaa;">MAE</span>
                <span style="font-size:0.8rem; font-weight:600; color:#667eea;">€ 1,353</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.8rem; color:#888aaa;">R²</span>
                <span style="font-size:0.8rem; font-weight:600; color:#2ecc71;">0.622</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ① HOME PAGE
# ============================================================
if page == "🏠  Home":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎯 Sales Forecasting System</div>
            <div class="hero-sub">
                Predicting daily store revenue across
                <span class="hero-accent">1,115 Rossmann stores</span>
                using gradient boosting — trained on 844K real transactions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("🥇 XGBoost", "Best Model",  "26% better than RF",       "#667eea"),
        ("€ 1,866",    "RMSE",        "prediction error per day",  "#667eea"),
        ("0.622",      "R² Score",    "variance explained",        "#2ecc71"),
        ("1,115",      "Stores",      "Rossmann · 2013–2015",      "#f39c12"),
    ]
    for col, (val, label, sub, color) in zip([c1, c2, c3, c4], kpis):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">What this system delivers</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc in zip(
        [f1, f2, f3],
        ["📈", "🧠", "🔍"],
        ["Real-time Prediction", "XGBoost Engine", "Explainable AI"],
        [
            "Configure any store and get an instant daily revenue forecast — factoring in promotions, competition, and seasonality.",
            "Gradient boosting trained on 648K rows with 25 engineered features and time-based validation to prevent leakage.",
            "Feature importance reveals exactly what drives each prediction — Promo, DayOfWeek, and CompetitionDistance lead.",
        ],
    ):
        col.markdown(
            f"""
            <div class="kpi-card" style="text-align:left; padding:20px 22px;">
                <div style="font-size:1.6rem; margin-bottom:10px;">{icon}</div>
                <div style="font-weight:600; font-size:0.95rem; color:#e0e0f0; margin-bottom:6px;">{title}</div>
                <div style="font-size:0.82rem; color:#888aaa; line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model comparison at a glance</div>', unsafe_allow_html=True)

    fig_home = go.Figure(go.Bar(
        x=["Linear Regression", "Random Forest", "XGBoost"],
        y=[2678, 2425, 1866],
        marker_color=[RED, ORANGE, GREEN],
        marker_line_width=0,
        text=["€2,678", "€2,425", "€1,866"],
        textposition="outside",
        textfont=dict(color="#c5c8f0", size=13),
        width=0.45,
    ))
    fig_home.update_layout(
        **CHART_THEME,
        title=dict(text="RMSE by Model — lower is better", font=dict(size=14, color="#c5c8f0")),
        yaxis=dict(title="RMSE (€)", gridcolor="#1e1e35", showgrid=True, zeroline=False),
        xaxis=dict(showgrid=False),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_home, use_container_width=True)


# ============================================================
# ② PREDICT PAGE
# ============================================================
elif page == "🔮  Predict":

    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <div style="font-size:1.6rem; font-weight:700; color:#ffffff;">🔮 Sales Prediction</div>
            <div style="font-size:0.9rem; color:#888aaa; margin-top:4px;">
                Configure store parameters and get an AI-powered daily revenue forecast.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_OK:
        st.warning("⚠️ Model not loaded. Run `python main.py --stage all` first.")
        st.stop()

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div class="section-header">Store Configuration</div>', unsafe_allow_html=True)

        store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)

        col_m, col_d = st.columns(2)
        with col_m:
            month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: pd.to_datetime(f"2024-{x:02d}-01").strftime("%B"),
                index=11,
            )
        with col_d:
            day = st.slider("Day", 1, 28, 15)

        day_of_week = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x],
            index=5,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        promo          = st.toggle("🎉 Promotion Active", value=True)
        school_holiday = st.toggle("🏫 School Holiday",  value=False)
        state_holiday  = st.selectbox(
            "State Holiday", ["None", "Public Holiday", "Easter", "Christmas"], index=0
        )

        st.markdown("<br>", unsafe_allow_html=True)
        competition_distance = st.slider(
            "📍 Competitor Distance (m)", 0, 20000, 500, step=100
        )

        col_st, col_as = st.columns(2)
        with col_st:
            store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])
        with col_as:
            assortment = st.selectbox("Assortment", ["a", "b", "c"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Generate Forecast", use_container_width=True)

    with right:
        if predict_btn:
            is_weekend = 1 if day_of_week in [5, 6] else 0

            sh_map = {
                "None":           (1, 0, 0, 0),
                "Public Holiday": (0, 1, 0, 0),
                "Easter":         (0, 0, 1, 0),
                "Christmas":      (0, 0, 0, 1),
            }
            sh0, sha, shb, shc = sh_map[state_holiday]

            input_df = pd.DataFrame([{
                "Store":                    store_id,
                "DayOfWeek":               day_of_week,
                "Promo":                   int(promo),
                "SchoolHoliday":           int(school_holiday),
                "CompetitionDistance":     float(competition_distance),
                "CompetitionOpenSinceMonth": 6.0,
                "CompetitionOpenSinceYear":  2010.0,
                "Promo2":                  0,
                "Promo2SinceWeek":         0.0,
                "Promo2SinceYear":         0.0,
                "Year":                    2024,
                "Month":                   month,
                "Day":                     day,
                "WeekOfYear":              int(
                    pd.to_datetime(f"2024-{month:02d}-{day:02d}").strftime("%V")
                ),
                "IsWeekend":               is_weekend,
                "IsPromo":                 int(promo),
                "StoreType_b":             1 if store_type == "b" else 0,
                "StoreType_c":             1 if store_type == "c" else 0,
                "StoreType_d":             1 if store_type == "d" else 0,
                "Assortment_b":            1 if assortment == "b" else 0,
                "Assortment_c":            1 if assortment == "c" else 0,
                "StateHoliday_0":          sh0,
                "StateHoliday_a":          sha,
                "StateHoliday_b":          shb,
                "StateHoliday_c":          shc,
            }])

            prediction = float(model.predict(input_df)[0])

            st.markdown(
                f"""
                <div class="pred-box">
                    <div class="pred-label">Predicted Daily Sales</div>
                    <div class="pred-amount">€{prediction:,.2f}</div>
                    <div class="pred-ci">Confidence interval ±€1,866 (RMSE)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Est. Customers",  f"{int(prediction / 9.5):,}", "~€9.5 / visit")
            m2.metric("Weekly Revenue",  f"€{prediction * 7:,.0f}",    "× 7 days")
            m3.metric("Monthly Revenue", f"€{prediction * 30:,.0f}",   "× 30 days")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={
                    "reference": 6500,
                    "valueformat": ",.0f",
                    "prefix": "€",
                    "increasing": {"color": GREEN},
                    "decreasing": {"color": RED},
                },
                number={"prefix": "€", "valueformat": ",.0f",
                        "font": {"size": 26, "color": "#ffffff"}},
                gauge={
                    "axis": {
                        "range": [0, 15000],
                        "tickprefix": "€",
                        "tickformat": ",",
                        "tickcolor": "#555780",
                        "tickfont": {"color": "#888aaa", "size": 10},
                    },
                    "bar": {"color": ACCENT},
                    "bgcolor": "#13132a",
                    "bordercolor": "#2a2a50",
                    "steps": [
                        {"range": [0, 5000],    "color": "#1a1a30"},
                        {"range": [5000, 10000], "color": "#1e1e38"},
                        {"range": [10000, 15000],"color": "#222245"},
                    ],
                    "threshold": {
                        "line": {"color": GREEN, "width": 2},
                        "thickness": 0.75,
                        "value": 6500,
                    },
                },
                title={
                    "text": "vs store avg (€6,500)",
                    "font": {"color": "#888aaa", "size": 12},
                },
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(13,13,30,0)",
                plot_bgcolor="rgba(13,13,30,0)",
                font=dict(family="DM Sans, sans-serif", color="#c5c8f0"),
                height=240,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Insights
            insights = []
            if promo:
                insights.append(
                    f"🎉 **Promotion boost** — adds ~€{prediction * 0.26:,.0f} vs no-promo equivalent"
                )
            if is_weekend:
                insights.append(
                    "📅 **Weekend uplift** — Sat/Sun average 15–30% above weekdays"
                )
            if competition_distance < 300:
                insights.append(
                    "⚠️ **Close competitor** — within 300 m may suppress sales 5–10%"
                )
            if month == 12:
                insights.append(
                    "🎄 **December peak** — holiday season historically highest sales month"
                )

            if insights:
                st.markdown(
                    '<div class="section-header">Business Insights</div>',
                    unsafe_allow_html=True,
                )
                for insight in insights:
                    st.markdown(
                        f'<div class="insight-chip">{insight}</div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("🔍 View raw input features"):
                st.dataframe(
                    input_df.T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )

        else:
            st.markdown(
                """
                <div style="background:#13132a; border:2px dashed #2a2a50;
                            border-radius:20px; padding:70px 30px;
                            text-align:center; margin-top:10px;">
                    <div style="font-size:3rem; margin-bottom:12px;">🔮</div>
                    <div style="color:#555780; font-size:1rem;">
                        Set your parameters on the left<br>
                        and click <strong style="color:#667eea;">Generate Forecast</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ③ ANALYTICS — ALL CHARTS FROM REAL DATA
# ============================================================
elif page == "📊  Analytics":

    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <div style="font-size:1.6rem; font-weight:700; color:#ffffff;">📊 Analytics</div>
            <div style="font-size:0.9rem; color:#888aaa; margin-top:4px;">
                Model performance, feature importance, and business patterns — all from real data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_OK:
        st.warning("⚠️ Model not loaded.")
        st.stop()

    with st.spinner("Loading data…"):
        try:
            X_test, y_test = load_processed_data(config)
            y_pred  = compute_predictions(model, X_test)
            raw_df  = load_raw_data(config)
            DATA_OK = True
        except Exception as e:
            DATA_OK = False
            st.error(
                f"Could not load processed data. "
                f"Run `python main.py --stage all` first.\n\n`{e}`"
            )

    if not DATA_OK:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Model Performance", "🎯 Feature Importance", "🏪 Business Patterns", "📉 Residuals"]
    )

    # ── Tab 1: Model Performance ─────────────────────────────
    with tab1:
        st.markdown(
            '<div class="section-header">Predicted vs Actual Sales (XGBoost — real holdout data)</div>',
            unsafe_allow_html=True,
        )

        n   = min(4000, len(y_test))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_test), n, replace=False)
        yt  = np.array(y_test)[idx]
        yp  = y_pred[idx]

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=yt, y=yp,
            mode="markers",
            marker=dict(color=ACCENT, size=4, opacity=0.5, line=dict(width=0)),
            name="Predictions",
        ))
        max_val = max(yt.max(), yp.max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color=RED, dash="dash", width=2),
            name="Perfect Prediction",
        ))

        rmse_live = float(np.sqrt(np.mean((np.array(y_test) - y_pred) ** 2)))
        mae_live  = float(np.mean(np.abs(np.array(y_test) - y_pred)))
        r2_live   = float(
            1 - np.sum((np.array(y_test) - y_pred) ** 2)
              / np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
        )

        fig_scatter.add_annotation(
            x=0.03, y=0.97, xref="paper", yref="paper",
            text=f"RMSE: €{rmse_live:,.0f}   MAE: €{mae_live:,.0f}   R²: {r2_live:.3f}",
            showarrow=False,
            bgcolor="#1a1a35",
            bordercolor="#2a2a50",
            borderwidth=1,
            font=dict(size=11, color="#c5c8f0", family="DM Mono"),
            align="left",
        )
        fig_scatter.update_layout(
            **CHART_THEME,
            title="XGBoost: Predicted vs Actual (4,000 sample from holdout set)",
            xaxis=dict(title="Actual Sales (€)", gridcolor="#1e1e35", zeroline=False),
            yaxis=dict(title="Predicted Sales (€)", gridcolor="#1e1e35", zeroline=False),
            height=480,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(
            '<div class="section-header">Three-Model Comparison</div>',
            unsafe_allow_html=True,
        )
        a1, a2 = st.columns(2)

        models_c = ["Linear Regression", "Random Forest", "XGBoost"]
        rmse_c   = [2678, 2425, 1866]
        mae_c    = [1959, 1765, 1353]
        colors_c = [RED, ORANGE, GREEN]

        with a1:
            fig_rmse = go.Figure(go.Bar(
                x=models_c, y=rmse_c,
                marker_color=colors_c, marker_line_width=0,
                text=[f"€{v:,}" for v in rmse_c],
                textposition="outside",
                textfont=dict(color="#c5c8f0"),
                width=0.5,
            ))
            fig_rmse.update_layout(
                **CHART_THEME,
                title="RMSE — lower is better",
                yaxis=dict(title="RMSE (€)", gridcolor="#1e1e35", zeroline=False),
                xaxis=dict(showgrid=False),
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with a2:
            fig_mae = go.Figure(go.Bar(
                x=models_c, y=mae_c,
                marker_color=colors_c, marker_line_width=0,
                text=[f"€{v:,}" for v in mae_c],
                textposition="outside",
                textfont=dict(color="#c5c8f0"),
                width=0.5,
            ))
            fig_mae.update_layout(
                **CHART_THEME,
                title="MAE — lower is better",
                yaxis=dict(title="MAE (€)", gridcolor="#1e1e35", zeroline=False),
                xaxis=dict(showgrid=False),
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        metrics_df = pd.DataFrame({
            "Model":          models_c,
            "RMSE (€)":       rmse_c,
            "MAE (€)":        mae_c,
            "Training Time":  ["0.11 s", "11.49 s", "2.54 s"],
            "vs Baseline":    ["—", "−9.5 %", "−30.3 % ✅"],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Feature Importance ────────────────────────────
    with tab2:
        st.markdown(
            '<div class="section-header">XGBoost Feature Importance (from your trained model)</div>',
            unsafe_allow_html=True,
        )

        importance_df = (
            pd.DataFrame({
                "Feature":    X_test.columns,
                "Importance": model.feature_importances_,
            })
            .sort_values("Importance", ascending=True)
            .tail(15)
        )

        fig_imp = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            marker=dict(
                color=importance_df["Importance"],
                colorscale=[[0, "#2a1580"], [0.5, ACCENT], [1, "#a78bfa"]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"{v:.4f}" for v in importance_df["Importance"]],
            textposition="outside",
            textfont=dict(color="#888aaa", size=11, family="DM Mono"),
        ))
        fig_imp.update_layout(
            **CHART_THEME,
            title="Top 15 Features Driving Sales Predictions",
            xaxis=dict(title="Importance Score", gridcolor="#1e1e35", zeroline=False),
            yaxis=dict(showgrid=False),
            height=520,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info(
            "🧠 **Key insight:** `Promo`, `DayOfWeek`, and `CompetitionDistance` are the three "
            "strongest predictors — confirming that promotions, day timing, and competition "
            "proximity explain the most sales variance in the Rossmann dataset."
        )

    # ── Tab 3: Business Patterns ─────────────────────────────
    with tab3:

        b1, b2 = st.columns(2)

        with b1:
            st.markdown(
                '<div class="section-header">Average Sales by Day of Week</div>',
                unsafe_allow_html=True,
            )
            day_avg = (
                raw_df[raw_df["Sales"] > 0]
                .groupby("DayOfWeek")["Sales"]
                .mean()
                .reset_index()
            )
            day_avg["Day"]   = day_avg["DayOfWeek"].map(
                {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            )
            day_avg["Color"] = day_avg["DayOfWeek"].apply(
                lambda x: GREEN if x in [5, 6] else ACCENT
            )

            fig_day = go.Figure(go.Bar(
                x=day_avg["Day"],
                y=day_avg["Sales"],
                marker_color=day_avg["Color"],
                marker_line_width=0,
                text=[f"€{v:,.0f}" for v in day_avg["Sales"]],
                textposition="outside",
                textfont=dict(color="#c5c8f0", size=11),
                width=0.6,
            ))
            fig_day.update_layout(
                **CHART_THEME,
                yaxis=dict(title="Avg Sales (€)", gridcolor="#1e1e35", zeroline=False),
                xaxis=dict(showgrid=False),
                height=360,
                showlegend=False,
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with b2:
            st.markdown(
                '<div class="section-header">Promotion Impact on Sales</div>',
                unsafe_allow_html=True,
            )
            promo_day = (
                raw_df[raw_df["Sales"] > 0]
                .groupby(["DayOfWeek", "Promo"])["Sales"]
                .mean()
                .reset_index()
            )
            promo_day["Day"] = promo_day["DayOfWeek"].map(
                {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            )
            no_promo  = promo_day[promo_day["Promo"] == 0]
            yes_promo = promo_day[promo_day["Promo"] == 1]

            avg_boost = (
                yes_promo["Sales"].mean() / no_promo["Sales"].mean() - 1
            ) * 100

            fig_promo = go.Figure()
            fig_promo.add_trace(go.Bar(
                name="No Promotion",
                x=no_promo["Day"], y=no_promo["Sales"],
                marker_color="#2a2a50", marker_line_width=0,
                width=0.35, offset=-0.18,
            ))
            fig_promo.add_trace(go.Bar(
                name="Promotion Active",
                x=yes_promo["Day"], y=yes_promo["Sales"],
                marker_color=GREEN, marker_line_width=0,
                width=0.35, offset=0.18,
            ))
            fig_promo.update_layout(
                **CHART_THEME,
                yaxis=dict(title="Avg Sales (€)", gridcolor="#1e1e35", zeroline=False),
                xaxis=dict(showgrid=False),
                barmode="overlay",
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    xref="paper", yref="paper", x=0.98, y=0.96,
                    text=f"Avg boost: +{avg_boost:.1f}%",
                    showarrow=False,
                    bgcolor="#13132a", bordercolor="#2a2a50",
                    font=dict(color=GREEN, size=11),
                )],
            )
            st.plotly_chart(fig_promo, use_container_width=True)

        # Monthly trend
        st.markdown(
            '<div class="section-header">Monthly Sales Trend (full dataset)</div>',
            unsafe_allow_html=True,
        )
        raw_df["YearMonth"] = raw_df["Date"].dt.to_period("M").astype(str)
        monthly = (
            raw_df[raw_df["Sales"] > 0]
            .groupby("YearMonth")["Sales"]
            .mean()
            .reset_index()
        )
        monthly["YearMonth_dt"] = pd.to_datetime(monthly["YearMonth"])

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly["YearMonth_dt"],
            y=monthly["Sales"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5, color=ACCENT),
            fill="tozeroy",
            fillcolor="rgba(102,126,234,0.07)",
            name="Avg Daily Sales",
        ))
        fig_trend.update_layout(
            **CHART_THEME,
            title="Average Daily Sales by Month",
            xaxis=dict(title="Month", showgrid=False),
            yaxis=dict(title="Avg Sales (€)", gridcolor="#1e1e35", zeroline=False),
            height=360,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Store type breakdown
        st.markdown(
            '<div class="section-header">Average Sales by Store Type</div>',
            unsafe_allow_html=True,
        )
        type_avg = (
            raw_df[raw_df["Sales"] > 0]
            .groupby("StoreType")["Sales"]
            .mean()
            .reset_index()
            .rename(columns={"Sales": "AvgSales"})
        )
        fig_type = go.Figure(go.Bar(
            x=type_avg["StoreType"],
            y=type_avg["AvgSales"],
            marker_color=[ACCENT, GREEN, ORANGE, "#a855f7"],
            marker_line_width=0,
            text=[f"€{v:,.0f}" for v in type_avg["AvgSales"]],
            textposition="outside",
            textfont=dict(color="#c5c8f0"),
            width=0.5,
        ))
        fig_type.update_layout(
            **CHART_THEME,
            title="Avg Daily Sales by Store Type (a / b / c / d)",
            xaxis=dict(title="Store Type", showgrid=False),
            yaxis=dict(title="Avg Sales (€)", gridcolor="#1e1e35", zeroline=False),
            height=340,
            showlegend=False,
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # ── Tab 4: Residuals ─────────────────────────────────────
    with tab4:
        st.markdown(
            '<div class="section-header">Residuals Analysis — model error distribution</div>',
            unsafe_allow_html=True,
        )

        residuals = np.array(y_test) - y_pred

        r1, r2 = st.columns(2)

        with r1:
            fig_hist = go.Figure(go.Histogram(
                x=residuals,
                nbinsx=60,
                marker_color=ACCENT,
                marker_line_color="#0b0b16",
                marker_line_width=0.5,
                opacity=0.85,
            ))
            fig_hist.add_vline(
                x=0, line_color=RED, line_dash="dash", line_width=2,
                annotation_text="Zero error",
                annotation_font_color=RED,
            )
            fig_hist.update_layout(
                **CHART_THEME,
                title="Distribution of Prediction Errors",
                xaxis=dict(title="Residual (Actual − Predicted) €", gridcolor="#1e1e35"),
                yaxis=dict(title="Count", gridcolor="#1e1e35"),
                height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with r2:
            samp_n = min(5000, len(y_pred))
            rng2   = np.random.default_rng(0)
            samp_i = rng2.choice(len(y_pred), samp_n, replace=False)

            fig_res = go.Figure(go.Scatter(
                x=y_pred[samp_i],
                y=residuals[samp_i],
                mode="markers",
                marker=dict(color=ACCENT, size=4, opacity=0.45, line=dict(width=0)),
            ))
            fig_res.add_hline(
                y=0, line_color=RED, line_dash="dash", line_width=2
            )
            fig_res.update_layout(
                **CHART_THEME,
                title="Residuals vs Predicted Values",
                xaxis=dict(title="Predicted Sales (€)", gridcolor="#1e1e35", zeroline=False),
                yaxis=dict(title="Residual (€)", gridcolor="#1e1e35", zeroline=False),
                height=380,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown(
            '<div class="section-header">Error Statistics</div>',
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean Error",        f"€{np.mean(residuals):,.1f}",  "should be ≈ 0")
        s2.metric("Std of Errors",     f"€{np.std(residuals):,.0f}",   "spread")
        s3.metric("Max Overestimate",  f"€{np.min(residuals):,.0f}",   "worst under")
        s4.metric("Max Underestimate", f"€{np.max(residuals):,.0f}",   "worst over")


# ============================================================
# ④ MODEL INFO PAGE
# ============================================================
elif page == "ℹ️   Model Info":

    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <div style="font-size:1.6rem; font-weight:700; color:#ffffff;">ℹ️ Model Details</div>
            <div style="font-size:0.9rem; color:#888aaa; margin-top:4px;">
                Architecture, validation strategy, pipeline structure, and file locations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">XGBoost Architecture</div>',
        unsafe_allow_html=True,
    )

    col_arch, col_stats = st.columns([1.6, 1])

    with col_arch:
        st.code(
            """XGBRegressor(
  objective        = 'reg:squarederror',
  n_estimators     = 100,   # boosting rounds
  max_depth        = 6,     # tree depth
  learning_rate    = 0.1,   # shrinkage
  subsample        = 0.8,   # row sampling
  colsample_bytree = 0.8,   # feature sampling
  reg_alpha        = 0.1,   # L1 regularisation
  reg_lambda       = 1.0,   # L2 regularisation
  random_state     = 42,
  n_jobs           = -1,    # all CPU cores
)""",
            language="python",
        )

    with col_stats:
        for label, val, color, sub in [
            ("Training Samples", "100,000",  "#667eea", "from 648K total (15%)"),
            ("Features",         "25",       "#2ecc71", "engineered from 18 raw"),
            ("Model File Size",  "0.46 MB",  "#f39c12", "xgboost_sales_model.pkl"),
        ]:
            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom:12px;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color}; font-size:1.5rem;">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-header">Validation Strategy</div>',
        unsafe_allow_html=True,
    )
    val_df = pd.DataFrame({
        "Aspect":    ["Train / Test Split", "Sampling", "Primary Metric",
                      "Secondary Metrics",  "Leakage Prevention"],
        "Method":    ["Time-based (pre / post 2015-01-01)", "Random n=100K (seed=42)",
                      "RMSE", "MAE, R², MAPE",
                      "Customers, Open, PromoInterval removed"],
        "Rationale": [
            "Mirrors real forecasting — no future data in training",
            "Fast iteration without kernel crash",
            "Penalises large errors heavily",
            "Comprehensive view of error distribution",
            "Only features available at prediction time are retained",
        ],
    })
    st.dataframe(val_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="section-header">Performance Summary</div>',
        unsafe_allow_html=True,
    )
    p1, p2, p3, p4 = st.columns(4)
    for col, label, val, color, sub in zip(
        [p1, p2, p3, p4],
        ["RMSE",    "MAE",     "R²",     "MAPE"],
        ["€ 1,866", "€ 1,353", "0.622",  "~18 %"],
        [ACCENT,    ACCENT,    GREEN,    ORANGE],
        ["avg error magnitude", "avg absolute error", "variance explained", "% error"],
    ):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}; font-size:1.6rem;">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-header">Pipeline Structure</div>',
        unsafe_allow_html=True,
    )
    st.code(
        """python main.py --stage all
  │
  ├── Stage 1  data_preprocessing.py
  │     Load train.csv + store.csv → merge → remove closed stores
  │     Handle missing values → cleaned_data.csv  [844,392 rows]
  │
  ├── Stage 2  feature_engineering.py
  │     Extract date features (Year, Month, Day, WeekOfYear, DayOfWeek, IsWeekend)
  │     One-hot encode StoreType, Assortment, StateHoliday
  │     Remove leakage columns (Customers, Open, PromoInterval)
  │     Time-based split → X_train [648,360 × 25]  X_test [196,032 × 25]
  │
  ├── Stage 3  train.py
  │     Linear Regression  →  RMSE €2,678  (0.11 s)
  │     Random Forest      →  RMSE €2,425  (11.49 s)
  │     XGBoost            →  RMSE €1,866  (2.54 s)  ✅ winner
  │     Save all 3 models as .pkl files
  │
  └── Stage 4  evaluate.py
        Load XGBoost → predict on full X_test
        Compute RMSE / MAE / R² → generate diagnostic plots""",
        language="bash",
    )

    st.markdown(
        '<div class="section-header">File Locations</div>',
        unsafe_allow_html=True,
    )

    if MODEL_OK:
        file_rows = [
            ("🤖 Primary Model (XGBoost)",  config.XGB_MODEL),
            ("🌲 Backup Model (RF)",         config.RF_MODEL),
            ("📏 Baseline (Linear Reg.)",    config.LR_MODEL),
            ("📂 Processed Data folder",     config.PROCESSED_DATA),
            ("📊 Visualisations folder",     config.VIZ_DIR),
            ("🗂️ Raw Data folder",            config.RAW_DATA),
        ]
        for label, path in file_rows:
            exists = Path(str(path)).exists()
            tick   = "✅" if exists else "❌"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            background:#13132a; border:1px solid #1e1e35; border-radius:10px;
                            padding:10px 16px; margin-bottom:8px;">
                    <span style="color:#c5c8f0; font-size:0.88rem; min-width:220px;">{label}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.76rem; color:#555780;">
                        {tick}&nbsp;&nbsp;{path}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Dataset: Rossmann Store Sales (Kaggle) · "
        "Framework: XGBoost + Streamlit · "
        "Built with 🏪"
    )
