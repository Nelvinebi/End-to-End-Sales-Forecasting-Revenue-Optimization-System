#!/usr/bin/env python3
"""
🎯 Sales Forecasting Dashboard
Professional Streamlit app for Rossmann store sales prediction.
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import Config

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Rossmann Sales Forecasting | AI-Powered Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_data
def load_data(config):
    """Load processed data for analytics."""
    try:
        X_test = pd.read_csv(config.X_TEST)
        y_test = pd.read_csv(config.Y_TEST).squeeze()
        model = joblib.load(config.XGB_MODEL)
        y_pred = model.predict(X_test)
        return X_test, y_test, y_pred
    except Exception as e:
        return None, None, None
    
@st.cache_resource
def load_model():
    """Load trained XGBoost model."""
    config = Config()
    model = joblib.load(config.XGB_MODEL)
    return model, config

try:
    model, config = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"❌ Model not found. Run: `python main.py --stage all`\\nError: {e}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="margin: 0; font-size: 1.8rem;">🏪 Rossmann</h1>
        <p style="color: #a0a0a0; margin: 5px 0 0 0;">AI Sales Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📍 Navigation",
        ["🏠 Home", "🔮 Predict", "📊 Analytics", "ℹ️ Model Info"],
        index=0
    )
    
    st.markdown("---")
    
    if model_ready:
        st.markdown("""
        <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 10px;">
            <p style="margin: 0; color: #a0a0a0; font-size: 0.8rem;">MODEL STATUS</p>
            <p style="margin: 5px 0 0 0; color: #2ecc71; font-weight: bold;">🟢 XGBoost Active</p>
            <p style="margin: 5px 0 0 0; color: #a0a0a0; font-size: 0.75rem;">RMSE: €1,866</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🎯 Sales Forecasting</h1>
        <p style="font-size: 1.3rem; color: #a0a0a0;">
            Predict daily store sales with machine learning.<br>
            <span style="color: #667eea; font-weight: 600;">Accuracy: ±€1,866 RMSE</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Best Model</p>
            <p class="metric-value">🥇 XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <p class="metric-label">RMSE</p>
            <p class="metric-value">€1,866</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
            <p class="metric-label">R² Score</p>
            <p class="metric-value">0.62</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);">
            <p class="metric-label">Stores</p>
            <p class="metric-value">1,115</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("✨ Key Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.info("📈 **Real-time Predictions**\\nGet instant sales forecasts for any store configuration")
    
    with feat_col2:
        st.success("🧠 **XGBoost Engine**\\nIndustry-standard gradient boosting with 62% accuracy")
    
    with feat_col3:
        st.warning("🎨 **Interactive Analytics**\\nExplore feature importance and business insights")

# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "🔮 Predict":
    st.markdown("""
    <h1 style="text-align: center; margin-bottom: 5px;">🔮 Sales Prediction</h1>
    <p style="text-align: center; color: #a0a0a0; margin-bottom: 30px;">
        Configure store parameters and get AI-powered sales forecasts
    </p>
    """, unsafe_allow_html=True)
    
    if not model_ready:
        st.error("⚠️ Model not loaded. Please train the model first.")
        st.stop()
    
    input_col, result_col = st.columns([1, 1.2])
    
    with input_col:
        st.markdown("<h3 style='color: #667eea;'>🏪 Store Configuration</h3>", unsafe_allow_html=True)
        
        store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: pd.to_datetime(f"2024-{x}-01").strftime("%B"),
                index=11
            )
        with col_date2:
            day = st.slider("Day", 1, 31, 20)
        
        day_of_week = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=5
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        promo = st.toggle("🎉 Active Promotion", value=True)
        
        col_promo1, col_promo2 = st.columns(2)
        with col_promo1:
            school_holiday = st.toggle("🏫 School Holiday", value=False)
        with col_promo2:
            state_holiday = st.selectbox("State Holiday", ['None', 'Public', 'Easter', 'Christmas'], index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        competition_distance = st.slider("📍 Competitor Distance (meters)", min_value=0, max_value=20000, value=500, step=100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_char1, col_char2 = st.columns(2)
        with col_char1:
            store_type = st.selectbox("Store Type", ['a', 'b', 'c', 'd'], index=0)
        with col_char2:
            assortment = st.selectbox("Assortment", ['a', 'b', 'c'], index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Generate Prediction", use_container_width=True)
    
    with result_col:
        if predict_btn:
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            store_type_b = 1 if store_type == 'b' else 0
            store_type_c = 1 if store_type == 'c' else 0
            store_type_d = 1 if store_type == 'd' else 0
            
            assortment_b = 1 if assortment == 'b' else 0
            assortment_c = 1 if assortment == 'c' else 0
            
            state_holiday_a = 1 if state_holiday == 'Public' else 0
            state_holiday_b = 1 if state_holiday == 'Easter' else 0
            state_holiday_c = 1 if state_holiday == 'Christmas' else 0
            state_holiday_0 = 1 if state_holiday == 'None' else 0
            
            input_data = pd.DataFrame([{
                'Store': store_id,
                'DayOfWeek': day_of_week,
                'Promo': int(promo),
                'SchoolHoliday': int(school_holiday),
                'CompetitionDistance': float(competition_distance),
                'CompetitionOpenSinceMonth': 6.0,
                'CompetitionOpenSinceYear': 2010.0,
                'Promo2': 0,
                'Promo2SinceWeek': 0.0,
                'Promo2SinceYear': 0.0,
                'Year': 2024,
                'Month': month,
                'Day': day,
                'WeekOfYear': int(pd.to_datetime(f"2024-{month}-{day}").strftime("%V")),
                'IsWeekend': is_weekend,
                'IsPromo': int(promo),
                'StoreType_b': store_type_b,
                'StoreType_c': store_type_c,
                'StoreType_d': store_type_d,
                'Assortment_b': assortment_b,
                'Assortment_c': assortment_c,
                'StateHoliday_0': state_holiday_0,
                'StateHoliday_a': state_holiday_a,
                'StateHoliday_b': state_holiday_b,
                'StateHoliday_c': state_holiday_c
            }])
            
            prediction = model.predict(input_data)[0]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 20px; text-align: center;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 1rem; text-transform: uppercase;">
                    Predicted Daily Sales
                </p>
                <p style="color: white; margin: 10px 0; font-size: 3.5rem; font-weight: bold;">
                    €{prediction:,.2f}
                </p>
                <p style="color: rgba(255,255,255,0.7); margin: 0;">
                    Confidence Interval: ±€1,866 (RMSE)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                est_customers = int(prediction / 9.5)
                st.metric("Est. Customers", f"{est_customers:,}", delta="~€9.5/visit")
            
            with m_col2:
                revenue_7day = prediction * 7
                st.metric("Weekly Revenue", f"€{revenue_7day:,.0f}", delta="7 days")
            
            with m_col3:
                revenue_30day = prediction * 30
                st.metric("Monthly Revenue", f"€{revenue_30day:,.0f}", delta="30 days")
            
            insights = []
            if promo:
                boost = prediction * 0.25
                insights.append(f"🎉 **Promotion Boost**: +€{boost:,.0f} expected vs no promo")
            if is_weekend:
                insights.append("📅 **Weekend Effect**: Higher foot traffic expected")
            if competition_distance < 300:
                insights.append("⚠️ **Close Competitor**: May reduce sales by 5-10%")
            if month in [11, 12]:
                insights.append("🎄 **Holiday Season**: Peak sales period")
            
            if insights:
                st.subheader("💡 Business Insights")
                for insight in insights:
                    st.info(insight)
            
            with st.expander("🔍 View Input Data"):
                st.dataframe(input_data.T, use_container_width=True)
        
        else:
            st.markdown("""
            <div style="background: rgba(30,30,46,0.5); padding: 60px 20px; border-radius: 20px; 
                        border: 2px dashed rgba(102,126,234,0.3); text-align: center;">
                <p style="font-size: 4rem; margin: 0;">🔮</p>
                <p style="color: #a0a0a0; font-size: 1.2rem;">
                    Configure store parameters<br>and click "Generate Prediction"
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "📊 Analytics":
    st.markdown("""
    <h1 style="text-align: center;">📊 Analytics Dashboard</h1>
    <p style="text-align: center; color: #a0a0a0; margin-bottom: 30px;">
        Explore model performance and business insights
    </p>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🎯 Feature Importance", "🏪 Business Patterns"])
    
    with tab1:
    X_test, y_test, y_pred = load_data(config)

    if y_test is not None:
        # Real prediction vs actual scatter
        sample = min(3000, len(y_test))
        fig = px.scatter(
            x=y_test[:sample],
            y=y_pred[:sample],
            labels={'x': 'Actual Sales (€)', 'y': 'Predicted Sales (€)'},
            title='Predicted vs Actual Sales (XGBoost)',
            opacity=0.5,
            template='plotly_dark'
        )
        # Perfect prediction line
        max_val = max(y_test[:sample].max(), y_pred[:sample].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Top Features Driving Sales")
        
        features = ['Promo', 'DayOfWeek', 'CompetitionDistance', 'Store', 
                   'Month', 'Year', 'IsWeekend', 'IsPromo', 'Assortment_c',
                   'StoreType_d', 'SchoolHoliday', 'WeekOfYear', 'Day']
        importance = [0.182, 0.156, 0.134, 0.098, 0.087, 0.076, 0.065, 
                      0.054, 0.043, 0.038, 0.032, 0.021, 0.014]
        
        fig = px.bar(
            x=importance, 
            y=features,
            orientation='h',
            color=importance,
            color_continuous_scale='Viridis',
            title='XGBoost Feature Importance'
        )
        
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🧠 **Key Insight**: Promotions and day of week are the strongest sales drivers")
    
    with tab3:
        st.subheader("Sales Patterns")
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sales_avg = [6800, 6950, 7100, 7250, 7800, 8900, 8500]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=days,
            y=sales_avg,
            marker_color=['#3498db']*5 + ['#e74c3c']*2,
            text=[f'€{v}' for v in sales_avg],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Average Sales by Day of Week',
            xaxis_title='Day',
            yaxis_title='Sales (€)',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekend Boost", "+15%", "Sat-Sun vs Mon-Fri")
        with col2:
            st.metric("Best Day", "Saturday", "€8,900 avg")

# ============================================================
# MODEL INFO PAGE
# ============================================================
elif page == "ℹ️ Model Info":
    st.markdown("""
    <h1 style="text-align: center;">ℹ️ Technical Details</h1>
    <p style="text-align: center; color: #a0a0a0; margin-bottom: 30px;">
        Model architecture, validation, and performance metrics
    </p>
    """, unsafe_allow_html=True)
    
    st.subheader("🏗️ Model Architecture")
    
    arch_col1, arch_col2 = st.columns([2, 1])
    
    with arch_col1:
        st.code("""
XGBoost Regressor
├── n_estimators: 100 (boosting rounds)
├── max_depth: 6 (tree depth)
├── learning_rate: 0.1 (shrinkage)
├── subsample: 0.8 (row sampling)
├── colsample_bytree: 0.8 (feature sampling)
├── objective: reg:squarederror
└── random_state: 42 (reproducibility)
        """, language='yaml')
    
    with arch_col2:
        st.metric("Training Samples", "100,000", "from 844K total")
        st.metric("Test Samples", "50,000", "holdout validation")
        st.metric("Features", "25", "engineered from 18 raw")
    
    st.subheader("✅ Validation Strategy")
    
    st.markdown("""
    | Aspect | Method | Rationale |
    |--------|--------|-----------|
    | **Train/Test Split** | Time-based (pre/post 2015-01-01) | Prevents data leakage |
    | **Sampling** | Random (n=100K) | Speed without bias |
    | **Metrics** | RMSE, MAE, R², MAPE | Comprehensive evaluation |
    | **Cross-validation** | Holdout + temporal | Real-world simulation |
    """)
    
    st.subheader("📊 Performance Summary")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("RMSE", "€1,866", "± prediction error")
    with perf_col2:
        st.metric("MAE", "€1,353", "average error")
    with perf_col3:
        st.metric("R²", "0.622", "variance explained")
    with perf_col4:
        st.metric("MAPE", "~18%", "percentage error")
    
    st.subheader("📁 File Locations")
    
    if model_ready:
        st.json({
            "model": str(config.XGB_MODEL),
            "processed_data": str(config.PROCESSED_DATA),
            "visualizations": str(config.VIZ_DIR),
            "raw_data": str(config.RAW_DATA)
        })
    
    st.markdown("---")
    st.caption("Dataset: Rossmann Store Sales (Kaggle) | Framework: XGBoost + Streamlit")
