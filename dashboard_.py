import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------------------------------------------
# DARK THEME STYLING (GLOBAL)
# ------------------------------------------------------------
st.markdown("""
<style>

:root {
    --primary-bg: #0d1117;
    --secondary-bg: #161b22;
    --card-bg: rgba(22, 27, 34, 0.55);
    --border-color: #30363d;
    --text-color: #e6edf3;
    --accent: #58a6ff;
}

/* Main background */
body, .main, .block-container {
    background-color: var(--primary-bg) !important;
    color: var(--text-color) !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--secondary-bg) !important;
    color: var(--text-color);
}
section[data-testid="stSidebar"] * {
    color: var(--text-color) !important;
}

/* Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 15px;
    backdrop-filter: blur(6px);
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    backdrop-filter: blur(4px);
}

/* Inputs */
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
    background-color: var(--secondary-bg) !important;
    color: var(--text-color) !important;
}

/* Buttons */
.stButton>button {
    background: #238636 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    height: 42px;
}
.stButton>button:hover {
    background: #2ea043 !important;
}

/* Tables */
tbody, thead, tr, th, td {
    color: var(--text-color) !important;
    background-color: var(--secondary-bg) !important;
}

/* Plotly charts */
.js-plotly-plot .plotly .main-svg {
    background-color: var(--primary-bg) !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="RBI Macro Dashboard",
    layout="wide",
    page_icon="🏦"
)

st.markdown("<h1 style='text-align:center;'>🏦 RBI Macro Dashboard — Premium Dark Edition</h1>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Dummy Data Loader (Replace with API or CSV Data)
# ------------------------------------------------------------
def generate_dummy_series(name, start="2015-01-01", end=datetime.today()):
    dates = pd.date_range(start, end)
    values = np.cumsum(np.random.randn(len(dates))) + 100
    return pd.DataFrame({"date": dates, "value": values})


# Example datasets
liquidity_df = generate_dummy_series("Liquidity")
inflation_df = generate_dummy_series("Inflation")
gdp_df = generate_dummy_series("GDP")


# ------------------------------------------------------------
# Safe Latest Value
# ------------------------------------------------------------
def safe_latest(df, fmt="0.2f"):
    try:
        return f"{df['value'].iloc[-1]:{fmt}}"
    except:
        return "N/A"


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Filters")
    chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Area Chart", "Bar Chart"])
    show_table = st.checkbox("Show Data Table", False)


# ------------------------------------------------------------
# Chart Function
# ------------------------------------------------------------
def plot_chart(df, title):
    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines'))
    elif chart_type == "Area Chart":
        fig.add_trace(go.Scatter(x=df['date'], y=df['value'], fill='tozeroy', mode='lines'))
    elif chart_type == "Bar Chart":
        fig.add_trace(go.Bar(x=df['date'], y=df['value']))

    fig.update_layout(
        title=title,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ------------------------------------------------------------
# KPI Section
# ------------------------------------------------------------
st.subheader("📊 Key Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("System Liquidity", safe_latest(liquidity_df))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Inflation", safe_latest(inflation_df))
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("GDP Trend", safe_latest(gdp_df))
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Charts Section
# ------------------------------------------------------------
st.subheader("📈 Market Indicators")

colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(plot_chart(liquidity_df, "System Liquidity Trend"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(plot_chart(inflation_df, "Inflation Trend"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.plotly_chart(plot_chart(gdp_df, "GDP Growth Trend"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Optional Data Table
# ------------------------------------------------------------
if show_table:
    st.subheader("📄 Data Table")
    st.dataframe(liquidity_df)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown(
    "<p style='text-align:center; margin-top:50px; color:#666;'>Made with ❤️ in Dark Mode</p>",
    unsafe_allow_html=True
)
