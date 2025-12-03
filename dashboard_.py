# app.py (enhanced RBI Macro Dashboard v2.0 — FIXED)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from io import BytesIO
import zipfile

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="RBI Macro Dashboard v2.0", layout="wide",
                   page_icon="🏦", initial_sidebar_state="collapsed")

# ---------- STYLES ----------
PRIMARY = "#0B63A8"
ACCENT = "#0b84a5"
BG = "#f7fbff"

st.markdown(f"""
<style>
    .stApp {{ background: {BG}; }}
    header .decoration {{ display: none; }}
    .big-title {{
        font-size:28px;
        font-weight:700;
        color: {PRIMARY};
        margin-bottom: 0px;
    }}
    .subtitle {{
        color: #475569;
        margin-top: 0px;
        margin-bottom: 12px;
    }}
    .card {{
        background: white;
        border-radius:12px;
        padding: 14px;
        box-shadow: 0 2px 10px rgba(12, 36, 60, 0.06);
    }}
</style>
""", unsafe_allow_html=True)

# Top header
st.markdown("<div class='big-title'>🏦 RBI Macro Economic Dashboard — v2.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Inflation • Liquidity • Monetary Policy • Forecasts • Exports — polished for RBI application</div>", unsafe_allow_html=True)

# ---------- CONFIG ----------
FRED_API_KEY = st.secrets.get("fred_api_key")

# ---------- HELPERS ----------
def get_fred(series_id):
    if not FRED_API_KEY:
        return pd.DataFrame(columns=["date", "value"])
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json().get("observations", [])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna()[["date", "value"]]
    except:
        return pd.DataFrame(columns=["date", "value"])

def india_cpi():
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL?format=json&per_page=500"
        r = requests.get(url, timeout=15)
        payload = r.json()
        data = payload[1]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%Y", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna()[["date", "value"]].sort_values("date")
    except:
        return pd.DataFrame(columns=["date", "value"])

def fetch_usdinr():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=10)
        return r.json()["rates"]["INR"]
    except:
        return None

def linear_forecast(df, periods=12, freq='M'):
    if df.empty or len(df) < 3:
        df["is_forecast"] = False
        return df

    df = df.sort_values("date")
    x = np.array([d.toordinal() for d in df["date"]])
    y = df["value"].astype(float).values
    slope, intercept = np.polyfit(x, y, 1)

    last = df["date"].iloc[-1]
    future_dates = [
        (last + pd.DateOffset(months=i)).to_pydatetime() for i in range(1, periods + 1)
    ]

    x_future = np.array([d.toordinal() for d in future_dates])
    y_future = intercept + slope * x_future

    hist = df.copy()
    hist["is_forecast"] = False

    fut = pd.DataFrame({"date": future_dates, "value": y_future, "is_forecast": True})
    return pd.concat([hist, fut])

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def zip_datasets(dict_of_dfs):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in dict_of_dfs.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ---------- TABS ----------
tabs = st.tabs([
    "Overview",
    "Inflation (India + US)",
    "Liquidity (India + US)",
    "Riskometer",
    "Correlations & Forecasts",
    "Yield Curve & Policy",
    "Export / Report"
])

# -------------------------------------------------------------------
# ---------------------------- OVERVIEW -----------------------------
# -------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dashboard snapshot")

    col1, col2, col3, col4 = st.columns(4)

    us_cpi = get_fred("CPIAUCSL")
    fed_bs = get_fred("WALCL")
    ind_cpi_data = india_cpi()
    usd_inr_rate = fetch_usdinr()

    with col1:
        st.metric("US CPI (latest)", f"{us_cpi['value'].iloc[-1]:.2f}" if not us_cpi.empty else "N/A")

    with col2:
        st.metric("India CPI (annual)", f"{ind_cpi_data['value'].iloc[-1]:.2f}" if not ind_cpi_data.empty else "N/A")

    with col3:
        st.metric("Fed Balance Sheet (WALCL)", f"{fed_bs['value'].iloc[-1]:,.0f}" if not fed_bs.empty else "N/A")

    with col4:
        st.metric("USD → INR (spot)", f"{usd_inr_rate:.2f}" if usd_inr_rate else "N/A")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# ----------------------------- INFLATION ----------------------------
# -------------------------------------------------------------------
with tabs[1]:
    st.header("📌 Inflation: US & India")

    col_us, col_ind = st.columns(2)

    # US CPI
    with col_us:
        st.subheader("🇺🇸 US CPI (CPIAUCSL)")
        if us_cpi.empty:
            st.info("US CPI data unavailable — check API key.")
        else:
            st.plotly_chart(px.line(us_cpi, x="date", y="value",
                                    title="US CPI"), use_container_width=True)

            us_proj = linear_forecast(us_cpi, 12)
            st.plotly_chart(px.line(us_proj, x="date", y="value", color="is_forecast",
                                    title="US CPI + Forecast"), use_container_width=True)

    # India CPI
    with col_ind:
        st.subheader("🇮🇳 India CPI (Annual)")
        if ind_cpi_data.empty:
            st.info("India CPI API error.")
        else:
            st.plotly_chart(px.line(ind_cpi_data, x="date", y="value",
                                    title="India CPI"), use_container_width=True)

            ind_proj = linear_forecast(ind_cpi_data, 5)
            st.plotly_chart(px.line(ind_proj, x="date", y="value", color="is_forecast",
                                    title="India CPI + Forecast"), use_container_width=True)

# -------------------------------------------------------------------
# ----------------------------- LIQUIDITY ----------------------------
# -------------------------------------------------------------------
with tabs[2]:
    st.header("📌 Liquidity — Fed & India")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("🇺🇸 Fed Balance Sheet (WALCL)")
        if fed_bs.empty:
            st.info("FRED key missing.")
        else:
            st.plotly_chart(px.line(fed_bs, x="date", y="value",
                                    title="Fed Balance Sheet"), use_container_width=True)

    with col_b:
        st.subheader("🇮🇳 Upload India Liquidity CSV")
        st.info("CSV must include `date` and `value` columns.")

        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if "date" not in df.columns or "value" not in df.columns:
                    st.error("Missing required columns.")
                else:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna()

                    st.line_chart(df.set_index("date"))
                    proj = linear_forecast(df, 12)
                    st.plotly_chart(px.line(proj, x="date", y="value", color="is_forecast",
                                            title="India Liquidity + Forecast"), use_container_width=True)

                    st.download_button("Download cleaned CSV",
                                       df_to_csv_bytes(df),
                                       "india_liquidity_cleaned.csv")
            except Exception as e:
                st.error(f"Parse error: {e}")

# -------------------------------------------------------------------
# ---------------------------- RISKOMETER ----------------------------
# -------------------------------------------------------------------
with tabs[3]:
    st.header("📌 Portfolio Riskometer (RBI-style)")

    col1, col2, col3 = st.columns(3)
    eq = col1.number_input("Equity (%)", 0.0, 100.0, 40.0)
    debt = col2.number_input("Debt (%)", 0.0, 100.0, 40.0)
    gold = col3.number_input("Gold (%)", 0.0, 100.0, 20.0)

    total = eq + debt + gold
    st.write(f"Total: {total:.1f}%")

    if total > 0:
        risk_score = eq * 0.7 + gold * 0.2 + debt * 0.1

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Risk Score"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig)

# -------------------------------------------------------------------
# ----------------- CORRELATIONS & FORECASTS ------------------------
# -------------------------------------------------------------------
with tabs[4]:
    st.header("📌 Correlations & Simple Forecasts")

    uploaded = st.file_uploader("Upload multiple CSVs", type="csv", accept_multiple_files=True)

    dfs = {}
    if uploaded:
        for f in uploaded:
            try:
                name = f.name.split(".")[0]
                d = pd.read_csv(f)
                d.columns = [c.strip().lower() for c in d.columns]
                if "date" not in d.columns or "value" not in d.columns:
                    continue
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
                d = d.dropna()[["date", "value"]].rename(columns={"value": name})
                dfs[name] = d
            except:
                pass

    if dfs:
        merged = None
        for name, d in dfs.items():
            if merged is None:
                merged = d
            else:
                merged = pd.merge(merged, d, on="date", how="outer")

        merged = merged.sort_values("date").set_index("date").interpolate()
        st.dataframe(merged.tail())

        if merged.shape[1] > 1:
            st.plotly_chart(px.imshow(merged.corr(), text_auto=True,
                                      title="Correlation Heatmap"),
                            use_container_width=True)

# -------------------------------------------------------------------
# ------------------------ YIELD CURVE ------------------------------
# -------------------------------------------------------------------
with tabs[5]:
    st.header("📉 Yield Curve & Policy Rates")

    st.subheader("Manual Yield Curve Input")
    tenors = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    cols = st.columns(len(tenors))

    yields = [cols[i].number_input(f"{t} (%)", 0.0, 20.0, 7.0) for i, t in enumerate(tenors)]

    if st.button("Plot Yield Curve"):
        st.plotly_chart(px.line(x=tenors, y=yields, markers=True,
                                title="India G-Sec Yield Curve"),
                        use_container_width=True)

# -------------------------------------------------------------------
# -------------------------- EXPORT REPORT --------------------------
# -------------------------------------------------------------------
with tabs[6]:
    st.header("📤 Export & Report")

    us_df = us_cpi
    ind_df = ind_cpi_data
    fed_df = fed_bs

    summary = [
        "RBI Macro Dashboard — Summary Report",
        f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"- Latest US CPI: {us_df['value'].iloc[-1]:.2f}" if not us_df.empty else "- US CPI unavailable",
        f"- Latest India CPI: {ind_df['value'].iloc[-1]:.2f}" if not ind_df.empty else "- India CPI unavailable",
        f"- Fed Balance Sheet: {fed_df['value'].iloc[-1]:,.0f}" if not fed_df.empty else "- Fed BS unavailable",
    ]

    st.download_button(
        "Download Summary Report",
        "\n".join(summary).encode(),
        "dashboard_summary.txt"
    )
