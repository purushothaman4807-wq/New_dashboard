import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from io import BytesIO
import zipfile

# ------------------------------------------------------------
# PAGE CONFIG (single)
# ------------------------------------------------------------
st.set_page_config(page_title="RBI Macro Dashboard v2.0 — Light Mode", layout="wide",
                   page_icon="🏦", initial_sidebar_state="collapsed")

# ------------------------------------------------------------
# COMMON HELPERS / UTILITIES
# ------------------------------------------------------------
FRED_API_KEY = st.secrets.get("fred_api_key", None)

def get_fred(series_id):
    """Fetch data from FRED API safely (monthly/daily depending on series)."""
    if not FRED_API_KEY:
        return pd.DataFrame(columns=["date", "value"])
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        observations = r.json().get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["date", "value"])
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
        return df[["date", "value"]]
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def india_cpi():
    """Fetch India CPI (World Bank series FP.CPI.TOTL) by year. Returns annual data."""
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL?format=json&per_page=500"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if not payload or len(payload) < 2:
            return pd.DataFrame(columns=["date", "value"])
        data = payload[1]
        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "year", "value": "value"})
        df["date"] = pd.to_datetime(df["year"], format="%Y", errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        df = df[["date", "value"]].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def fetch_usdinr():
    """Fetch USD -> INR latest exchange rate using exchangerate.host (free)."""
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=10)
        r.raise_for_status()
        rate = r.json().get("rates", {}).get("INR", None)
        return rate
    except Exception:
        return None

def linear_forecast(df, periods=12, freq='M'):
    """
    Simple linear forecast using numpy.polyfit.
    Expects df with ['date','value'] sorted asc.
    Returns combined DataFrame with 'is_forecast' boolean.
    """
    if df.empty or len(df) < 3:
        out = df.copy()
        out["is_forecast"] = False
        return out
    df = df.sort_values("date").reset_index(drop=True)
    x = np.array([d.toordinal() for d in df["date"]])
    y = df["value"].values.astype(float)
    p = np.polyfit(x, y, deg=1)
    slope, intercept = p[0], p[1]
    last = df["date"].iloc[-1]
    future_dates = []
    if freq == 'M':
        for i in range(1, periods + 1):
            future_dates.append((last + pd.DateOffset(months=i)).to_pydatetime())
    else:
        for i in range(1, periods + 1):
            future_dates.append((last + timedelta(days=30 * i)).to_pydatetime())
    x_future = np.array([d.toordinal() for d in future_dates])
    y_future = intercept + slope * x_future
    fut_df = pd.DataFrame({"date": future_dates, "value": y_future})
    fut_df["is_forecast"] = True
    hist_df = df.copy()
    hist_df["is_forecast"] = False
    out = pd.concat([hist_df, fut_df], ignore_index=True)
    return out

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def zip_datasets(dict_of_dfs):
    """Return bytes of a ZIP containing the given DataFrames as CSVs."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in dict_of_dfs.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# A safe formatting helper
def safe_latest(df, fmt="0.2f"):
    try:
        # Corrected line: Removed the extra '<' character
        return f"{df['value'].iloc[-1]:{fmt}}"
    except Exception:
        return "N/A"

# ------------------------------------------------------------
# LIGHT THEME GLOBAL STYLES (RBI-inspired colors)
# ------------------------------------------------------------
PRIMARY = "#004D99" # Deep RBI Blue for accents/titles
ACCENT = "#0b84a5"
BG = "#F7F8F4" # Soft Cream Background
CARD_BG = "white"
CARD_TEXT = "#1A1A1A" # Near Black text for visibility
CARD_SUBTITLE = "#545454" # Dark Gray text
CARD_SHADOW = "0 2px 10px rgba(12, 36, 60, 0.06)" # Light subtle shadow
PLOTLY_THEME = "plotly_white" # Light theme for Plotly charts

st.markdown(f"""
<style>
    /* Global App Background and Text Color */
    .stApp {{ background: {BG}; color: {CARD_TEXT}; }}
    
    header .decoration {{ display: none; }}
    
    /* Main Title Styling */
    .big-title {{
        font-size:28px;
        font-weight:700;
        color: {PRIMARY};
        margin-bottom: 0px;
    }}
    /* Subtitle Styling */
    .subtitle {{
        color: {CARD_SUBTITLE};
        margin-top: 0px;
        margin-bottom: 12px;
    }}
    /* Card/Container Styling */
    .card {{
        background: {CARD_BG};
        border-radius:12px;
        padding: 14px;
        box-shadow: {CARD_SHADOW};
        border: 1px solid rgba(0, 0, 0, 0.05); /* subtle light border */
    }}
    /* Metric label color fix for light mode */
    [data-testid="stMetricLabel"] {{ color: {CARD_SUBTITLE} !important; }}
    
    /* Plotly and DataFrame background adjustments for transparency */
    .js-plotly-plot .plotly .main-svg {{ background-color: transparent !important; }}
    .stDataFrame table {{ background-color: {CARD_BG} !important; }}
    
    /* H tags inside tabs for consistency */
    h1, h2, h3, h4 {{ color: {CARD_TEXT}; }}
    /* Metric value color to ensure visibility */
    [data-testid="stMetricValue"] {{ color: {CARD_TEXT}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🏦 RBI Macro Economic Dashboard — v2.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Inflation • Liquidity • Monetary Policy • Forecasts • Exports — polished for RBI application</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TABS: original 7 + new Pro visuals tab (index 7)
# ------------------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Inflation (India + US)",
    "Liquidity (India + US)",
    "Riskometer",
    "Correlations & Forecasts",
    "Yield Curve & Policy",
    "Export / Report",
    "Pro Visuals (3D)"
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
        # Formatting adjusted to ensure readability even if the rate is None
        st.metric("USD → INR (spot)", f"{usd_inr_rate:.4f}" if usd_inr_rate else "N/A")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Quick interpretation")
    st.write("""
    - Use the tabs to view detailed charts, upload India liquidity CSVs, compute correlations, or generate a report.
    - The dashboard is built to be robust when APIs or keys are missing — upload CSVs where possible.
    """)

# -------------------------------------------------------------------
# ----------------------------- INFLATION ----------------------------
# -------------------------------------------------------------------
with tabs[1]:
    st.header("📌 Inflation: US & India")
    col_us, col_ind = st.columns(2)

    with col_us:
        st.subheader("🇺🇸 US CPI (CPIAUCSL)")
        us_df = us_cpi
        if us_df.empty:
            st.info("US CPI not available (FRED key missing or API error). Set `fred_api_key` in Streamlit secrets to enable.")
        else:
            fig_us = px.line(us_df, x="date", y="value", title="US CPI (CPIAUCSL)", labels={"value":"CPI"}, template=PLOTLY_THEME)
            fig_us.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_us, use_container_width=True)
            # forecast
            us_proj = linear_forecast(us_df, periods=12, freq='M')
            fig_proj = px.line(us_proj, x="date", y="value", color="is_forecast",
                               labels={"value":"CPI", "is_forecast":"Forecast (True)"}, title="US CPI + linear forecast (12 months)", template=PLOTLY_THEME)
            fig_proj.update_layout(legend=dict(title="Series"))
            st.plotly_chart(fig_proj, use_container_width=True)

    with col_ind:
        st.subheader("🇮🇳 India CPI (Annual) — World Bank")
        ind_df = ind_cpi_data
        if ind_df.empty:
            st.info("India CPI not available (World Bank API error).")
        else:
            fig_ind = px.line(ind_df, x="date", y="value", title="India CPI (World Bank annual series)", labels={"value":"CPI (avg)"}, template=PLOTLY_THEME)
            fig_ind.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_ind, use_container_width=True)
            ind_proj = linear_forecast(ind_df, periods=5, freq='M')
            st.plotly_chart(px.line(ind_proj, x="date", y="value", color="is_forecast", labels={"value":"CPI"}, title="India CPI + linear forecast (5 periods)", template=PLOTLY_THEME), use_container_width=True)

    st.markdown("### Inflation calculator")
    with st.expander("Calculate future price with constant inflation"):
        initial = st.number_input("Initial Price (₹)", value=100.0)
        inflation = st.number_input("Inflation Rate (%)", value=6.0)
        years = st.number_input("Years", value=5, min_value=1)
        future_price = initial * ((1 + inflation / 100) ** years)
        st.success(f"Future Price after {years} years → ₹{future_price:.2f}")

# -------------------------------------------------------------------
# ----------------------------- LIQUIDITY ----------------------------
# -------------------------------------------------------------------
with tabs[2]:
    st.header("📌 Liquidity — Fed & India (upload)")
    col_a, col_b = st.columns([2,1])

    with col_a:
        st.subheader("🇺🇸 Fed Balance Sheet (WALCL)")
        fed_df = fed_bs
        if fed_df.empty:
            st.info("Fed balance sheet not available (FRED key missing).")
        else:
            fig = px.line(fed_df, x="date", y="value", title="Fed Balance Sheet (WALCL)", template=PLOTLY_THEME)
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Latest (WALCL)", f"${fed_df['value'].iloc[-1]:,.0f}")

    with col_b:
        st.subheader("🇮🇳 India Liquidity — upload CSV")
        st.info("CSV must include `date` and `value` columns. Date can be daily/monthly/year (ISO). Example columns: date,value")
        file = st.file_uploader("Upload India liquidity CSV", type=['csv'], key="india_liq")
        uploaded_df = pd.DataFrame()
        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if 'date' not in df.columns or 'value' not in df.columns:
                    st.error("CSV must contain 'date' and 'value' columns (case-insensitive).")
                else:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['date', 'value']).sort_values('date').reset_index(drop=True)
                    if df.empty:
                        st.error("After parsing, no valid date/value rows found.")
                    else:
                        uploaded_df = df.copy()
                        # Use a light theme compatible chart
                        fig_uploaded = px.line(df, x="date", y="value", template=PLOTLY_THEME)
                        st.plotly_chart(fig_uploaded, use_container_width=True)

                        st.metric("Latest India Liquidity", f"{df['value'].iloc[-1]:,.2f}")
                        st.download_button("Download uploaded CSV", data=df_to_csv_bytes(df), file_name="india_liquidity_uploaded.csv", mime="text/csv")
                        proj = linear_forecast(df, periods=12, freq='M')
                        st.plotly_chart(px.line(proj, x="date", y="value", color="is_forecast", labels={"value":"Liquidity"}, title="India Liquidity + forecast", template=PLOTLY_THEME), use_container_width=True)
            except Exception as e:
                st.error(f"CSV parsing error: {e}")
        else:
            st.info("No file uploaded — upload a CSV with `date` & `value` to visualize India liquidity.")

# -------------------------------------------------------------------
# ---------------------------- RISKOMETER ----------------------------
# -------------------------------------------------------------------
with tabs[3]:
    st.header("📌 Portfolio Riskometer (RBI-style)")
    st.write("Enter allocations (they can be normalized to 100%). Risk score is a simple heuristic — tweak weights if needed.")
    col1, col2, col3 = st.columns(3)
    with col1:
        eq = st.number_input("Equity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, format="%.1f")
    with col2:
        debt = st.number_input("Debt (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, format="%.1f")
    with col3:
        gold = st.number_input("Gold/Commodities (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.1f")

    total = eq + debt + gold
    st.write(f"**Total allocation:** {total:.1f}%")
    if total == 0:
        st.warning("All allocations zero — enter values.")
    else:
        normalize = False
        if abs(total - 100.0) > 1e-6:
            if st.button("Normalize allocations to 100%"):
                normalize = True
        if normalize:
            eq = (eq / total) * 100.0
            debt = (debt / total) * 100.0
            gold = (gold / total) * 100.0
            st.success(f"Allocations normalized → Equity: {eq:.1f}%, Debt: {debt:.1f}%, Gold: {gold:.1f}%")

        # compute risk score
        risk_score = eq * 0.7 + gold * 0.2 + debt * 0.1
        
        # Use a light-mode friendly gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Portfolio Risk Score (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"}
                ],
                "bar": {"color": PRIMARY},
                "threshold": {"value": risk_score, "line": {"color": "red", "width": 4}}
            }
        ))
        # Ensure Plotly layout respects light theme background
        fig.update_layout(template=PLOTLY_THEME, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        st.plotly_chart(fig, use_container_width=True)
        if risk_score < 30:
            st.success(f"Score: {risk_score:.1f} → LOW RISK")
        elif risk_score < 60:
            st.warning(f"Score: {risk_score:.1f} → MODERATE RISK")
        else:
            st.error(f"Score: {risk_score:.1f} → HIGH RISK")

        st.write("Allocation breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"{eq:.1f}%")
        c2.metric("Debt", f"{debt:.1f}%")
        c3.metric("Gold", f"{gold:.1f}%")

# -------------------------------------------------------------------
# ----------------- CORRELATIONS & FORECASTS ------------------------
# -------------------------------------------------------------------
with tabs[4]:
    st.header("📌 Correlations & Simple Forecasts")
    st.write("Upload multiple CSVs (each must have `date` and `value`). We'll merge by date, interpolate, show correlations and simple linear forecasts.")
    uploaded = st.file_uploader("Upload multiple CSVs (hold Ctrl/Cmd)", accept_multiple_files=True, type=['csv'], key="multi")
    dfs = {}
    if uploaded:
        for f in uploaded:
            try:
                name = f.name.rsplit('.', 1)[0]
                d = pd.read_csv(f)
                d.columns = [c.strip().lower() for c in d.columns]
                if 'date' not in d.columns or 'value' not in d.columns:
                    st.warning(f"Skipping {f.name}: needs 'date' & 'value' columns.")
                    continue
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d['value'] = pd.to_numeric(d['value'], errors='coerce')
                d = d.dropna(subset=['date', 'value'])
                d = d[['date', 'value']].rename(columns={'value': name})
                dfs[name] = d
            except Exception as e:
                st.warning(f"Couldn't parse {f.name}: {e}")
        if dfs:
            merged = None
            for name, d in dfs.items():
                if merged is None:
                    merged = d.copy()
                else:
                    merged = pd.merge(merged, d, on='date', how='outer')
            merged = merged.sort_values('date').set_index('date').interpolate().dropna(axis=0, how='all')
            st.write("Merged preview (interpolated):")
            st.dataframe(merged.tail(10))
            if merged.shape[1] > 1:
                corr = merged.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap', template=PLOTLY_THEME, color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload at least 2 series to compute correlation.")
            forecast_horizon = st.slider("Forecast horizon (months)", 1, 24, 12)
            combined = pd.DataFrame(index=merged.index)
            for col in merged.columns:
                s = merged[[col]].dropna().reset_index().rename(columns={'date': 'date', col: 'value'})
                if not s.empty:
                    pf = linear_forecast(s, periods=forecast_horizon)
                    pf = pf.set_index('date')['value'].rename(col)
                    combined = combined.join(pf, how='outer')
            st.line_chart(combined, use_container_width=True) # Use Streamlit's native chart for forecast
            st.download_button("Download merged dataset (CSV)", data=df_to_csv_bytes(merged.reset_index()), file_name="merged_timeseries.csv")
    else:
        st.info("No files uploaded. Try uploading CSVs of liquidity, rates, or other macro series.")

# -------------------------------------------------------------------
# ------------------------ YIELD CURVE ------------------------------
# -------------------------------------------------------------------
with tabs[5]:
    st.header("📉 Yield Curve & Policy Rates")
    st.write("Input or upload policy rates and yield curve points. Use these in analysis and include in your report.")
    with st.expander("Upload policy rates CSV (date, repo, reverse_repo, msf)"):
        policy_file = st.file_uploader("Upload policy rates CSV", type="csv", key="policy")
        if policy_file:
            try:
                pr = pd.read_csv(policy_file)
                pr.columns = [c.strip().lower() for c in pr.columns]
                if 'date' not in pr.columns:
                    st.error("Policy CSV must contain 'date' column.")
                else:
                    pr['date'] = pd.to_datetime(pr['date'], errors='coerce')
                    st.dataframe(pr.head())
                    st.line_chart(pr.set_index('date'))
            except Exception as e:
                st.error(f"Could not parse policy file: {e}")

    st.subheader("Yield curve (manual input)")
    tenor_default = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    cols = st.columns(len(tenor_default))
    yields = []
    for i, t in enumerate(tenor_default):
        with cols[i]:
            y = st.number_input(f"{t} yield (%)", value=7.0, key=f"y_{t}")
            yields.append(y)
    if st.button("Plot yield curve"):
        fig = px.line(x=tenor_default, y=yields, markers=True, title="India Government Bond Yield Curve (manual)", template=PLOTLY_THEME)
        fig.update_yaxes(title_text="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# -------------------------- EXPORT / REPORT ------------------------
# -------------------------------------------------------------------
with tabs[6]:
    st.header("📤 Export & Report")
    st.write("Create a concise text report and bundle datasets into a ZIP for download. Useful for interview submission / portfolio.")
    # Gather latest available series
    us_df = us_cpi
    fed_df = fed_bs
    ind_df = ind_cpi_data
    usd_inr = fetch_usdinr()

    # Build summary lines
    lines = []
    lines.append("RBI Macro Dashboard — Brief Summary")
    lines.append(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    if not us_df.empty:
        lines.append(f"- Latest US CPI (CPIAUCSL): {us_df['value'].iloc[-1]:.2f} (as of {us_df['date'].iloc[-1].date()})")
    else:
        lines.append("- Latest US CPI: not available (FRED key missing or API error)")
    if not ind_df.empty:
        lines.append(f"- Latest India CPI (World Bank annual): {ind_df['value'].iloc[-1]:.2f} (year {ind_df['date'].iloc[-1].year})")
    else:
        lines.append("- Latest India CPI: not available (World Bank API issue)")
    if not fed_df.empty:
        lines.append(f"- Latest Fed Balance Sheet (WALCL): {fed_df['value'].iloc[-1]:,.0f}")
    else:
        lines.append("- Latest Fed Balance Sheet: not available")
    lines.append(f"- USD → INR (spot): {usd_inr:.4f}" if usd_inr else "- USD → INR: not available")

    st.download_button("Download Summary Report", "\n".join(lines).encode(), "dashboard_summary.txt")

# -------------------------------------------------------------------
# ------------------ PRO VISUALS TAB (3D + Light Styling) ------------
# -------------------------------------------------------------------
with tabs[7]:
    # Hardcode the light theme variables for the scoped container look
    PRO_TEXT_COLOR = CARD_TEXT # Use main text color
    PRO_CARD_BG = "rgba(255,255,255,0.92)" # Near opaque white on light background

    # We'll scope the Pro CSS inside a container div with id
    st.markdown("<div id='pro-container'>", unsafe_allow_html=True)
    
    # Top row: title
    st.markdown(f"<h2 style='margin:0; padding:0; color:{PRIMARY};'>🏦 Pro Visuals — 3D Charts (Light Theme)</h2>", unsafe_allow_html=True)

    # Scoped Pro CSS (only styles inside #pro-container)
    pro_css = f"""
    <style>
    /* scope everything to #pro-container to avoid affecting other tabs */
    #pro-container {{
        padding: 8px 4px 24px 4px;
        color: {PRO_TEXT_COLOR};
    }}
    #pro-container .pro-navbar {{
        background: {PRO_CARD_BG} ;
        backdrop-filter: blur(8px);
        padding: 12px;
        border-radius: 12px;
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom:12px;
        border: 1px solid rgba(0,0,0,0.08); /* Darker border for contrast */
    }}
    #pro-container .pro-title {{
        color: {PRIMARY};
        font-weight:600;
        font-size:18px;
    }}
    #pro-container .pro-card {{
        background: {PRO_CARD_BG};
        border-radius:12px;
        padding:14px;
        border: 1px solid rgba(0,0,0,0.08);
        margin-bottom:12px;
    }}
    #pro-container h3 {{ color: {PRO_TEXT_COLOR}; }}
    /* data table style for the pro container */
    #pro-container .stDataFrame table {{ background-color: {PRO_CARD_BG} !important; }}
    </style>
    """
    st.markdown(pro_css, unsafe_allow_html=True)

    # Pro navbar (visual only inside the tab)
    navbar_html = f"""
    <div class="pro-navbar">
      <div class="pro-title">Pro Visuals • Theme: Light Mode</div>
      <div style="color: {CARD_SUBTITLE}; font-size:13px;">3D scatter • 3D surface • Export</div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

    # ---------- Pro data (dummy or derived) ----------
    def generate_series_pro(name, start="2016-01-01", end=datetime.today(), freq="D"):
        dates = pd.date_range(start, end, freq=freq)
        rng = np.random.RandomState(abs(hash(name)) % 1234567)
        values = np.cumsum(rng.normal(loc=0.02, scale=0.8, size=len(dates))) + 100
        if name.lower().startswith("infl"):
            values = values + 2 * np.sin(np.linspace(0, 12 * np.pi, len(dates)))
        return pd.DataFrame({"date": dates, "value": values})

    pro_liq = generate_series_pro("ProLiquidity")
    pro_inf = generate_series_pro("ProInflation")
    pro_gdp = generate_series_pro("ProGDP")

    def to_monthly(df):
        tmp = df.copy()
        tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
        return tmp.groupby("month")["value"].mean().reset_index().rename(columns={"month": "date"})

    pro_liq_m = to_monthly(pro_liq)
    pro_inf_m = to_monthly(pro_inf)
    pro_gdp_m = to_monthly(pro_gdp)

    # KPI row
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns([1,1,1,1])
    with k1:
        st.markdown("<div style='text-align:center; padding:6px;'>", unsafe_allow_html=True)
        st.metric("Liquidity (latest)", safe_latest(pro_liq))
        st.markdown("</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div style='text-align:center; padding:6px;'>", unsafe_allow_html=True)
        st.metric("Inflation (latest)", safe_latest(pro_inf))
        st.markdown("</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div style='text-align:center; padding:6px;'>", unsafe_allow_html=True)
        st.metric("GDP (latest)", safe_latest(pro_gdp))
        st.markdown("</div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div style='text-align:center; padding:6px;'>", unsafe_allow_html=True)
        st.metric("Generated", datetime.utcnow().strftime("%Y-%m-%d"))
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Main Pro layout: left (charts), right (controls & surface)
    pcolL, pcolR = st.columns([2, 1])

    with pcolL:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 8px 0;'>Time Series — Liquidity & Inflation (monthly)</h3>", unsafe_allow_html=True)
        merged_monthly = pd.merge(pro_liq_m.rename(columns={"value": "liquidity"}),
                                  pro_inf_m.rename(columns={"value": "inflation"}),
                                  on="date", how="inner")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=merged_monthly["date"], y=merged_monthly["liquidity"], mode="lines", name="Liquidity", line=dict(width=2)))
        fig_ts.add_trace(go.Scatter(x=merged_monthly["date"], y=merged_monthly["inflation"], mode="lines", name="Inflation", line=dict(width=2, dash="dash")))
        fig_ts.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), template=PLOTLY_THEME, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='pro-card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 8px 0;'>3D Scatter — Liquidity vs Inflation vs Time</h3>", unsafe_allow_html=True)

        months = st.slider("Months for 3D scatter", 12, min(120, len(merged_monthly)), value=60, step=1, key="pro_months")
        df3 = merged_monthly.tail(months).copy().reset_index(drop=True)
        df3["date_ord"] = df3["date"].apply(lambda d: d.toordinal())
        hover_text = df3["date"].dt.strftime("%Y-%m") + "<br>Liquidity: " + df3["liquidity"].round(2).astype(str) + "<br>Inflation: " + df3["inflation"].round(2).astype(str)

        scatter3d = go.Figure(data=[go.Scatter3d(
            x=df3["date_ord"],
            y=df3["liquidity"],
            z=df3["inflation"],
            mode='markers+lines',
            marker=dict(size=4, opacity=0.85, color=PRIMARY),
            line=dict(width=1, color=PRIMARY),
            text=hover_text,
            hoverinfo='text'
        )])
        scatter3d.update_layout(
            scene=dict(
                xaxis_title='Date',
                yaxis_title='Liquidity',
                zaxis_title='Inflation',
                xaxis=dict(
                    tickvals=df3["date_ord"][::max(1, len(df3)//6)],
                    ticktext=df3["date"].dt.strftime("%Y-%m")[::max(1, len(df3)//6)]
                )
            ),
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            template=PLOTLY_THEME,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(scatter3d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pcolR:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 8px 0;'>Controls & 3D Surface</h3>", unsafe_allow_html=True)
        smooth = st.slider("Surface smoothing (size)", 3, 18, value=6, key="pro_smooth")

        df_grid = pd.DataFrame({
            "date": pro_liq_m["date"],
            "liquidity": pro_liq_m["value"],
            "inflation": pro_inf_m["value"],
            "gdp": pro_gdp_m["value"]
        }).dropna().reset_index(drop=True)
        df_grid = df_grid.iloc[::max(1, len(df_grid)//120)].reset_index(drop=True)

        x = np.arange(len(df_grid))
        y = np.linspace(df_grid["gdp"].min(), df_grid["gdp"].max(), max(8, smooth))
        X, Y = np.meshgrid(x, y)
        liquidity_interp = np.interp(X[0], x, df_grid["liquidity"].values)
        Z = np.tile(liquidity_interp, (Y.shape[0], 1)) + (np.sin((Y - Y.mean()) / (Y.std() if Y.std()!=0 else 1)) * 2)

        surface = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='RdBu', showscale=False, opacity=0.9)])
        surface.update_layout(
            title="3D Surface (toy model): Liquidity over Time × GDP",
            scene=dict(xaxis_title="Time Index", yaxis_title="GDP (synthetic)", zaxis_title="Liquidity"),
            height=520, margin=dict(l=0, r=0, t=40, b=0),
            template=PLOTLY_THEME,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(surface, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlations & export inside pro tab
    st.markdown("<div class='pro-card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 8px 0;'>Correlations & Export</h3>", unsafe_allow_html=True)
    pro_corr = pd.concat([
        pro_liq_m.rename(columns={"value": "liquidity"}).set_index("date")["liquidity"],
        pro_inf_m.rename(columns={"value": "inflation"}).set_index("date")["inflation"],
        pro_gdp_m.rename(columns={"value": "gdp"}).set_index("date")["gdp"]
    ], axis=1).dropna()
    corr = pro_corr.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", template=PLOTLY_THEME)
    fig_corr.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    ca, cb = st.columns([3,1])
    with ca:
        st.write("### Monthly series preview (latest)")
        st.dataframe(pro_corr.tail(10).reset_index().rename(columns={"index":"date"}))
    with cb:
        st.write("### Export")
        csv_bytes = pro_corr.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download merged CSV", csv_bytes, file_name="pro_merged_monthly_series.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    # close pro container
    st.markdown("</div>", unsafe_allow_html=True)
