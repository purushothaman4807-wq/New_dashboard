# app.py (enhanced RBI Macro Dashboard v2.0)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from io import BytesIO
import base64
import zipfile

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="RBI Macro Dashboard v2.0", layout="wide",
                   page_icon="🏦", initial_sidebar_state="collapsed")

# ---------- STYLES & THEME ----------
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
    .metric-label {{ color: #666; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

# Top header
st.markdown("<div class='big-title'>🏦 RBI Macro Economic Dashboard — v2.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Inflation • Liquidity • Monetary Policy • Forecasts • Exports — polished for RBI application</div>", unsafe_allow_html=True)

# ---------- CONFIG / KEYS ----------
FRED_API_KEY = st.secrets.get("fred_api_key")  # optional: set this in Streamlit secrets to enable FRED

# ---------- HELPERS ----------
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

def make_text_report(lines):
    return "\n".join(lines).encode("utf-8")

def zip_datasets(dict_of_dfs):
    """Return bytes of a ZIP containing the given DataFrames as CSVs."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in dict_of_dfs.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ---------- LAYOUT: TABS ----------
tabs = st.tabs([
    "Overview",
    "Inflation (India + US)",
    "Liquidity (India + US)",
    "Riskometer",
    "Correlations & Forecasts",
    "Yield Curve & Policy",
    "Export / Report"
])

# ---------- TAB: OVERVIEW ----------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dashboard snapshot")
    col1, col2, col3, col4 = st.columns(4)

    # fetch a few series (non-blocking: will return empty df if no key)
    us_cpi = get_fred("CPIAUCSL")
    fed_bs = get_fred("WALCL")
    ind_cpi = india_cpi()
    usd_inr_rate = fetch_usdinr()

    def safe_latest(df, fmt="0.2f"):
        try:
            return f"{df['value'].iloc[-1]:.{fmt.split('.')[-1]}"
        except Exception:
            return "N/A"

    # metrics
    with col1:
        if not us_cpi.empty:
            st.metric("US CPI (latest)", f"{us_cpi['value'].iloc[-1]:.2f}", delta=None)
        else:
            st.metric("US CPI (latest)", "N/A")
    with col2:
        if not ind_cpi.empty:
            st.metric("India CPI (annual)", f"{ind_cpi['value'].iloc[-1]:.2f}")
        else:
            st.metric("India CPI (annual)", "N/A")
    with col3:
        if not fed_bs.empty:
            fed_latest = fed_bs['value'].iloc[-1]
            # usually in dollars - scale
            st.metric("Fed Balance Sheet (WALCL)", f"{fed_latest:,.0f}")
        else:
            st.metric("Fed Balance Sheet (WALCL)", "N/A")
    with col4:
        st.metric("USD → INR (spot)", f"{usd_inr_rate:.2f}" if usd_inr_rate else "N/A")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Quick interpretation")
    st.write("""
    - Use the tabs to view detailed charts, upload India liquidity CSVs, compute correlations, or generate a report.
    - The dashboard is built to be robust when APIs or keys are missing — upload CSVs where possible.
    """)

# ---------- TAB: INFLATION ----------
with tabs[1]:
    st.header("📌 Inflation: US & India")
    col_us, col_ind = st.columns(2)

    with col_us:
        st.subheader("🇺🇸 US CPI (CPIAUCSL)")
        us_df = us_cpi
        if us_df.empty:
            st.info("US CPI not available (FRED key missing or API error). Set `fred_api_key` in Streamlit secrets to enable.")
        else:
            fig_us = px.line(us_df, x="date", y="value", title="US CPI (CPIAUCSL)", labels={"value":"CPI"})
            fig_us.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_us, use_container_width=True)
            # forecast
            us_proj = linear_forecast(us_df, periods=12, freq='M')
            fig_proj = px.line(us_proj, x="date", y="value", color="is_forecast",
                               labels={"value":"CPI", "is_forecast":"Forecast (True)"}, title="US CPI + linear forecast (12 months)")
            fig_proj.update_layout(legend=dict(title="Series"))
            st.plotly_chart(fig_proj, use_container_width=True)

    with col_ind:
        st.subheader("🇮🇳 India CPI (Annual) — World Bank")
        ind_df = ind_cpi
        if ind_df.empty:
            st.info("India CPI not available (World Bank API error).")
        else:
            fig_ind = px.line(ind_df, x="date", y="value", title="India CPI (World Bank annual series)", labels={"value":"CPI (avg)"})
            fig_ind.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_ind, use_container_width=True)
            ind_proj = linear_forecast(ind_df, periods=5, freq='M')
            st.plotly_chart(px.line(ind_proj, x="date", y="value", color="is_forecast", labels={"value":"CPI"}, title="India CPI + linear forecast (5 periods)"), use_container_width=True)

    st.markdown("### Inflation calculator")
    with st.expander("Calculate future price with constant inflation"):
        initial = st.number_input("Initial Price (₹)", value=100.0)
        inflation = st.number_input("Inflation Rate (%)", value=6.0)
        years = st.number_input("Years", value=5, min_value=1)
        future_price = initial * ((1 + inflation / 100) ** years)
        st.success(f"Future Price after {years} years → ₹{future_price:.2f}")

# ---------- TAB: LIQUIDITY ----------
with tabs[2]:
    st.header("📌 Liquidity — Fed & India (upload)")
    col_a, col_b = st.columns([2,1])

    with col_a:
        st.subheader("🇺🇸 Fed Balance Sheet (WALCL)")
        fed_df = fed_bs
        if fed_df.empty:
            st.info("Fed balance sheet not available (FRED key missing).")
        else:
            fig = px.line(fed_df, x="date", y="value", title="Fed Balance Sheet (WALCL)")
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
                        st.line_chart(df.set_index("date"))
                        st.metric("Latest India Liquidity", f"{df['value'].iloc[-1]:,.2f}")
                        st.download_button("Download uploaded CSV", data=df_to_csv_bytes(df), file_name="india_liquidity_uploaded.csv", mime="text/csv")
                        proj = linear_forecast(df, periods=12, freq='M')
                        st.plotly_chart(px.line(proj, x="date", y="value", color="is_forecast", labels={"value":"Liquidity"}, title="India Liquidity + forecast"), use_container_width=True)
            except Exception as e:
                st.error(f"CSV parsing error: {e}")
        else:
            st.info("No file uploaded — upload a CSV with `date` & `value` to visualize India liquidity.")

# ---------- TAB: RISKOMETER ----------
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
                "bar": {"color": "darkblue"},
                "threshold": {"value": risk_score, "line": {"color": "red", "width": 4}}
            }
        ))
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

# ---------- TAB: CORRELATIONS & FORECASTS ----------
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
                fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap')
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
            st.line_chart(combined)
            st.download_button("Download merged dataset (CSV)", data=df_to_csv_bytes(merged.reset_index()), file_name="merged_timeseries.csv")
    else:
        st.info("No files uploaded. Try uploading CSVs of liquidity, rates, or other macro series.")

# ---------- TAB: YIELD CURVE & POLICY ----------
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
        fig = px.line(x=tenor_default, y=yields, markers=True, title="India Government Bond Yield Curve (manual)")
        fig.update_yaxes(title_text="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB: EXPORT / REPORT ----------
with tabs[6]:
    st.header("📤 Export & Report")
    st.write("Create a concise text report and bundle datasets into a ZIP for download. Useful for interview submission / portfolio.")

    # Gather latest available series
    us_df = us_cpi
    fed_df = fed_bs
    ind_df = ind_cpi
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
    if not fed_df.em
