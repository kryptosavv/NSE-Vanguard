import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, date
import os

# --- 1. CONFIGURATION & BRANDING ---
st.set_page_config(
    page_title="NSE Vanguard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {font-size: 3em; font-weight: bold; color: #FF4B4B;}
    .sub-title {font-size: 1.2em; color: #555;}
    .date-banner {
        background-color: #000000; 
        color: #ffffff;
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid #FF4B4B;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Green Background for Found X Metrics */
    .metric-box {
        padding: 10px;
        background-color: #c3e6cb; /* Darker Green */
        color: #0f5132; /* Darker Text */
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #b1dfbb;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING LAYER (Cached) ---
@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers):
    if not tickers:
        return pd.DataFrame()
    
    download_list = tickers + ["^NSEI"]
    download_list = list(set(download_list))
    
    try:
        data = yf.download(
            download_list,
            period="2y", 
            group_by='ticker',
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Download API failed: {e}")
        return pd.DataFrame()

# --- 3. METRIC ENGINE ---
def calculate_advanced_metrics(df, bench_series):
    if df.empty or len(df) < 260: return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Trend Indicators
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    c = close.iloc[-1]
    
    # 52 Week High/Low
    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    h52 = high_52w.iloc[-1]
    l52 = low_52w.iloc[-1]

    # --- A) TREND STRENGTH SCORE (0-100) ---
    s50 = sma50.iloc[-1]
    s150 = sma150.iloc[-1]
    s200 = sma200.iloc[-1]
    
    d50 = (c - s50) / s50
    d150 = (c - s150) / s150
    d200 = (c - s200) / s200
    ma_dist_score = min(max((d50 + d150 + d200) * 200, 0), 50)
    
    spread = (s50 - s200) / s200 if s200 > 0 else 0
    alignment_score = min(max(spread * 300, 0), 30)
    
    s200_prev_20 = sma200.iloc[-20]
    slope = (s200 - s200_prev_20) / s200_prev_20 if s200_prev_20 > 0 else 0
    slope_score = min(max(slope * 500, 0), 20)
    
    trend_score = ma_dist_score + alignment_score + slope_score

    # --- B) RS PERCENTAGE (Score 0-100) ---
    rs_line = close / bench_series
    rs_curr = rs_line.iloc[-1]
    rs_prev_20 = rs_line.iloc[-20]
    rs_mom = (rs_curr - rs_prev_20) / rs_prev_20 if rs_prev_20 > 0 else 0
    rs_mom_norm = min(max(rs_mom * 10, 0), 1) * 100 
    
    rs_raw = rs_curr

    # --- C) TIGHTNESS % (Score 0-100) ---
    def get_range(window):
        h = high.tail(window).max()
        l = low.tail(window).min()
        return (h - l) / h if h > 0 else 1

    r20 = get_range(20)
    r60 = get_range(60)
    compression_ratio = r20 / r60 if r60 > 0 else 1
    tight_score = min(max((1 - compression_ratio) * 100, 0), 100)

    # --- D) VOL DRY SCORE (0-100) ---
    v = volume.iloc[-1]
    v_5d = volume.tail(5).mean()
    v_50d = volume.rolling(50).mean().iloc[-1]
    
    if v_50d > 0:
        dry_ratio = v_5d / v_50d
        dry_score = min(max((1 - dry_ratio) * 100, 0), 50)
        exp_ratio = v / v_50d
        exp_score = min(exp_ratio * 10, 50)
        vol_expansion = exp_ratio 
    else:
        dry_score, exp_score, vol_expansion = 0, 0, 0
        
    vol_score = dry_score + exp_score

    # --- E) NEAR BREAKOUT (Score 0-100) ---
    if h52 > 0:
        dist = (h52 - c) / h52
        readiness_score = min(max((1 - (dist / 0.10)) * 100, 0), 100)
    else: readiness_score = 0

    # --- F) FAILURE RISK (0-100) ---
    failure_score = 0
    if vol_expansion < 1.2: failure_score += 30
    h_day = high.iloc[-1]
    l_day = low.iloc[-1]
    if (h_day - l_day) > 0 and ((c - l_day)/(h_day - l_day)) < 0.5: failure_score += 20
    sma20 = close.rolling(20).mean().iloc[-1]
    if sma20 > 0 and ((c - sma20)/sma20) > 0.15: failure_score += 20
    failure_risk = min(failure_score, 100)

    # --- G) PERSISTENCE (0-100) ---
    persist_score = 0
    up_days = (close.diff() > 0).tail(20).sum()
    persist_score += (up_days / 20) * 40
    if rs_curr > rs_prev_20: persist_score += 20
    higher_highs = (high.diff() > 0).tail(20).sum()
    persist_score += min((higher_highs / 20) * 30, 30)
    if r20 < 0.10: persist_score += 10
    persistence = min(persist_score, 100)

    breakout_20d = c > high.rolling(20).max().shift(1).iloc[-1]

    return {
        "Ticker": "",
        "Price": c,
        "Trend Score": int(trend_score),
        "RS Raw": rs_raw,
        "RS Mom Score": rs_mom_norm,
        "Tightness %": int(tight_score), 
        "Vol Dry Score": int(vol_score),
        "Near Breakout": int(readiness_score),
        "Failure Risk": int(failure_risk),
        "Persistence": int(persistence),
        "Vol Expansion": round(vol_expansion, 2),
        "Breakout 20D": breakout_20d
    }

# --- 4. BREADTH ENGINE ---
def calculate_market_breadth(raw_data, start_date, end_date):
    if isinstance(raw_data.columns, pd.MultiIndex):
        stock_data = raw_data.drop(columns=["^NSEI"], level=0, errors='ignore')
    else: return []

    try:
        closes = stock_data.xs('Close', level=1, axis=1)
        highs = stock_data.xs('High', level=1, axis=1)
        lows = stock_data.xs('Low', level=1, axis=1)
        
        universe_size = closes.shape[1]

        sma20 = closes.rolling(20).mean()
        sma200 = closes.rolling(200).mean()
        above_20dma = (closes > sma20)
        above_200dma = (closes > sma200)
        
        roll_high_252 = highs.rolling(252).max()
        roll_low_252 = lows.rolling(252).min() 
        is_new_high = (highs >= roll_high_252)
        is_new_low = (lows <= roll_low_252)
        
        daily_diff = closes.diff()
        advances = (daily_diff > 0).sum(axis=1)
        declines = (daily_diff < 0).sum(axis=1)
        net_ad = advances - declines
        ad_line = net_ad.cumsum() 
        
        pivot_20 = highs.rolling(20).max().shift(1)
        is_breakout = (closes > pivot_20)
        future_close = closes.shift(-5) 
        is_successful_breakout = (future_close > pivot_20) & is_breakout

        break_below_20dma = (closes < sma20) & (closes.shift(1) > sma20.shift(1))
        ret_20d = closes.pct_change(20)
        positive_20d = (ret_20d > 0)
        thrust_20d = (ret_20d > 0.05)

        mask = (closes.index.date >= start_date) & (closes.index.date <= end_date)
        valid_dates = closes.index[mask]
        
        breadth_records = []
        
        for d in valid_dates:
            total_valid_stocks = closes.loc[d].count()
            if total_valid_stocks == 0: continue
            
            idx_loc = closes.index.get_loc(d)

            slope_val = np.nan
            if idx_loc >= 20:
                y = ad_line.iloc[idx_loc-19 : idx_loc+1].values
                x = np.arange(len(y))
                if len(y) == 20:
                    slope = np.polyfit(x, y, 1)[0]
                    slope_val = (slope / universe_size) * 100 

            if idx_loc >= 20:
                ad_change_20d = ad_line.iloc[idx_loc] - ad_line.iloc[idx_loc - 20]
            else:
                ad_change_20d = np.nan

            bo_count = is_breakout.loc[d].sum()
            if idx_loc <= len(closes.index) - 6 and bo_count > 0:
                bo_success_count = is_successful_breakout.loc[d].sum()
                bo_rate = (bo_success_count / bo_count) * 100
            else:
                bo_rate = np.nan

            breakout_density = (bo_count / total_valid_stocks) * 100
            nh = is_new_high.loc[d].sum()
            nl = is_new_low.loc[d].sum()

            breadth_records.append({
                "Date": d.date(),
                "% > 20 DMA": round((above_20dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "% > 200 DMA": round((above_200dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                
                "New Highs": int(nh),
                "New Lows": int(nl),
                "Net New Highs": int(nh - nl),
                
                "AD Line": int(ad_line.loc[d]),
                "AD Slope 20D": round(slope_val, 2) if not np.isnan(slope_val) else np.nan,
                "AD Change 20D": round(ad_change_20d, 2) if not np.isnan(ad_change_20d) else np.nan,

                "Breakout Count": int(bo_count),
                "Breakout Density %": round(breakout_density, 2),
                "Breakout Success 5D": round(bo_rate, 1) if not np.isnan(bo_rate) else np.nan,
                
                "% Breaking < 20DMA": round((break_below_20dma.loc[d].sum() / total_valid_stocks) * 100, 1),
                "% Positive 20D": round((positive_20d.loc[d].sum() / total_valid_stocks) * 100, 1),
                "% > 5% in 20D": round((thrust_20d.loc[d].sum() / total_valid_stocks) * 100, 1)
            })
            
        return breadth_records
        
    except Exception as e:
        return []

# --- 5. SCANNER ORCHESTRATOR ---
def scan_stocks(tickers, start_date, end_date, progress_bar, status_text):
    status_text.text("🔌 Downloading Data...")
    raw_data = download_data(tickers)
    
    if raw_data.empty:
        st.error("⚠️ Data download failed.")
        return [], [], [], [] 
    
    status_text.text("📊 Calculating Breadth...")
    breadth_data = calculate_market_breadth(raw_data, start_date, end_date)
    
    try:
        bench_data = raw_data["^NSEI"]['Close'] if isinstance(raw_data.columns, pd.MultiIndex) and "^NSEI" in raw_data.columns.levels[0] else pd.Series()
    except: bench_data = pd.Series()

    raw_results = []
    if isinstance(raw_data.columns, pd.MultiIndex):
        downloaded_tickers = list(set([col[0] for col in raw_data.columns]))
    else: downloaded_tickers = tickers
    if "^NSEI" in downloaded_tickers: downloaded_tickers.remove("^NSEI")
    
    total = len(downloaded_tickers)
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            df = raw_data[ticker].copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.copy()
            df = df.loc[:end_date]
            bench_aligned = bench_data.reindex(df.index).ffill()
            
            metrics = calculate_advanced_metrics(df, bench_aligned)
            if metrics:
                metrics["Ticker"] = ticker.replace(".NS", "")
                df['Prev_ATH'] = df['High'].expanding().max().shift(1)
                mask_range = (df.index.date >= start_date) & (df.index.date <= end_date)
                range_df = df.loc[mask_range]
                
                metrics["Is ATH"] = False
                if not range_df.empty:
                    ath_break = range_df[range_df['High'] > range_df['Prev_ATH']]
                    if not ath_break.empty:
                        metrics["Is ATH"] = True
                        metrics["Breakout Date"] = ath_break.index[0].date()
                        evt_price = ath_break.iloc[0]['High']
                        metrics["Return"] = ((metrics["Price"] - evt_price) / evt_price) * 100

                raw_results.append(metrics)
        except: continue

    if not raw_results: return [], [], [], breadth_data
    
    df_res = pd.DataFrame(raw_results)
    
    df_res['RS Percentile'] = df_res['RS Raw'].rank(pct=True) * 100
    df_res['RS %'] = ((df_res['RS Percentile'] * 0.7) + (df_res['RS Mom Score'] * 0.3)).astype(int)
    
    df_res['Rocket Score'] = (
        0.30 * df_res['Trend Score'] +
        0.25 * df_res['RS %'] +
        0.20 * df_res['Tightness %'] +
        0.15 * df_res['Vol Dry Score'] +
        0.10 * df_res['Near Breakout']
    ).round(2)
    
    ath_list = df_res[df_res['Is ATH'] == True].to_dict('records')
    rocket_list = df_res[df_res['Rocket Score'] >= 50].to_dict('records')
    breakout_list = df_res[(df_res['Breakout 20D']) & (df_res['Vol Expansion'] > 1.2)].to_dict('records')
            
    return ath_list, rocket_list, breakout_list, breadth_data

# --- 6. STYLING LOGIC ---
def apply_text_styling(val, mode='standard'):
    if not isinstance(val, (int, float)): return ''
    
    green = 'color: #008000; font-weight: bold;' 
    amber = 'color: #DAA520; font-weight: bold;'
    red = 'color: #FF0000; font-weight: bold;' 
    
    if mode == 'standard':
        if val >= 70: return green
        elif val >= 40: return amber
        else: return red
    elif mode == 'inverse': 
        if val <= 30: return green
        elif val <= 60: return amber
        else: return red
    elif mode == 'return': 
        if val > 0: return green
        else: return red
    elif mode == 'vol_expansion':
        if val >= 2.0: return green
        elif val >= 1.2: return amber
        else: return red
    elif mode == 'bo_success':
        if val >= 60: return green
        elif val >= 40: return amber
        else: return red
    return ''

# --- 7. MAIN UI ---
with st.sidebar:
    st.title("⚙️ Configuration")
    file_path = "NiftyTM.txt" 
    tickers = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f: tickers = [line.strip() for line in f.readlines() if line.strip()]
    if not tickers:
        st.sidebar.warning(f"'{file_path}' not found.")
        uploaded = st.sidebar.file_uploader("Upload Ticker List", type=["txt"])
        if uploaded: tickers = [line.strip() for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]

    st.divider()
    preset = st.radio("Analysis Date:", ["Today", "Yesterday", "This Week", "Last Week", "This Month", "This Year", "Custom"])
    today = date.today()
    if preset == "Today": s_d, e_d = today, today
    elif preset == "Yesterday": s_d, e_d = today-timedelta(1), today-timedelta(1)
    elif preset == "This Week": s_d, e_d = today-timedelta(today.weekday()), today
    elif preset == "Last Week": s_d, e_d = today-timedelta(today.weekday()+7), today-timedelta(today.weekday()+1)
    elif preset == "This Month": s_d, e_d = date(today.year, today.month, 1), today
    elif preset == "This Year": s_d, e_d = date(today.year, 1, 1), today
    else: 
        c1, c2 = st.columns(2)
        with c1: s_d = st.date_input("Start", value=today-timedelta(7))
        with c2: e_d = st.date_input("End", value=today)
    
    st.divider()
    run_btn = st.button("🚀 Run Vanguard Scan", type="primary", use_container_width=True)

st.markdown('<div class="main-title">NSE Vanguard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Institutional Scanner</div>', unsafe_allow_html=True)
st.markdown(f'<div class="date-banner">📅 Period: {s_d} — {e_d}</div>', unsafe_allow_html=True)

if run_btn and tickers:
    bar = st.progress(0)
    status = st.empty()
    ath, rockets, breakouts, breadth = scan_stocks(tickers, s_d, e_d, bar, status)
    bar.empty()
    status.empty()
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ ATH Breakouts", "🚀 Rockets", "💥 Volume Poppers", "📊 Market Breadth"])

    def copy_tv(data):
        if not data: return
        tickers = [f"NSE:{x['Ticker']}" for x in data]
        batches = [", ".join(tickers[i:i+30]) for i in range(0, len(tickers), 30)]
        st.markdown("### 📋 TradingView Watchlist")
        for b in batches: st.code(b, language="text")

    # 1. ATH
    with tab1:
        st.markdown(f"<div class='metric-box'>Found {len(ath)} Stocks</div>", unsafe_allow_html=True)
        if ath:
            df = pd.DataFrame(ath)
            cols = ["Ticker", "Breakout Date", "Price", "Return", "Failure Risk", "Persistence"]
            styled = df[cols].style.format({"Price": "₹ {:.2f}", "Return": "{:.2f}%"})\
                .applymap(lambda v: apply_text_styling(v, 'return'), subset=["Return"])\
                .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["Failure Risk"])\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Persistence"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(ath)
        else: st.info("No ATH Breakouts.")

    # 2. ROCKETS
    with tab2:
        st.markdown(f"<div class='metric-box'>Found {len(rockets)} Rocket Setups</div>", unsafe_allow_html=True)
        st.caption("Criteria: Trend (30%) + RS (25%) + Tightness (20%) + Volume (15%) + Near Breakout (10%)")
        if rockets:
            df = pd.DataFrame(rockets).sort_values(by="Rocket Score", ascending=False)
            cols = ["Ticker", "Price", "Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Failure Risk", "Persistence"]
            styled = df[cols].style.format({"Price": "₹ {:.2f}", "Rocket Score": "{:.2f}"})\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Persistence"])\
                .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["Failure Risk"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(rockets)
        else: st.info("No Rocket Setups found.")

    # 3. VOLUME POPPERS
    with tab3:
        st.markdown(f"<div class='metric-box'>Found {len(breakouts)} Volume Poppers</div>", unsafe_allow_html=True)
        st.caption("Criteria: Price > 20-Day High AND Volume Expansion > 1.2x")
        if breakouts:
            df = pd.DataFrame(breakouts)
            cols = ["Ticker", "Price", "Vol Expansion", "Failure Risk", "Persistence"]
            styled = df[cols].style.format({"Price": "₹ {:.2f}", "Vol Expansion": "{:.2f}x"})\
                .applymap(lambda v: apply_text_styling(v, 'vol_expansion'), subset=["Vol Expansion"])\
                .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["Failure Risk"])\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Persistence"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(breakouts)
        else: st.info("No Volume Poppers found.")

    # 4. MARKET BREADTH
    with tab4:
        st.markdown("### 📊 Market Regime & Internals")
        
        # COMPLETE CHEATSHEET
        with st.expander("ℹ️ Breadth Cheatsheet (Click to Expand)", expanded=True):
            st.markdown("""
            | Metric | Interpretation |
            | :--- | :--- |
            | **New Highs / Lows** | Raw count of stocks hitting 52-Week Highs/Lows. |
            | **Net New Highs** | **Leadership Strength.** Positive = Bullish, Negative = Bearish. |
            | **AD Slope 20D** | **Trend Acceleration.** Positive % means advances are expanding daily. |
            | **AD Change 20D** | **Magnitude.** Raw change in the Cumulative AD Line over 20 days. |
            | **Breakout Density %** | **Thrust.** % of universe breaking out today. High density (>1-2%) = Coordinated move. |
            | **Breakout Success 5D** | **Setup Quality.** > 60% = Breakouts working well. < 40% = High failure rate. |
            | **% > 20 DMA** | **Short-term Participation.** > 70% Overbought, < 30% Oversold. |
            | **% > 200 DMA** | **Structural Regime.** > 50% Bull Market, < 50% Bear Market. |
            | **% Breaking < 20DMA** | **Distribution.** High values indicate institutional selling. |
            | **% Positive 20D** | **Broad Momentum.** % of stocks up over the last month. |
            | **% > 5% in 20D** | **Thrust Strength.** % of stocks with significant gains recently. |
            """, unsafe_allow_html=True)

        if breadth:
            df = pd.DataFrame(breadth)
            
            cols = [
                "Date", "New Highs", "New Lows", "Net New Highs", "AD Slope 20D", "AD Change 20D", 
                "Breakout Density %", "Breakout Success 5D", "% > 20 DMA", "% > 200 DMA"
            ]
            
            def color_breadth(val):
                if isinstance(val, (int, float)):
                    if val > 70: return 'color: #008000; font-weight: bold;'
                    elif val > 40: return 'color: #DAA520; font-weight: bold;'
                    else: return 'color: #FF0000; font-weight: bold;'
                return ''
            
            def color_slope(val):
                return 'color: #008000; font-weight: bold;' if val > 0 else 'color: #FF0000; font-weight: bold;'

            styled = df[cols].style.format({
                "AD Slope 20D": "{:.2f}%", 
                "AD Change 20D": "{:.2f}",
                "Breakout Success 5D": "{:.1f}%",
                "Breakout Density %": "{:.2f}%",
                "% > 20 DMA": "{:.2f}%",
                "% > 200 DMA": "{:.2f}%"
            })\
            .applymap(color_breadth, subset=["% > 20 DMA", "% > 200 DMA"])\
            .applymap(lambda v: apply_text_styling(v, 'bo_success'), subset=["Breakout Success 5D"])\
            .applymap(color_slope, subset=["AD Slope 20D", "Net New Highs", "AD Change 20D"])

            st.dataframe(styled, use_container_width=True, hide_index=True)
        else: st.info("No Breadth Data.")
