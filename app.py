import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta, date
import os

# Set page configuration
st.set_page_config(page_title="ATH & Yearly High Scanner", layout="wide")

# --- 1. DATA LOADING LAYER (Cached) ---
@st.cache_data(show_spinner=False, ttl=3600) # Cache data for 1 hour
def download_data(tickers):
    """
    Downloads historical data for all tickers.
    Cached to prevent re-downloading when changing date filters.
    """
    if not tickers:
        return pd.DataFrame()
    
    # yfinance requires a space-separated string or list.
    # We pass it through directly.
    try:
        data = yf.download(
            tickers,
            period="max",
            group_by='ticker',
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Download failed: {e}")
        return pd.DataFrame()

# --- 2. LOGIC LAYER (Processing) ---
def process_single_ticker(ticker, df, start_date, end_date):
    """
    Analyzes a single ticker dataframe for ATH and Yearly High breakouts.
    Returns a dict if a breakout is found, else None.
    """
    try:
        # Clean data: Drop rows where OHLC is missing
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty: return None, None
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

        # 1. Calculate Historical Levels
        # Prev 52W High (Yearly High) - Rolling 252 days, shifted by 1 to exclude today
        df['Prev_Yearly_High'] = df['High'].rolling(window=252).max().shift(1)
        # Prev All-Time High - Expanding max, shifted by 1
        df['Prev_ATH'] = df['High'].expanding().max().shift(1)
        
        # 2. Filter for User Date Range
        mask_range = (df.index.date >= start_date) & (df.index.date <= end_date)
        range_df = df.loc[mask_range].copy()
        
        if range_df.empty: return None, None

        # 3. Detect Breakouts
        ath_breakouts = range_df[range_df['High'] > range_df['Prev_ATH']]
        yh_breakouts = range_df[range_df['High'] > range_df['Prev_Yearly_High']]

        # 4. Format Result
        def format_result(breakout_df):
            if breakout_df.empty: return None
            
            # First occurrence in the selected window
            first = breakout_df.iloc[0]
            b_date = breakout_df.index[0].date()
            b_price = first['High']
            
            # Safe latest price fetch
            # We use the original full DF to get the absolute latest data available
            try:
                latest_valid = df.dropna(subset=['Close']).iloc[-1]
                latest_price = latest_valid['Close']
            except:
                latest_price = b_price # Fallback

            ret_pct = ((latest_price - b_price) / b_price) * 100
            
            return {
                "Ticker": ticker.replace(".NS", ""),
                "Breakout Date": b_date,
                "Breakout Price": round(b_price, 2),
                "CMP": round(latest_price, 2),
                "Return (%)": round(ret_pct, 2),
            }

        return format_result(ath_breakouts), format_result(yh_breakouts)

    except Exception as e:
        # print(f"Error processing {ticker}: {e}")
        return None, None

def scan_stocks(tickers, start_date, end_date, progress_bar, status_text):
    """
    Orchestrates the download and processing.
    """
    # 1. Download Data (Cached)
    status_text.text("Downloading/Loading market data... (First run may take time)")
    # Convert list to tuple for caching hashing stability if needed, though list works in modern Streamlit
    raw_data = download_data(tickers)
    
    if raw_data.empty:
        return [], []

    ath_list = []
    yh_list = []
    
    # Safe Ticker Extraction
    # Handles MultiIndex columns safely to avoid failed downloads
    if isinstance(raw_data.columns, pd.MultiIndex):
        downloaded_tickers = list(set([col[0] for col in raw_data.columns]))
    else:
        # Fallback if only 1 ticker or flat structure
        downloaded_tickers = tickers

    total = len(downloaded_tickers)
    
    # 2. Process Data
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzing {ticker}...")
        
        # Extract Ticker DF
        try:
            if total == 1:
                df = raw_data.copy()
            else:
                # Use xs or direct access depending on structure
                # yfinance group_by='ticker' creates (Ticker, OHLC)
                df = raw_data[ticker].copy()
            
            # Run Logic
            ath_res, yh_res = process_single_ticker(ticker, df, start_date, end_date)
            
            if ath_res: ath_list.append(ath_res)
            if yh_res: yh_list.append(yh_res)
            
        except Exception:
            continue
            
    return ath_list, yh_list

# --- 3. HELPER FUNCTIONS ---
def load_tickers():
    file_path = "NiftyTM.txt" 
    tickers = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
    
    if not tickers:
        st.sidebar.warning(f"'{file_path}' not found. Please upload it.")
        uploaded_file = st.sidebar.file_uploader("Upload Ticker List (txt)", type=["txt"])
        if uploaded_file is not None:
            string_data = uploaded_file.read().decode("utf-8")
            tickers = [line.strip() for line in string_data.splitlines() if line.strip()]
    return tickers

def generate_tradingview_batches(data_list, batch_size=30):
    if not data_list: return []
    tv_tickers = [f"NSE:{item['Ticker']}" for item in data_list]
    batches = [tv_tickers[i:i + batch_size] for i in range(0, len(tv_tickers), batch_size)]
    return [", ".join(batch) for batch in batches]

# --- 4. UI SETUP ---
st.title("🚀 ATH & Yearly High Scanner")

TICKER_LIST = load_tickers()

if not TICKER_LIST:
    st.info("Upload 'NiftyTM.txt' to start.")
    st.stop()

# --- DATE PRESETS ---
st.write("### 📅 Select Time Period")
preset = st.radio(
    "Quick Select:", 
    ["Today", "Yesterday", "This Week", "Last Week", "This Month", "Custom"], 
    horizontal=True
)

today = date.today()

if preset == "Today":
    start_d = today
    end_d = today
elif preset == "Yesterday":
    # If today is Monday, Yesterday is Sunday (market closed), so we might want Friday.
    # For simplicity, we just do timedelta(1) and rely on data existence.
    start_d = today - timedelta(days=1)
    end_d = start_d
elif preset == "This Week":
    start_d = today - timedelta(days=today.weekday())
    end_d = today
elif preset == "Last Week":
    start_current = today - timedelta(days=today.weekday())
    start_d = start_current - timedelta(days=7)
    end_d = start_current - timedelta(days=1)
elif preset == "This Month":
    start_d = date(today.year, today.month, 1)
    end_d = today
else: # Custom
    c1, c2 = st.columns(2)
    with c1: start_d = st.date_input("Start Date", value=today - timedelta(days=7))
    with c2: end_d = st.date_input("End Date", value=today)

st.caption(f"🔎 Scanning: **{start_d}** to **{end_d}**")

if st.button("Start Scan", type="primary"):
    bar = st.progress(0)
    status = st.empty()
    
    # Process
    ath_results, yh_results = scan_stocks(TICKER_LIST, start_d, end_d, bar, status)
    
    bar.empty()
    status.empty()
    
    # Display Results
    tab1, tab2 = st.tabs(["🚀 All-Time Highs", "📈 Yearly Highs"])

    # Helper to display tab content
    def show_tab_content(results, label):
        if results:
            st.success(f"Found {len(results)} {label} Breakouts!")
            df = pd.DataFrame(results).sort_values(by="Breakout Date", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader(f"📋 TradingView Watchlist ({label})")
            
            batches = generate_tradingview_batches(results, 30)
            for i, batch in enumerate(batches):
                st.caption(f"Batch {i+1} ({len(batch.split(','))} stocks)")
                st.code(batch, language="text")
        else:
            st.info(f"No stocks hit {label} in this period.")

    with tab1: show_tab_content(ath_results, "ATH")
    with tab2: show_tab_content(yh_results, "Yearly High")