import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os, warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f2027 100%); min-height: 100vh; }
.block-container { padding: 1.5rem 2rem; max-width: 1300px; }
.main-title {
    text-align: center; font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, #56CCF2, #2F80ED, #56CCF2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { text-align: center; color: #aaa; font-size: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 800; color: #56CCF2; }
.metric-lbl { font-size: 0.78rem; color: #aaa; margin-top: 0.2rem; }
.up   { color: #2ecc71 !important; }
.down { color: #e74c3c !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ticker = st.selectbox("Stock Ticker", ['AAPL','MSFT','GOOGL','AMZN','TSLA','NVDA','META','BRK-B','JPM','V'])
    period = st.selectbox("Historical Period", ['2y','3y','5y','10y'], index=1)
    n_forecast = st.slider("Forecast Days", 7, 60, 30)
    seq_len = 60
    st.markdown("---")
    st.markdown("**Model**: Bidirectional LSTM")
    st.markdown("**Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, Momentum, Volume")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, period):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('models/lstm_model.keras')
        return model
    except Exception:
        return None


# ── Technical indicators ───────────────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    d['SMA_20'] = d['Close'].rolling(20).mean()
    d['SMA_50'] = d['Close'].rolling(50).mean()
    d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']   = d['EMA_12'] - d['EMA_26']
    d['Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['Signal']
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    d['BB_Mid']   = d['Close'].rolling(20).mean()
    d['BB_Std']   = d['Close'].rolling(20).std()
    d['BB_Upper'] = d['BB_Mid'] + 2 * d['BB_Std']
    d['BB_Lower'] = d['BB_Mid'] - 2 * d['BB_Std']
    d['BB_Width'] = (d['BB_Upper'] - d['BB_Lower']) / (d['BB_Mid'] + 1e-9)
    d['Return']     = d['Close'].pct_change()
    d['Volatility'] = d['Return'].rolling(20).std()
    d['Momentum_5'] = d['Close'] / d['Close'].shift(5) - 1
    d['Vol_Ratio']  = d['Volume'] / (d['Volume'].rolling(20).mean() + 1)
    return d.dropna()


FEATURES = ['Close','SMA_20','SMA_50','MACD','RSI','BB_Width',
            'Return','Volatility','Momentum_5','Vol_Ratio']


def make_prediction(df_feat, model, n_days):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat[FEATURES].values)

    # Test set predictions (last 20%)
    X_seq, y_seq = [], []
    for i in range(seq_len, len(scaled)):
        X_seq.append(scaled[i-seq_len:i])
        y_seq.append(scaled[i, 0])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    split = int(len(X_seq) * 0.8)
    X_test, y_test = X_seq[split:], y_seq[split:]

    def inv_close(arr):
        dummy = np.zeros((len(arr), len(FEATURES)))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_prices = inv_close(pred_scaled)
    true_prices = inv_close(y_test)

    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae  = mean_absolute_error(true_prices, pred_prices)
    mape = np.mean(np.abs((true_prices - pred_prices) / (true_prices + 1e-9))) * 100
    dacc = np.mean(np.sign(np.diff(true_prices)) == np.sign(np.diff(pred_prices))) * 100

    # Future forecast
    last_seq = scaled[-seq_len:].copy()
    future_scaled = []
    for _ in range(n_days):
        x_in = last_seq.reshape(1, seq_len, len(FEATURES))
        p = model.predict(x_in, verbose=0)[0, 0]
        future_scaled.append(p)
        new_row = last_seq[-1].copy(); new_row[0] = p
        last_seq = np.vstack([last_seq[1:], [new_row]])
    future_prices = inv_close(np.array(future_scaled))
    future_dates  = pd.date_range(df_feat.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='B')

    return (pred_prices, true_prices, df_feat.index[-len(true_prices):],
            future_prices, future_dates, rmse, mae, mape, dacc)


# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>📈 Stock Market Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Bidirectional LSTM · 10 Technical Indicators · Real-Time Data via yfinance</div>",
            unsafe_allow_html=True)
st.markdown("---")

with st.spinner(f"Fetching {ticker} data..."):
    df_raw = load_data(ticker, period)

if df_raw.empty:
    st.error("Could not fetch data. Check your internet connection.")
    st.stop()

df_feat = add_indicators(df_raw)
model   = load_model()

if model is None:
    st.warning("Pre-trained model not found. Run the notebook first to generate `models/lstm_model.keras`.")
    st.stop()

# ── Run predictions ────────────────────────────────────────────────────────────
with st.spinner("Running LSTM predictions..."):
    pred_p, true_p, test_dates, fut_p, fut_dates, rmse, mae, mape, dacc = \
        make_prediction(df_feat, model, n_forecast)

# ── Metrics row ────────────────────────────────────────────────────────────────
last_close = float(df_raw['Close'].iloc[-1])
prev_close = float(df_raw['Close'].iloc[-2])
day_chg    = (last_close - prev_close) / prev_close * 100
fut_chg    = (fut_p[-1] - last_close) / last_close * 100
chg_color  = "up" if day_chg >= 0 else "down"
fut_color  = "up" if fut_chg >= 0 else "down"

c1,c2,c3,c4,c5 = st.columns(5)
for col, val, lbl in [
    (c1, f"${last_close:.2f}", f"Last Close ({ticker})"),
    (c2, f"{'▲' if day_chg>=0 else '▼'} {abs(day_chg):.2f}%", "Day Change"),
    (c3, f"${rmse:.2f}", "Test RMSE"),
    (c4, f"{mape:.2f}%", "Test MAPE"),
    (c5, f"{dacc:.1f}%", "Direction Accuracy"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "🤖 LSTM Predictions", "🔮 Forecast"])

# TAB 1: Technical Analysis
with tab1:
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                              facecolor='#0f0f23')
    for ax in axes: ax.set_facecolor('#1a1a3e'); ax.tick_params(colors='#aaa'); ax.spines['bottom'].set_color('#333')

    recent = df_feat.tail(365)

    axes[0].plot(recent.index, recent['Close'],   color='#56CCF2', linewidth=1.5, label='Close')
    axes[0].plot(recent.index, recent['SMA_20'],  color='#F2994A', linewidth=1,   label='SMA 20')
    axes[0].plot(recent.index, recent['SMA_50'],  color='#9B59B6', linewidth=1,   label='SMA 50')
    axes[0].fill_between(recent.index, recent['BB_Upper'], recent['BB_Lower'],
                         alpha=0.1, color='white', label='Bollinger Bands')
    axes[0].set_title(f'{ticker} — Price & Moving Averages', color='white', fontsize=11)
    axes[0].legend(fontsize=8, labelcolor='white', facecolor='#1a1a3e')
    axes[0].yaxis.label.set_color('white')

    axes[1].plot(recent.index, recent['MACD'],   color='#56CCF2', linewidth=1, label='MACD')
    axes[1].plot(recent.index, recent['Signal'], color='#E74C3C', linewidth=1, label='Signal')
    axes[1].bar(recent.index, recent['MACD_Hist'],
                color=['#2ecc71' if x >= 0 else '#e74c3c' for x in recent['MACD_Hist']], alpha=0.7)
    axes[1].set_title('MACD', color='white', fontsize=11)
    axes[1].legend(fontsize=8, labelcolor='white', facecolor='#1a1a3e')

    axes[2].plot(recent.index, recent['RSI'], color='#8172B2', linewidth=1.5)
    axes[2].axhline(70, color='#e74c3c', linestyle='--', linewidth=0.8)
    axes[2].axhline(30, color='#2ecc71', linestyle='--', linewidth=0.8)
    axes[2].axhline(50, color='gray',    linestyle=':', linewidth=0.6)
    axes[2].set_ylim(0, 100)
    axes[2].set_title('RSI (14)', color='white', fontsize=11)

    axes[3].bar(recent.index, recent['Volume'] / 1e6,
                color=['#2ecc71' if r >= 0 else '#e74c3c' for r in recent['Return']], alpha=0.8)
    axes[3].plot(recent.index, recent['Volume'].rolling(20).mean() / 1e6,
                 color='#F2994A', linewidth=1.2)
    axes[3].set_title('Volume (M)', color='white', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# TAB 2: LSTM Predictions
with tab2:
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0f0f23')
    ax.set_facecolor('#1a1a3e')
    ax.plot(test_dates, true_p,  color='#56CCF2', linewidth=1.5, label='Actual')
    ax.plot(test_dates, pred_p,  color='#E74C3C', linewidth=1.5, linestyle='--', label='Predicted')
    ax.set_title(f'{ticker} — LSTM: Predicted vs Actual (Test Set)', color='white', fontsize=12)
    ax.set_ylabel('Price (USD)', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.legend(labelcolor='white', facecolor='#1a1a3e', fontsize=10)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    r1, r2, r3, r4 = st.columns(4)
    for col, val, lbl in [(r1,f'${rmse:.2f}','RMSE'),(r2,f'${mae:.2f}','MAE'),
                           (r3,f'{mape:.2f}%','MAPE'),(r4,f'{dacc:.1f}%','Direction Accuracy')]:
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>",
                        unsafe_allow_html=True)

# TAB 3: Forecast
with tab3:
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0f0f23')
    ax.set_facecolor('#1a1a3e')
    hist = df_feat['Close'].iloc[-120:]
    ax.plot(hist.index, hist.values,        color='#56CCF2', linewidth=1.5, label='Historical')
    ax.plot(fut_dates,  fut_p,              color='#F2994A', linewidth=2, linestyle='--', label=f'{n_forecast}-Day Forecast')
    ax.fill_between(fut_dates, fut_p*0.95, fut_p*1.05, alpha=0.2, color='#F2994A', label='±5% Band')
    ax.axvline(df_feat.index[-1], color='gray', linestyle=':', linewidth=1)
    ax.set_title(f'{ticker} — {n_forecast}-Day Price Forecast', color='white', fontsize=12)
    ax.set_ylabel('Price (USD)', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.legend(labelcolor='white', facecolor='#1a1a3e', fontsize=10)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"**Forecast summary:** Current price **${last_close:.2f}** → predicted **${fut_p[-1]:.2f}** "
                f"({'▲' if fut_chg>=0 else '▼'} {abs(fut_chg):.2f}% over {n_forecast} days)")
    fc_df = pd.DataFrame({'Date': [str(d.date()) for d in fut_dates],
                          'Forecast Price': [f'${p:.2f}' for p in fut_p]})
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:0.78rem'>"
            "Built with TensorFlow · Bidirectional LSTM · yfinance · Streamlit | Not financial advice</p>",
            unsafe_allow_html=True)
