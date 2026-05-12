# Stock Market Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-red)](https://keras.io)
[![GPU](https://img.shields.io/badge/GPU-Mixed%20Precision-76b900)](https://www.tensorflow.org/guide/mixed_precision)
[![yfinance](https://img.shields.io/badge/yfinance-live%20data-brightgreen)](https://github.com/ranaroussi/yfinance)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-blueviolet)](https://rajneeshbabu.github.io/stock-market-prediction)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Predict stock closing prices using a **Bidirectional LSTM** neural network trained on real historical OHLCV data (Kaggle dataset) enriched with 10 technical indicators. GPU-accelerated training with mixed precision (float16). Includes a real-time Streamlit app and a GitHub Pages demo.

**Live Demo**: [rajneeshbabu.github.io/stock-market-prediction](https://rajneeshbabu.github.io/stock-market-prediction)

---

## How It Works

```
Historical OHLCV Data (Kaggle — All US Stocks & ETFs)
          |
          v
Feature Engineering — 10 Technical Indicators
  SMA 20/50 · EMA 12/26 · MACD · RSI(14)
  Bollinger Band Width · Return · Volatility
  5-Day Momentum · Volume Ratio
          |
          v
MinMaxScaler → 60-day rolling sequences
          |
          v
Bidirectional LSTM (64→32 units) + BatchNorm + Dropout + L2
          |
          v
Predicted Next-Day Close Price
          |
          v
30-Day Future Forecast
```

---

## Model Architecture

```
Input:  (60 timesteps × 10 features)
        |
Bidirectional LSTM (64 units, return_sequences=True)  →  (60, 128)
BatchNormalization  →  Dropout(0.2)
        |
Bidirectional LSTM (32 units)  →  (64,)
BatchNormalization  →  Dropout(0.2)
        |
Dense(32, relu)  →  Dropout(0.1)  →  Dense(16, relu)  →  Dense(1, float32)
        |
Output: predicted Close price
```

**Training**: Huber loss · Adam (lr=0.001) · EarlyStopping(patience=7) · ReduceLROnPlateau · ModelCheckpoint  
**GPU**: MirroredStrategy + Mixed Precision (float16) — ~2–3× speedup on T4/P100/A100

---

## Performance (AAPL — Real Kaggle Dataset)

### Model Comparison

| Model | RMSE ($) | MAE ($) | MAPE (%) | Direction Accuracy |
|---|---|---|---|---|
| Naive (last price) | 1.63 | 1.17 | 1.02% | 49.1% |
| Vanilla LSTM | 26.99 | 25.15 | 20.75% | 50.3% |
| **Bidirectional LSTM** | **15.99** | **13.54** | **10.81%** | **52.0%** |

> Trained on real AAPL daily OHLCV data from the Kaggle dataset (up to Nov 2017). The Bidirectional LSTM outperforms the Vanilla LSTM by **40%** on RMSE.

---

## Technical Indicators

| Indicator | Description |
|---|---|
| SMA 20 / SMA 50 | Simple Moving Averages — trend direction |
| EMA 12 / EMA 26 | Exponential MAs — faster trend signal |
| MACD | EMA12 − EMA26 — momentum |
| RSI (14) | Relative Strength Index — overbought/oversold |
| Bollinger Band Width | Volatility band spread |
| Daily Return | pct_change(Close) |
| 20-day Volatility | Rolling std of returns |
| 5-Day Momentum | Close / Close.shift(5) − 1 |
| Volume Ratio | Volume vs 20-day average |

---

## GPU Acceleration

The notebook is optimized to run on GPU (Kaggle / Google Colab):

```python
# Auto-detected at runtime — no code changes needed
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu, True)
mixed_precision.set_global_policy('mixed_float16')   # float16 compute, float32 weights
strategy = tf.distribute.MirroredStrategy()           # multi-GPU support
```

| Feature | Benefit |
|---|---|
| Memory growth | Prevents OOM errors |
| Mixed precision (float16) | ~2–3× faster training on NVIDIA GPUs |
| MirroredStrategy | Scales to T4 x2 on Kaggle |
| Larger batch size (64) | Better GPU utilization |

**To enable GPU on Kaggle**: Settings → Accelerator → **GPU T4 x2**  
**To enable GPU on Colab**: Runtime → Change runtime type → **T4 GPU**

---

## Project Structure

```
stock-market-prediction/
├── index.html                       # GitHub Pages demo — no backend needed
├── app.py                           # Streamlit app — real-time yfinance data
├── requirements.txt
├── stock_market_prediction.ipynb    # GPU-ready training notebook (Kaggle/Colab)
├── README.md
│
├── models/
│   ├── lstm_model.keras             # trained Bidirectional LSTM
│   ├── bilstm_best.keras            # best checkpoint (ModelCheckpoint)
│   ├── scaler.pkl                   # MinMaxScaler
│   └── metadata.json                # metrics, feature list, 30-day forecast
│
└── plots/
    ├── price_history.png
    ├── normalised_returns.png
    ├── correlation_heatmap.png
    ├── technical_indicators.png
    ├── training_history.png
    ├── predictions_vs_actual.png
    └── forecast_30days.png
```

---

## Quick Start

### Option 1 — Live GitHub Pages Demo
Visit: **[rajneeshbabu.github.io/stock-market-prediction](https://rajneeshbabu.github.io/stock-market-prediction)**

### Option 2 — Run Streamlit App (real-time data)

```bash
# 1. Clone the repo
git clone https://github.com/rajneeshbabu/stock-market-prediction.git
cd stock-market-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
# → opens at http://localhost:8501
# → pick any ticker (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA...)
# → live technical analysis + LSTM predictions + 30-day forecast
```

### Option 3 — Run / Retrain the Notebook

**On Kaggle (recommended — free GPU):**
1. Go to [kaggle.com](https://kaggle.com) → New Notebook
2. Add dataset: **borismarjanovic/price-volume-data-for-all-us-stocks-etfs**
3. Settings → Accelerator → **GPU T4 x2**
4. Upload `stock_market_prediction.ipynb` → Run All
5. Download `models/` outputs → copy into this repo

**On Google Colab:**
1. Upload `stock_market_prediction.ipynb` to Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run All — GPU is auto-detected and mixed precision enabled

---

## Dataset

- **Source**: [Price/Volume Data for All US Stocks & ETFs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) (Kaggle)
- **Content**: Daily OHLCV + OpenInt for all NYSE, NASDAQ, NYSE MKT stocks and ETFs
- **Period**: Up to November 2017 (Kaggle dataset); yfinance fetches current data for the app
- **Format**: `Date, Open, High, Low, Close, Volume, OpenInt`
- **Stocks used for training**: AAPL, MSFT, GOOGL, AMZN, TSLA

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Bidirectional LSTM (TensorFlow / Keras) |
| GPU Training | MirroredStrategy + Mixed Precision (float16) |
| Features | 10 technical indicators (pandas, numpy) |
| Live Data | yfinance |
| Kaggle Data | kagglehub API |
| Web App | Streamlit |
| Static Demo | Chart.js + GitHub Pages |

---

## License

MIT License — free to use, modify, and distribute.

---

⚠️ **Disclaimer**: This project is for educational purposes only. Stock price predictions are inherently uncertain. Do not use this for real investment decisions.

---

*Built with TensorFlow · Bidirectional LSTM · GPU Mixed Precision · yfinance · Streamlit · Chart.js*
