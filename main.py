import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from binance import ThreadedWebsocketManager
from binance.client import Client
from sklearn.cluster import KMeans
from alpaca_trade_api.rest import REST  # Alpaca API

# Alpaca API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading endpoint
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters
ATR_LEN = 10
ATR_FACTOR_MIN = 2.0
ATR_FACTOR_MAX = 3.0
FIXED_BUY_VALUE = 100  # $100 per buy
SIGNAL_INTERVAL = 30  # Process signals every 30 seconds
SYMBOL = "BTC/USD"
ALPACA_SYMBOL = SYMBOL.replace("/", "")  # Alpaca doesn't use "/"
BINANCE_SYMBOL = "BTCUSDT"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Change to INFO for cleaner logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Trading bot initialized.")

# Global Variables
volatility = []  # Initialize volatility list
last_price = None
last_direction = 0
last_signal_time = 0  # Track the last processed signal time
high, low, close = [], [], []  # Historical data buffers for real-time updates

# Flask Server
app = Flask(__name__)
@app.route("/")
def home():
    return "Bot is running!", 200

# Perform K-Means Clustering on volatility
def cluster_volatility(volatility, n_clusters=3):
    if len(volatility) < n_clusters:
        logging.error("Not enough data points for clustering. Skipping.")
        return None, None, None, None

    try:
        volatility = np.array(volatility).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(volatility)

        # Centroids and cluster assignments
        centroids = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        # Assign the latest volatility value to a cluster
        latest_volatility = volatility[-1][0]
        assigned_cluster = kmeans.predict([[latest_volatility]])[0]
        assigned_centroid = centroids[assigned_cluster]

        # Calculate cluster sizes
        cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]

        # Calculate volatility level
        volatility_level = assigned_cluster + 1  # Pine uses 1-based indexing

        return centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        return None, None, None, None, None

# Calculate ATR
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=ATR_LEN).mean().iloc[-1]

# Initialize Volatility and Historical Data
def initialize_historical_data():
    """Fetch historical data and calculate ATR values to initialize volatility and price buffers."""
    global high, low, close, volatility
    try:
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="1m", limit=100 + ATR_LEN)
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", 
                                             "close_time", "quote_asset_volume", "number_of_trades",
                                             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        high = data["high"].astype(float).tolist()
        low = data["low"].astype(float).tolist()
        close = data["close"].astype(float).tolist()

        # Calculate ATR for historical data
        tr1 = data["high"].astype(float) - data["low"].astype(float)
        tr2 = abs(data["high"].astype(float) - data["close"].shift(1).astype(float))
        tr3 = abs(data["low"].astype(float) - data["close"].shift(1).astype(float))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=ATR_LEN).mean()

        # Populate the global volatility list
        volatility = atr[-100:].dropna().tolist()
        logging.info(f"Initialized historical data with {len(close)} entries and {len(volatility)} ATR values.")
    except Exception as e:
        logging.error(f"Error initializing historical data: {e}")
        high, low, close, volatility = [], [], [], []

# Calculate SuperTrend with Dynamic ATR Factor
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logging.error("Insufficient market data for SuperTrend calculation. Skipping.")
        return None, None, None, None

    if assigned_centroid is None:
        logging.error("Invalid centroid from clustering. Skipping SuperTrend calculation.")
        return None, None, None, None

    # Dynamic ATR Factor
    atr_factor = max(ATR_FACTOR_MIN, min(ATR_FACTOR_MAX, assigned_centroid / np.mean(volatility)))

    hl2 = (high + low) / 2
    upper_band = hl2 + atr_factor * assigned_centroid
    lower_band = hl2 - atr_factor * assigned_centroid

    # Determine direction
    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1  # Bearish
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1  # Bullish
    else:
        direction = 0  # Neutral

    # Assign SuperTrend based on direction
    if direction == 1:
        supertrend = lower_band.iloc[-1]
    elif direction == -1:
        supertrend = upper_band.iloc[-1]
    else:
        supertrend = None

    return supertrend, direction, upper_band, lower_band

# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close

    try:
        # Extract price data from WebSocket message
        candle = msg['k']
        last_price = float(candle['c'])  # Closing price
        high.append(float(candle['h']))
        low.append(float(candle['l']))
        close.append(float(candle['c']))

        # Ensure lists don't grow indefinitely (keep only the last ATR_LEN + 1 entries)
        if len(high) > ATR_LEN + 1:
            high.pop(0)
        if len(low) > ATR_LEN + 1:
            low.pop(0)
        if len(close) > ATR_LEN + 1:
            close.pop(0)

    except Exception as e:
        logging.error(f"Error processing WebSocket message: {e}")

# Signal Processing Loop
def process_signals():
    """Processes signals at fixed intervals."""
    global last_signal_time
    while True:
        current_time = time.time()
        if current_time - last_signal_time >= SIGNAL_INTERVAL:
            if last_price is not None:
                calculate_and_execute(last_price)
            last_signal_time = current_time
        time.sleep(1)  # Avoid CPU overuse

# Start Flask and WebSocket
if __name__ == "__main__":
    initialize_historical_data()
    Thread(target=process_signals, daemon=True).start()
    start_websocket()


