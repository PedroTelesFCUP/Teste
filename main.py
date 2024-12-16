import time
import numpy as np
import pandas as pd
import logging
import os
from flask import Flask
from threading import Thread
from binance import ThreadedWebsocketManager
from binance.client import Client
from sklearn.cluster import KMeans
from alpaca_trade_api.rest import REST  # Alpaca API

# Alpaca API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters
ATR_LEN = 10
ATR_FACTOR = 2.7
ALPACA_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)
SIGNAL_INTERVAL = 30

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.log"),
        logging.StreamHandler()
    ]
)

logging.info("Trading bot initialized.")

# Global Variables
volatility = []
last_price = None
last_direction = 0
last_signal_time = 0
high, low, close = [], [], []
upper_band_history, lower_band_history = [], []

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

        centroids = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        latest_volatility = volatility[-1][0]
        assigned_cluster = kmeans.predict([[latest_volatility]])[0]
        assigned_centroid = centroids[assigned_cluster]

        cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]
        volatility_level = assigned_cluster + 1

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

# Calculate SuperTrend
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * assigned_centroid
    lower_band = hl2 - ATR_FACTOR * assigned_centroid

    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1
    else:
        direction = 0

    return None, direction, upper_band, lower_band

# Simplified `calculate_and_execute` with Enhanced Logging and History Comparison
def calculate_and_execute(price):
    global last_direction, upper_band_history, lower_band_history

    if not volatility or len(volatility) < 3:
        logging.warning("Volatility list is empty or insufficient. Skipping this cycle.")
        return

    # Perform clustering
    centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
    if assigned_centroid is None:
        logging.warning("Clustering failed. Skipping cycle.")
        return

    # Calculate ATR and SuperTrend
    atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
    _, direction, upper_band, lower_band = calculate_supertrend_with_clusters(
        pd.Series(high), pd.Series(low), pd.Series(close), assigned_centroid
    )

    # Update historical bands
    upper_band_history.append(float(upper_band.iloc[-1]))
    lower_band_history.append(float(lower_band.iloc[-1]))
    if len(upper_band_history) > 4:
        upper_band_history.pop(0)
    if len(lower_band_history) > 4:
        lower_band_history.pop(0)

    # Log information with improved readability
    logging.info(
        f"\n=== Signal Processing ===\n"
        f"Price: {price:.2f}\n"
        f"ATR: {atr:.2f}\n"
        f"Volatility Level: {volatility_level}\n"
        f"Cluster Centroids: {', '.join(f'{x:.2f}' for x in centroids)}\n"
        f"Cluster Sizes: {', '.join(str(size) for size in cluster_sizes)}\n"
        f"Direction: {'Neutral (0)' if direction == 0 else 'Bullish (1)' if direction == 1 else 'Bearish (-1)'}\n"
        f"Upper Bands (Last 4): {', '.join(f'{x:.2f}' for x in upper_band_history)}\n"
        f"Lower Bands (Last 4): {', '.join(f'{x:.2f}' for x in lower_band_history)}\n"
        f"========================="
    )

    # Execute trades based on signals
    if last_direction == 0 and direction in [1, -1]:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "buy" if direction == 1 else "sell")
    elif last_direction == 1 and direction == -1:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
    elif last_direction == -1 and direction == 1:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "buy")

    last_direction = direction

# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close, volatility

    try:
        if 'k' not in msg:
            return

        candle = msg['k']
        last_price = float(candle['c'])
        high.append(float(candle['h']))
        low.append(float(candle['l']))
        close.append(float(candle['c']))

        if len(high) > ATR_LEN + 1:
            high.pop(0)
        if len(low) > ATR_LEN + 1:
            low.pop(0)
        if len(close) > ATR_LEN + 1:
            close.pop(0)

        atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
        volatility.append(atr)
        if len(volatility) > 100:
            volatility.pop(0)

        calculate_and_execute(last_price)

    except Exception as e:
        logging.error(f"Error processing WebSocket message: {e}")

# Robust WebSocket Start
def start_websocket():
    while True:
        try:
            logging.info("Starting WebSocket connection...")
            twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            twm.start()
            twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="5m")
            twm.join()
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            time.sleep(30)

# Execute a Trade
def execute_trade(symbol, quantity, side):
    try:
        logging.info(f"Submitting {side} order for {quantity} {symbol}.")
        alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        logging.info(f"{side.capitalize()} order submitted successfully.")
    except Exception as e:
        logging.error(f"Error executing {side} order: {e}")

# Signal Processing Loop
def process_signals():
    global last_signal_time
    while True:
        current_time = time.time()
        if current_time - last_signal_time >= SIGNAL_INTERVAL:
            if last_price is not None:
                try:
                    calculate_and_execute(last_price)
                except Exception as e:
                    logging.error(f"Error during signal processing: {e}")
            last_signal_time = current_time
        time.sleep(1)

# Main Script Entry Point
if __name__ == "__main__":
    initialize_historical_data = True  # Historical Data Initialization
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    Thread(target=process_signals, daemon=True).start()
    start_websocket()