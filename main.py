import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from binance.client import Client
from sklearn.cluster import KMeans

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters
ATR_LEN = 10
ATR_FACTOR = 3.0
SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Trading bot initialized.")

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
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]

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

# Initialize Volatility from Historical Data
def initialize_volatility_from_history():
    """Fetch historical data and calculate ATR values to initialize volatility."""
    try:
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="1m", limit=100 + ATR_LEN)
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", 
                                             "close_time", "quote_asset_volume", "number_of_trades",
                                             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        # Calculate ATR for historical data
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=ATR_LEN).mean()

        # Return the last 100 ATR values as volatility
        volatility = atr[-100:].dropna().tolist()
        logging.info(f"Initialized volatility with {len(volatility)} historical ATR values.")
        return volatility
    except Exception as e:
        logging.error(f"Error initializing volatility: {e}")
        return []

# Rest of the script remains unchanged (refer to the previous trading_bot logic)

if __name__ == "__main__":
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    trading_bot()

