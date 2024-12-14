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
ATR_FACTOR = 3.0
SYMBOL = "BTC/USD"
ALPACA_SYMBOL = SYMBOL.replace("/", "")  # Alpaca doesn't use "/"
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

# Calculate SuperTrend
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logging.error("Insufficient market data for SuperTrend calculation. Skipping.")
        return None, None, None, None

    if assigned_centroid is None:
        logging.error("Invalid centroid from clustering. Skipping SuperTrend calculation.")
        return None, None, None, None

    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * assigned_centroid
    lower_band = hl2 - ATR_FACTOR * assigned_centroid

    # Determine direction
    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1  # Bearish
        logging.info(f"Direction: Bearish (-1). Price ({close.iloc[-1]}) > Upper Band ({upper_band.iloc[-1]}).")
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1  # Bullish
        logging.info(f"Direction: Bullish (1). Price ({close.iloc[-1]}) < Lower Band ({lower_band.iloc[-1]}).")
    else:
        direction = 0  # Neutral
        logging.info(f"Direction: Neutral (0). Price ({close.iloc[-1]}) within bands: Lower Band ({lower_band.iloc[-1]}), Upper Band ({upper_band.iloc[-1]}).")

    # Assign SuperTrend based on direction
    if direction == 1:
        supertrend = lower_band.iloc[-1]
    elif direction == -1:
        supertrend = upper_band.iloc[-1]
    else:
        supertrend = None
        logging.info("Neutral direction. SuperTrend is None. No action required.")

    return supertrend, direction, upper_band, lower_band

# WebSocket Handler
def on_message(msg):
    global last_price
    last_price = float(msg['k']['c'])  # Closing price from candlestick data

    # Perform real-time SuperTrend calculation
    calculate_and_execute(last_price)

def start_websocket():
    """Starts the Binance WebSocket for real-time price data."""
    try:
        twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
        twm.start()
        twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="1m")
        twm.join()
    except Exception as e:
        logging.error(f"WebSocket connection failed: {e}")
        raise

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

# Calculate and Execute Trades
def calculate_and_execute(price):
    global last_direction

    # Perform clustering and SuperTrend calculation
    centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
    if assigned_centroid is None:
        logging.warning("Clustering failed. Skipping cycle.")
        return

    supertrend, direction, upper_band, lower_band = calculate_supertrend_with_clusters(
        high, low, close, assigned_centroid
    )

    # Log details
    logging.info(f"Price: {price}, SuperTrend: {supertrend}, Direction: {direction}")

    # Handle signal changes
    if last_direction == 0 and direction in [1, -1]:
        trade_type = "buy" if direction == 1 else "sell"
        execute_trade(ALPACA_SYMBOL, QUANTITY, trade_type)
    elif last_direction == 1 and direction == -1:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
    elif last_direction == -1 and direction == 1:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "buy")

    last_direction = direction

# Start Flask and WebSocket
if __name__ == "__main__":
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    start_websocket()


