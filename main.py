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
ATR_FACTOR = 2.7
ALPACA_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Change from DEBUG to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Trading bot initialized.")

# Global Variables
volatility = []  # Initialize volatility list
last_price = None
last_direction = 0
high, low, close = [], [], []  # Historical data buffers for real-time updates
upper_band_history = []  # Store last 4 upper bands
lower_band_history = []  # Store last 4 lower bands
max_history_length = 4  # Maximum history to store

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
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="5m", limit=100 + ATR_LEN)
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
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1  # Bullish
    else:
        direction = 0  # Neutral

    return direction, upper_band, lower_band

# Simplified `calculate_and_execute` with Enhanced Logging and History Comparison
def calculate_and_execute(price):
    global last_direction, upper_band_history, lower_band_history

    if not volatility or len(volatility) < 3:  # Ensure there’s enough data for clustering
        logging.warning("Volatility list is empty or insufficient. Skipping this cycle.")
        return

    # Perform clustering and SuperTrend calculation
    centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
    if assigned_centroid is None:
        logging.warning("Clustering failed. Skipping cycle.")
        return

    # Calculate ATR
    atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))

    # Calculate SuperTrend
    direction, upper_band, lower_band = calculate_supertrend_with_clusters(
        pd.Series(high), pd.Series(low), pd.Series(close), assigned_centroid
    )

    # Update historical band lists
    upper_band_history.append(upper_band.iloc[-1])
    lower_band_history.append(lower_band.iloc[-1])

    # Limit history length to 4
    if len(upper_band_history) > max_history_length:
        upper_band_history.pop(0)
    if len(lower_band_history) > max_history_length:
        lower_band_history.pop(0)

    # Log Metrics
    logging.info(
        f"Price: {price:.2f}, ATR: {atr:.2f}, Direction: {direction}, "
        f"Upper Bands (Last 4): {upper_band_history}, Lower Bands (Last 4): {lower_band_history}"
    )

    # Compare price to historical bands
    for i in range(len(upper_band_history)):
        if price < lower_band_history[i] and last_direction != 1:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "buy")
            last_direction = 1
            break
        elif price > upper_band_history[i] and last_direction != -1:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
            last_direction = -1
            break

# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close

    try:
        if 'k' not in msg:
            return

        candle = msg['k']
        last_price = float(candle['c'])  # Closing price
        high.append(float(candle['h']))
        low.append(float(candle['l']))
        close.append(float(candle['c']))

        if len(high) > ATR_LEN + 1:
            high.pop(0)
        if len(low) > ATR_LEN + 1:
            low.pop(0)
        if len(close) > ATR_LEN + 1:
            close.pop(0)

        calculate_and_execute(last_price)
        time.sleep(5)
    except Exception as e:
        logging.error(f"Error processing WebSocket message: {e}")


# WebSocket Manager
def start_websocket():
    """Starts the Binance WebSocket for real-time price data."""
    while True:  # Keep the WebSocket running
        try:
            logging.info("Starting WebSocket connection...")
            # Initialize ThreadedWebsocketManager
            twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            twm.start()

            # Start streaming 5-minute candles
            twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="5m")
            twm.join()  # Keep the WebSocket connection open
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            logging.info("Reconnecting in 30 seconds...")
            time.sleep(30)  # Wait before reconnecting


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

# Start Flask and WebSocket
if __name__ == "__main__":
    initialize_historical_data()  # Initialize historical data before WebSocket starts
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    start_websocket()