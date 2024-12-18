import os
import time
import numpy as np
import pandas as pd
import logging
import datetime
from flask import Flask, send_file
from threading import Thread
from binance import ThreadedWebsocketManager
from binance.client import Client
from sklearn.cluster import KMeans
from alpaca_trade_api.rest import REST  # Alpaca API

def restart_program():
    """Restart the current program."""
    try:
        print("Restarting program...")
        time.sleep(2)  # Optional delay
        os.execv(sys.executable, ['python'] + sys.argv)
    except Exception as e:
        print(f"Failed to restart program: {e}")


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
ATR_FACTOR = 3.0
ALPACA_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)
SIGNAL_INTERVAL = 300  # Seconds

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.log"),  # Save logs to file
        logging.StreamHandler()              # Output logs to console
    ]
)
logging.info("Trading bot initialized.")

# Global Variables
volatility = []
last_price = None
last_direction = 0
last_signal_time = 0
last_log_time = 0  # New variable for heartbeat logs
high, low, close = [], [], []
upper_band_history = []
lower_band_history = []

# Flask Server
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

@app.route('/logs')
def download_logs():
    try:
        # Use the absolute path to ensure correct file location
        return send_file("bot_logs.log", as_attachment=True)
    except FileNotFoundError:
        return "Log file not found.", 404

# Perform K-Means Clustering on volatility
def cluster_volatility(volatility, n_clusters=3):
    if len(volatility) < n_clusters:
        logging.warning("Not enough data points for clustering. Skipping.")
        return None, None, None, None, None

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

# Initialize Volatility and Historical Data
def initialize_historical_data():
    global high, low, close, volatility
    try:
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="5m", limit=100 + ATR_LEN)
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume",
                                             "close_time", "quote_asset_volume", "number_of_trades",
                                             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        high = data["high"].astype(float).tolist()
        low = data["low"].astype(float).tolist()
        close = data["close"].astype(float).tolist()

        tr1 = data["high"].astype(float) - data["low"].astype(float)
        tr2 = abs(data["high"].astype(float) - data["close"].shift(1).astype(float))
        tr3 = abs(data["low"].astype(float) - data["close"].shift(1).astype(float))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=ATR_LEN).mean()

        volatility = atr[-100:].dropna().tolist()
        logging.info(f"Initialized historical data with {len(close)} entries and {len(volatility)} ATR values.")
    except Exception as e:
        logging.error(f"Error initializing historical data: {e}")
        high, low, close, volatility = [], [], [], []

# Calculate SuperTrend
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * assigned_centroid
    lower_band = hl2 - ATR_FACTOR * assigned_centroid

    prev_upper_band = upper_band.shift(1)
    prev_lower_band = lower_band.shift(1)
    lower_band = lower_band.where(lower_band > prev_lower_band, prev_lower_band)
    upper_band = upper_band.where(upper_band < prev_upper_band, prev_upper_band)

    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1
    else:
        direction = 0

    return direction, upper_band, lower_band

# Execute Trades
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

# Function for Heartbeat Logging
def heartbeat_logging():
    """
    Logs current status and data every LOG_INTERVAL seconds for monitoring purposes.
    Updates band history to ensure data is current.
    """
    if not volatility or len(volatility) < 3:
        logging.info("Heartbeat: Insufficient data for detailed logging.")
        return

    try:
        # Perform clustering
        centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
        if assigned_centroid is None:
            logging.info("Heartbeat: Clustering not available.")
            return

        # Calculate ATR and SuperTrend
        atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
        direction, upper_band, lower_band = calculate_supertrend_with_clusters(
            pd.Series(high), pd.Series(low), pd.Series(close), assigned_centroid
        )

        # Update band history
        upper_band_history.append(float(upper_band.iloc[-1]))
        lower_band_history.append(float(lower_band.iloc[-1]))
        if len(upper_band_history) > 4:
            upper_band_history.pop(0)
        if len(lower_band_history) > 4:
            lower_band_history.pop(0)

        # Logging detailed information
        logging.info(
            f"\n=== Heartbeat Logging ===\n"
            f"Price: {last_price:.2f}\n"
            f"ATR: {atr:.2f}\n"
            f"Volatility Level: {volatility_level}\n"
            f"Cluster Centroids: {', '.join(f'{x:.2f}' for x in centroids)}\n"
            f"Cluster Sizes: {', '.join(str(size) for size in cluster_sizes)}\n"
            f"Direction: {'Neutral (0)' if direction == 0 else 'Bullish (1)' if direction == 1 else 'Bearish (-1)'}\n"
            f"Upper Bands (Last 4): {', '.join(f'{x:.2f}' for x in upper_band_history)}\n"
            f"Lower Bands (Last 4): {', '.join(f'{x:.2f}' for x in lower_band_history)}\n"
            f"=========================="
        )
    except Exception as e:
        logging.error(f"Error during heartbeat logging: {e}", exc_info=True)

# Signal Processing
def calculate_and_execute(price):
    global last_direction, upper_band_history, lower_band_history

    if not volatility or len(volatility) < 3:
        logging.warning("Volatility list is empty or insufficient. Skipping this cycle.")
        return

    centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
    if assigned_centroid is None:
        logging.warning("Clustering failed. Skipping cycle.")
        return

    atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
    direction, upper_band, lower_band = calculate_supertrend_with_clusters(
        pd.Series(high), pd.Series(low), pd.Series(close), assigned_centroid
    )

    # Update the band history
    upper_band_history.append(float(upper_band.iloc[-1]))
    lower_band_history.append(float(lower_band.iloc[-1]))
    if len(upper_band_history) > 4:
        upper_band_history.pop(0)
    if len(lower_band_history) > 4:
        lower_band_history.pop(0)

    # Check buy/sell conditions
    buy_signal = any(price < band for band in lower_band_history)
    sell_signal = any(price > band for band in upper_band_history)

    # Determine the direction based on signals
    if buy_signal:
        direction = 1
    elif sell_signal:
        direction = -1
    else:
        direction = 0

    # Logging
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

    # Execute trade if direction changes
    if last_direction != direction:
        if direction == 1:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "buy")
        elif direction == -1:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")

    # Update last direction
    last_direction = direction


# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close

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
            logging.info(f"Reconnecting in 30 seconds...")
            time.sleep(30)  # Wait before reconnecting

def process_signals():
    global last_signal_time, last_heartbeat_time

    # Align to the next 30-second mark minus a few seconds (e.g., 3 seconds before)
    now = datetime.datetime.now()
    next_heartbeat = now + datetime.timedelta(seconds=(30 - now.second % 30))
    sync_offset = 3  # Start a few seconds earlier
    next_heartbeat_delay = max(0, (next_heartbeat - now).total_seconds() - sync_offset)
    logging.info(f"Synchronizing to next heartbeat in {next_heartbeat_delay:.2f} seconds.")
    time.sleep(next_heartbeat_delay)

    while True:
        current_time = time.time()

        # Check for signal processing (every 300 seconds, starting slightly earlier)
        if current_time - last_signal_time >= SIGNAL_INTERVAL - sync_offset:
            if last_price is not None:
                try:
                    calculate_and_execute(last_price)
                except Exception as e:
                    logging.error(f"Error during signal processing: {e}", exc_info=True)
            last_signal_time = current_time + sync_offset  # Adjust for sync offset
            last_heartbeat_time = current_time + sync_offset  # Skip heartbeat logging at this time

        # Check for heartbeat logging (every 30 seconds, starting slightly earlier)
        elif current_time - last_heartbeat_time >= 30 - sync_offset:
            heartbeat_logging()  # Just logs the state
            last_heartbeat_time = current_time + sync_offset

        # Sleep for a small interval to avoid excessive CPU usage
        time.sleep(1)


# Main Script Entry Point
if __name__ == "__main__":
    initialize_historical_data()  # Initialize historical data

    # Start Flask app in a separate thread
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    time.sleep(10)

    # Start Signal Processing Loop in a separate thread
    Thread(target=process_signals, daemon=True).start()

    # Start WebSocket for real-time data
    try:
        start_websocket()
    except Exception as e:
        logging.error(f"Critical failure in WebSocket connection: {e}. Restarting...")
        restart_program()
