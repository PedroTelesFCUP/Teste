import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask, send_file
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
ATR_FACTOR_MIN = 2.7
ATR_FACTOR_MAX = 3.0
FIXED_BUY_VALUE = 100  # $100 per buy
SIGNAL_INTERVAL = 30  # Process signals every 30 seconds
SYMBOL = "BTC/USD"
ALPACA_SYMBOL = SYMBOL.replace("/", "")  # Alpaca doesn't use "/"
BINANCE_SYMBOL = "BTCUSDT"

# Logging Configuration: File and Console
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
volatility = []  # Initialize volatility list
last_price = None
last_direction = 0
last_signal_time = 0  # Track the last processed signal time
high, low, close = [], [], []  # Historical data buffers for real-time updates
upper_band_history = []  # Store recent upper band values
lower_band_history = []  # Store recent lower band values
max_history_length = 4  # Number of time periods to track

# Flask Server
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

@app.route('/logs')
def download_logs():
    try:
        return send_file("bot_logs.log", as_attachment=True)
    except FileNotFoundError:
        return "Log file not found.", 404

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

# Initialize Historical Data
def initialize_historical_data():
    global high, low, close, volatility
    try:
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="1m", limit=100 + ATR_LEN)
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

# Calculate SuperTrend with Dynamic ATR Factor
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logging.error("Insufficient market data for SuperTrend calculation. Skipping.")
        return None, None, None, None

    if assigned_centroid is None:
        logging.error("Invalid centroid from clustering. Skipping SuperTrend calculation.")
        return None, None, None, None

    atr_factor = max(ATR_FACTOR_MIN, min(ATR_FACTOR_MAX, assigned_centroid / np.mean(volatility)))
    logging.info(f"Dynamic ATR Factor: {atr_factor:.2f}")

    hl2 = (high + low) / 2
    upper_band = hl2 + atr_factor * assigned_centroid
    lower_band = hl2 - atr_factor * assigned_centroid

    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1  # Bearish
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1  # Bullish
    else:
        direction = 0  # Neutral

    if direction == 1:
        supertrend = lower_band.iloc[-1]
    elif direction == -1:
        supertrend = upper_band.iloc[-1]
    else:
        supertrend = None

    return supertrend, direction, upper_band, lower_band

# Execute Trades
def execute_trade(symbol, fixed_value, side, price=None):
    try:
        if side == "buy":
            if not price:
                logging.error("Price is required for buy trades.")
                return
            quantity = round(fixed_value / price, 8)
            logging.info(f"Calculated buy quantity: {quantity} for ${fixed_value} at price ${price:.2f}")
        elif side == "sell":
            position = alpaca_api.get_position(symbol)
            quantity = float(position.qty) if position else 0.0
            if quantity <= 0:
                logging.warning("No holdings available to sell. Skipping trade.")
                return
            logging.info(f"Current holdings for sell: {quantity}")

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

# Process Signals
def calculate_and_execute(price):
    global last_direction, upper_band_history, lower_band_history

    if not volatility or len(volatility) < 3:
        logging.warning("Volatility list is empty or insufficient. Skipping this cycle.")
        return

    # Perform clustering and calculate SuperTrend
    centroids, assigned_cluster, assigned_centroid, cluster_sizes, volatility_level = cluster_volatility(volatility)
    if assigned_centroid is None:
        logging.warning("Clustering failed. Skipping cycle.")
        return

    atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
    supertrend, direction, upper_band, lower_band = calculate_supertrend_with_clusters(
        pd.Series(high), pd.Series(low), pd.Series(close), assigned_centroid
    )

    # Update historical bands
    upper_band_history.append(float(upper_band.iloc[-1]))
    lower_band_history.append(float(lower_band.iloc[-1]))

    # Trim history to the max length
    if len(upper_band_history) > max_history_length:
        upper_band_history.pop(0)
    if len(lower_band_history) > max_history_length:
        lower_band_history.pop(0)

    # Log current state with formatted output
    direction_str = "Bullish (1)" if direction == 1 else "Bearish (-1)" if direction == -1 else "Neutral (0)"
    logging.info(
        f"\nCurrent Price: {price:.2f}\n"
        f"ATR: {atr:.2f}\n"
        f"Volatility Level: {volatility_level}\n"
        f"Cluster Centroids: {', '.join(f'{x:.2f}' for x in centroids)}\n"
        f"Cluster Sizes: {', '.join(str(size) for size in cluster_sizes)}\n"
        f"Upper Band History: {[f'{x:.2f}' for x in upper_band_history]}\n"
        f"Lower Band History: {[f'{x:.2f}' for x in lower_band_history]}\n"
        f"Current Direction: {direction_str}\n"
    )

    # Check current price against historical bands
    for i in range(len(upper_band_history)):
        if price < lower_band_history[i] and last_direction != 1:
            logging.info(f"Buy signal: Current price {price:.2f} is below historical lower band {lower_band_history[i]:.2f}.")
            execute_trade(ALPACA_SYMBOL, FIXED_BUY_VALUE, "buy", price=price)
            last_direction = 1  # Update to "bullish"
            return  # Exit to prevent multiple trades in one cycle

        elif price > upper_band_history[i] and last_direction != -1:
            logging.info(f"Sell signal: Current price {price:.2f} is above historical upper band {upper_band_history[i]:.2f}.")
            execute_trade(ALPACA_SYMBOL, FIXED_BUY_VALUE, "sell")
            last_direction = -1  # Update to "bearish"
            return  # Exit to prevent multiple trades in one cycle

# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close

    try:
        if 'k' not in msg:
            logging.warning(f"Unexpected message format: {msg}")
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

    except Exception as e:
        logging.error(f"Error processing WebSocket message: {e}")

# WebSocket Manager
def start_websocket():
    while True:
        try:
            logging.info("Starting WebSocket connection...")
            twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            twm.start()
            twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="1m")
            twm.join()
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            logging.info("Reconnecting in 5 seconds...")
            time.sleep(5)

# Signal Processing Loop
def process_signals():
    global last_signal_time
    while True:
        current_time = time.time()
        if current_time - last_signal_time >= SIGNAL_INTERVAL:
            if last_price is not None:
                calculate_and_execute(last_price)
            last_signal_time = current_time
        time.sleep(1)

if __name__ == "__main__":
    initialize_historical_data()
    Thread(target=process_signals, daemon=True).start()
    Thread(target=lambda: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080))), daemon=True).start()
    start_websocket()







