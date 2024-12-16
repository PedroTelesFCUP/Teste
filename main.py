import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask, send_file
from threading import Thread
from binance import ThreadedWebsocketManager
from binance.client import Client
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
ATR_FACTOR = 3.0  # Fixed factor
FIXED_BUY_VALUE = 100  # $100 per buy
SIGNAL_INTERVAL = 30  # Process signals every 30 seconds
ALPACA_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"

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
high, low, close = [], [], []
upper_band_history = []
lower_band_history = []
max_history_length = 4

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

# Calculate Volatility Levels
def calculate_volatility_levels(volatility):
    """Calculate percentile-based volatility levels and sizes."""
    high_volatility = np.percentile(volatility, 75)
    medium_volatility = np.percentile(volatility, 50)
    low_volatility = np.percentile(volatility, 25)

    # Calculate sizes
    high_size = sum(1 for v in volatility if v >= high_volatility)
    medium_size = sum(1 for v in volatility if medium_volatility <= v < high_volatility)
    low_size = sum(1 for v in volatility if v < medium_volatility)

    return high_volatility, medium_volatility, low_volatility, high_size, medium_size, low_size

# Calculate ATR
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=ATR_LEN).mean().iloc[-1]

# Calculate SuperTrend
def calculate_supertrend(high, low, close, atr):
    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * atr
    lower_band = hl2 - ATR_FACTOR * atr

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

    # Calculate volatility levels and sizes
    high_volatility, medium_volatility, low_volatility, high_size, medium_size, low_size = calculate_volatility_levels(volatility)

    # Assign clusters based on centroids
    centroids = [high_volatility, medium_volatility, low_volatility]
    cluster_sizes = [high_size, medium_size, low_size]

    atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
    direction, upper_band, lower_band = calculate_supertrend(
        pd.Series(high), pd.Series(low), pd.Series(close), atr
    )

    # Update historical bands
    upper_band_history.append(float(upper_band.iloc[-1]))
    lower_band_history.append(float(lower_band.iloc[-1]))

    if len(upper_band_history) > max_history_length:
        upper_band_history.pop(0)
    if len(lower_band_history) > max_history_length:
        lower_band_history.pop(0)

    # Determine current cluster
    current_volatility = volatility[-1]
    if current_volatility >= high_volatility:
        current_cluster = 1  # High Volatility
    elif medium_volatility <= current_volatility < high_volatility:
        current_cluster = 2  # Medium Volatility
    else:
        current_cluster = 3  # Low Volatility

    # Log all details
    direction_str = "Bullish (1)" if direction == 1 else "Bearish (-1)" if direction == -1 else "Neutral (0)"
    logging.info(
        f"\nCurrent Price: {price:.2f}\n"
        f"ATR: {atr:.2f}\n"
        f"High Volatility: {high_volatility:.2f} (Size: {high_size})\n"
        f"Medium Volatility: {medium_volatility:.2f} (Size: {medium_size})\n"
        f"Low Volatility: {low_volatility:.2f} (Size: {low_size})\n"
        f"Current Cluster: {current_cluster} (Volatility: {current_volatility:.2f})\n"
        f"Cluster Centroids: {', '.join(f'{x:.2f}' for x in centroids)}\n"
        f"Cluster Sizes: {', '.join(str(size) for size in cluster_sizes)}\n"
        f"Upper Band History (Last 4): {[f'{x:.2f}' for x in upper_band_history]}\n"
        f"Lower Band History (Last 4): {[f'{x:.2f}' for x in lower_band_history]}\n"
        f"Current Direction: {direction_str}\n"
    )

    # Check current price against historical bands
    for i in range(len(upper_band_history)):
        if price < lower_band_history[i] and last_direction != 1:
            logging.info(f"Buy signal detected: Price {price:.2f} below historical lower band {lower_band_history[i]:.2f}.")
            execute_trade(ALPACA_SYMBOL, FIXED_BUY_VALUE, "buy", price=price)
            last_direction = 1
            return
        elif price > upper_band_history[i] and last_direction != -1:
            logging.info(f"Sell signal detected: Price {price:.2f} above historical upper band {upper_band_history[i]:.2f}.")
            execute_trade(ALPACA_SYMBOL, FIXED_BUY_VALUE, "sell")
            last_direction = -1
            return


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
            twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="5m")
            twm.join()
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            logging.info("Reconnecting in 30 seconds...")
            time.sleep(30)

# Signal Processing Loop
def process_signals():
    global last_signal_time
    while True:
        current_time = time.time()
        if current_time - last_signal_time >= SIGNAL_INTERVAL:
            if last_price is not None:
                calculate_and_execute(last_price



