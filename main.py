import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from binance.client import Client

# Binance API Credentials (read from environment variables)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Initialize Binance Client
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters for SuperTrend
ATR_LEN = 10
ATR_FACTOR = 3.0  # Default value, dynamically adjusted
SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)  # Adjust for fractional trading

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Trading bot initialized.")

# Flask Web Server for Uptime Monitoring
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

# Global variables
last_price = None
last_signal = None
initialized = False
atr_values = []
volatility_window = 100  # Lookback period for volatility classification

# Fetch real-time BTC/USD price using Binance
def fetch_realtime_price():
    try:
        ticker = binance_client.get_symbol_ticker(symbol=BINANCE_SYMBOL)
        price = float(ticker["price"])
        logging.info(f"Real-time BTC/USD price: {price}")
        return price
    except Exception as e:
        logging.error(f"Error fetching real-time price from Binance: {e}")
        return None

# Fetch historical market data using Binance
def fetch_historical_data(symbol=BINANCE_SYMBOL, interval="1m", limit=ATR_LEN + 1):
    try:
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", 
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ])
        return data["high"].astype(float), data["low"].astype(float), data["close"].astype(float)
    except Exception as e:
        logging.error(f"Error fetching historical data from Binance: {e}")
        return None, None, None

# Calculate Average True Range (ATR)
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=ATR_LEN).mean()
    return atr.iloc[-1]

# Calculate SuperTrend
def calculate_supertrend(high, low, close, atr, prev_upper_band, prev_lower_band, prev_supertrend):
    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * atr
    lower_band = hl2 - ATR_FACTOR * atr

    lower_band = np.where(
        (lower_band > prev_lower_band) | (close.shift(1) < prev_lower_band),
        lower_band, prev_lower_band
    )
    upper_band = np.where(
        (upper_band < prev_upper_band) | (close.shift(1) > prev_upper_band),
        upper_band, prev_upper_band
    )

    if prev_supertrend == prev_upper_band:
        direction = -1 if close.iloc[-1] > upper_band[-1] else 1
    else:
        direction = 1 if close.iloc[-1] < lower_band[-1] else -1

    supertrend = lower_band[-1] if direction == 1 else upper_band[-1]
    return supertrend, direction

# Execute a trade
def execute_trade(symbol, quantity, side):
    logging.info(f"Executing {side} order for {quantity} of {symbol}...")

# Main trading bot logic
def trading_bot():
    global last_price, last_signal, initialized, atr_values

    prev_upper_band = None
    prev_lower_band = None
    prev_supertrend = None

    while True:
        try:
            logging.info("Fetching real-time BTC/USD price...")
            latest_price = fetch_realtime_price()
            if latest_price is None:
                logging.warning("Failed to fetch price. Retrying...")
                time.sleep(60)
                continue

            if not initialized:
                initialized = True
                logging.info("Initialization complete.")
                time.sleep(60)
                continue

            high, low, close = fetch_historical_data()
            if high is None or low is None or close is None:
                logging.error("Historical data is missing. Skipping this cycle.")
                time.sleep(60)
                continue

            if high.isnull().any() or low.isnull().any() or close.isnull().any():
                logging.error("Historical data contains NaN values. Skipping this cycle.")
                time.sleep(60)
                continue

            atr = calculate_atr(high, low, close)
            if atr is None or np.isnan(atr):
                logging.error("ATR value is None or NaN. Skipping this cycle.")
                time.sleep(60)
                continue

            atr_values.append(atr)
            if len(atr_values) > volatility_window:
                atr_values.pop(0)

            supertrend_value, direction = calculate_supertrend(
                high, low, close, atr, prev_upper_band, prev_lower_band, prev_supertrend
            )

            logging.info(f"Latest Price: {latest_price}")
            logging.info(f"SuperTrend Value: {supertrend_value}")
            logging.info(f"Current Direction: {direction}")

            if last_signal == "sell" and direction == 1:
                execute_trade(SYMBOL, QUANTITY, "buy")
                last_signal = "buy"
            elif last_signal == "buy" and direction == -1:
                execute_trade(SYMBOL, QUANTITY, "sell")
                last_signal = "sell"

            prev_upper_band = supertrend_value if direction == -1 else prev_upper_band
            prev_lower_band = supertrend_value if direction == 1 else prev_lower_band
            prev_supertrend = supertrend_value

            time.sleep(60)

        except Exception as e:
            logging.error(f"Error in trading bot: {e}")
            time.sleep(60)

# Run Flask Web Server and Trading Bot
if __name__ == "__main__":
    thread = Thread(target=lambda: app.run(host="0.0.0.0", port=8080))
    thread.daemon = True
    thread.start()

    trading_bot()


