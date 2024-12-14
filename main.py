import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from binance.client import Client

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

# Globals
last_signal = None
initialized = False

# Fetch Real-Time Price
def fetch_realtime_price():
    try:
        ticker = binance_client.get_symbol_ticker(symbol=BINANCE_SYMBOL)
        return float(ticker["price"])
    except Exception as e:
        logging.error(f"Error fetching real-time price: {e}")
        return None

# Fetch Historical Data
def fetch_historical_data():
    try:
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="1m", limit=ATR_LEN + 1)
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", 
                                             "close_time", "quote_asset_volume", "number_of_trades",
                                             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        return data["high"].astype(float), data["low"].astype(float), data["close"].astype(float)
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None, None, None

# Calculate ATR
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=ATR_LEN).mean().iloc[-1]

# Calculate SuperTrend
def calculate_supertrend(high, low, close, atr, prev_upper_band, prev_lower_band, prev_supertrend):
    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * atr
    lower_band = hl2 - ATR_FACTOR * atr

    # Validate previous bands
    if prev_upper_band is None:
        prev_upper_band = upper_band.iloc[-1] if len(upper_band) > 0 else hl2.iloc[-1]
    if prev_lower_band is None:
        prev_lower_band = lower_band.iloc[-1] if len(lower_band) > 0 else hl2.iloc[-1]
    if prev_supertrend is None:
        prev_supertrend = lower_band.iloc[-1] if close.iloc[-1] < hl2.iloc[-1] else upper_band.iloc[-1]

    # Adjust bands based on previous values
    lower_band = np.where(
        (lower_band > prev_lower_band) | (close.shift(1) < prev_lower_band),
        lower_band, prev_lower_band
    )
    upper_band = np.where(
        (upper_band < prev_upper_band) | (close.shift(1) > prev_upper_band),
        upper_band, prev_upper_band
    )

    # Calculate direction
    if prev_supertrend == prev_upper_band:
        direction = -1 if close.iloc[-1] > upper_band[-1] else 1
    else:
        direction = 1 if close.iloc[-1] < lower_band[-1] else -1

    supertrend = lower_band[-1] if direction == 1 else upper_band[-1]
    return supertrend, direction, upper_band[-1], lower_band[-1]

# Execute a Trade
def execute_trade(symbol, quantity, side):
    logging.info(f"Executing {side} order for {quantity} of {symbol}...")

# Main Trading Bot
def trading_bot():
    global last_signal, initialized

    prev_upper_band = None
    prev_lower_band = None
    prev_supertrend = None

    while True:
        try:
            price = fetch_realtime_price()
            if price is None:
                time.sleep(60)
                continue

            high, low, close = fetch_historical_data()
            if high is None or low is None or close is None:
                time.sleep(60)
                continue

            atr = calculate_atr(high, low, close)
            if atr is None or np.isnan(atr):
                time.sleep(60)
                continue

            supertrend_value, direction, upper_band, lower_band = calculate_supertrend(
                high, low, close, atr, prev_upper_band, prev_lower_band, prev_supertrend
            )

            logging.info(f"SuperTrend: {supertrend_value}, Direction: {direction}")
            logging.info(f"Upper Band: {upper_band}, Lower Band: {lower_band}")

            if not initialized:
                initialized = True
                last_signal = "buy" if direction == 1 else "sell"
                logging.info(f"Initialization complete. First signal: {last_signal}")
                continue

            if last_signal == "sell" and direction == 1:
                execute_trade(SYMBOL, QUANTITY, "buy")
                last_signal = "buy"
            elif last_signal == "buy" and direction == -1:
                execute_trade(SYMBOL, QUANTITY, "sell")
                last_signal = "sell"

            prev_upper_band, prev_lower_band, prev_supertrend = upper_band, lower_band, supertrend_value
            time.sleep(60)

        except Exception as e:
            logging.error(f"Error in trading bot: {e}", exc_info=True)
            time.sleep(60)

# Run Flask and Trading Bot
if __name__ == "__main__":
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    trading_bot()



