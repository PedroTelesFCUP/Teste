import os
import time
import numpy as np
import pandas as pd
import logging
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
from flask import Flask
from threading import Thread
import asyncio
import sys

# Alpaca API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Trading API endpoint
DATA_URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"  # WebSocket endpoint for crypto

if not API_KEY or not SECRET_KEY:
    logging.error("Missing Alpaca API credentials! Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
    raise ValueError("Missing Alpaca API credentials!")

# Initialize Alpaca REST API
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# Initialize WebSocket Stream
try:
    stream = Stream(API_KEY, SECRET_KEY, base_url=URL(BASE_URL), data_url=URL(DATA_URL))
    logging.info("WebSocket stream initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize WebSocket stream: {e}")
    stream = None

# Parameters for SuperTrend
ATR_LEN = 10
FACTOR = 3
SYMBOL = "BTC/USD"
QUANTITY = round(0.001, 8)

# Configure Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Trading bot initialized.")

# Flask Web Server for Uptime Monitoring
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

# Global variable to track the last processed price
last_price = None

# Functions for Trading Logic
def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def supertrend(high, low, close, atr, factor):
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr

    supertrend = pd.Series(index=close.index, dtype="float64")
    direction = pd.Series(index=close.index, dtype="int")

    for i in range(len(close)):
        if i == 0:
            continue
        prev_upper_band = upper_band.iloc[i - 1]
        prev_lower_band = lower_band.iloc[i - 1]

        lower_band.iloc[i] = max(lower_band.iloc[i], prev_lower_band if close.iloc[i - 1] >= prev_lower_band else lower_band.iloc[i])
        upper_band.iloc[i] = min(upper_band.iloc[i], prev_upper_band if close.iloc[i - 1] <= prev_upper_band else upper_band.iloc[i])

        if supertrend.iloc[i - 1] == prev_upper_band:
            direction.iloc[i] = -1 if close.iloc[i] < upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > lower_band.iloc[i] else -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction

async def on_trade(trade):
    logging.info(f"Trade Data: {trade}")

async def start_stream():
    if not stream:
        logging.error("WebSocket stream is not initialized. Exiting WebSocket task.")
        return

    while True:
        try:
            await stream.subscribe_crypto_trades(on_trade, SYMBOL)
            await stream.run()
        except Exception as e:
            logging.error(f"WebSocket disconnected: {e}")
            await asyncio.sleep(5)  # Reconnect after a delay

def run_web_server():
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

# Run Flask Web Server, Trading Bot, and WebSocket
if __name__ == "__main__":
    # Start the web server in a separate thread
    thread1 = Thread(target=run_web_server)
    thread1.daemon = True
    thread1.start()

    # Run WebSocket asynchronously
    asyncio.run(start_stream())



