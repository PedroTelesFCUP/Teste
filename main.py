import os
import time
import numpy as np
import pandas as pd
import logging
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream
from flask import Flask
from threading import Thread
import asyncio
import sys  # Add sys for logging to stdout

# Alpaca API Credentials (read from environment variables)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

if not API_KEY or not SECRET_KEY:
    logging.error("Missing Alpaca API credentials! Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
    raise ValueError("Missing Alpaca API credentials!")

# Initialize Alpaca REST API and WebSocket Stream
api = REST(API_KEY, SECRET_KEY, BASE_URL)
stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL)

# Parameters for SuperTrend
ATR_LEN = 10
FACTOR = 3
SYMBOL = "BTC/USD"  # Cryptocurrency symbol
QUANTITY = round(0.001, 8)  # Adjust for fractional trading with required precision

# Configure Logging
logging.basicConfig(
    stream=sys.stdout,  # Ensure logs are visible in Render's dashboard
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Trading bot initialized.")

# Flask Web Server for Uptime Monitoring
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

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

def fetch_market_data(symbol, limit=100):
    logging.info(f"Fetching 1-minute market data for {symbol}...")
    try:
        bars = api.get_crypto_bars(symbol.replace("/", ""), "1Min", limit=limit).df
        if bars.empty:
            logging.warning("No market data returned!")
            return pd.DataFrame()
        bars = bars.tz_convert("America/New_York")
        return bars
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def execute_trade(symbol, quantity, side):
    logging.info(f"Executing {side} order for {quantity} of {symbol}...")
    try:
        order = api.submit_order(
            symbol=symbol.replace("/", ""),
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        logging.info(f"{side.capitalize()} order submitted successfully.")
    except Exception as e:
        logging.error(f"Failed to execute {side} order: {e}")

async def on_trade(trade):
    logging.info(f"Trade Data: {trade}")

async def start_stream():
    while True:
        try:
            await stream.subscribe_crypto_trades(on_trade, SYMBOL.replace("/", ""))
            await stream.run()
        except Exception as e:
            logging.error(f"WebSocket disconnected: {e}")
            await asyncio.sleep(5)  # Reconnect after a delay

def trading_bot():
    logging.info("Trading bot started.")
    while True:
        try:
            logging.info("Fetching market data...")
            data = fetch_market_data(SYMBOL, limit=ATR_LEN + 1)

            if data.empty:
                logging.warning("No market data available.")
                time.sleep(60)
                continue

            logging.info("Calculating SuperTrend...")
            data["atr"] = calculate_atr(data["high"], data["low"], data["close"], ATR_LEN)
            data["supertrend"], data["direction"] = supertrend(
                data["high"], data["low"], data["close"], data["atr"], FACTOR
            )

            # Log current SuperTrend values
            latest_price = data["close"].iloc[-1]
            latest_supertrend = data["supertrend"].iloc[-1]
            latest_direction = data["direction"].iloc[-1]
            previous_direction = data["direction"].iloc[-2]

            logging.info(f"Latest Price: {latest_price}")
            logging.info(f"Latest SuperTrend Value: {latest_supertrend}")
            logging.info(f"Current Direction: {latest_direction}, Previous Direction: {previous_direction}")

            # Determine buy/sell signals
            if latest_direction == 1 and previous_direction == -1:
                logging.info(f"Buy signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "buy")

            elif latest_direction == -1 and previous_direction == 1:
                logging.info(f"Sell signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "sell")

            time.sleep(60)

        except Exception as e:
            logging.error(f"Error in trading bot: {e}")
            time.sleep(60)

def run_web_server():
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

# Run Flask Web Server, Trading Bot, and WebSocket
if __name__ == "__main__":
    # Start the web server in a separate thread
    thread1 = Thread(target=run_web_server)
    thread1.daemon = True
    thread1.start()

    # Start the trading bot in a separate thread
    thread2 = Thread(target=trading_bot)
    thread2.daemon = True
    thread2.start()

    # Run WebSocket asynchronously
    asyncio.run(start_stream())
