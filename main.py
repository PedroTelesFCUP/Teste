import os
import time
import numpy as np
import pandas as pd
import logging
import requests
from alpaca_trade_api.rest import REST
from flask import Flask
from threading import Thread
import sys  # Add sys for logging to stdout

# Alpaca API Credentials (read from environment variables)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

if not API_KEY or not SECRET_KEY:
    logging.error("Missing Alpaca API credentials! Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
    raise ValueError("Missing Alpaca API credentials!")

# Initialize Alpaca REST API
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# Parameters for SuperTrend
ATR_LEN = 10
FACTOR = 3
SYMBOL = "BTC/USD"  # Correct cryptocurrency symbol format
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

# Global variable to track the last processed price
last_price = None

# Fetch real-time BTC/USD price using CoinGecko
def fetch_realtime_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        price = data["bitcoin"]["usd"]
        logging.info(f"Real-time BTC/USD price: {price}")
        return price
    except Exception as e:
        logging.error(f"Error fetching real-time price: {e}")
        return None

# Execute a trade on Alpaca
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

# Main trading bot logic
def trading_bot():
    global last_price
    logging.info("Trading bot started.")
    while True:
        try:
            logging.info("Fetching real-time BTC/USD price...")
            latest_price = fetch_realtime_price()

            if latest_price is None:
                logging.warning("Failed to fetch real-time price. Retrying...")
                time.sleep(60)
                continue

            # Skip processing if price hasn't changed
            if last_price == latest_price:
                logging.info("No price change. Skipping this cycle.")
                time.sleep(60)
                continue

            # Update last processed price
            last_price = latest_price

            logging.info(f"Real-time price: {latest_price}")

            # Example trading logic (update with your conditions)
            # For instance, you can compare with a moving average or predefined thresholds
            if latest_price > 50000:  # Replace with your buy condition
                logging.info(f"Buy signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "buy")
            elif latest_price < 40000:  # Replace with your sell condition
                logging.info(f"Sell signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "sell")

            time.sleep(60)  # Poll every minute

        except Exception as e:
            logging.error(f"Error in trading bot: {e}")
            time.sleep(60)

# Run Flask Web Server
def run_web_server():
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

# Start the bot
if __name__ == "__main__":
    # Start the web server in a separate thread
    thread1 = Thread(target=run_web_server)
    thread1.daemon = True
    thread1.start()

    # Start the trading bot in the main thread
    trading_bot()


