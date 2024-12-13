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

# Global variables to track the last processed price, signal, and total quantity
last_price = None
last_signal = None
initialized = False
accumulated_quantity = 0.0

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

# Calculate SuperTrend
def calculate_supertrend(latest_price):
    """Mock SuperTrend calculation for demonstration purposes."""
    supertrend_value = latest_price * 0.995  # Example calculation
    direction = 1 if latest_price > supertrend_value else -1
    return supertrend_value, direction

# Execute a trade on Alpaca
def execute_trade(symbol, quantity, side):
    global accumulated_quantity
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

        # Update accumulated quantity
        if side == "buy":
            accumulated_quantity += quantity
        elif side == "sell":
            accumulated_quantity = 0.0  # Reset after selling all
    except Exception as e:
        logging.error(f"Failed to execute {side} order: {e}")

# Main trading bot logic
def trading_bot():
    global last_price, last_signal, initialized, accumulated_quantity
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

            # Skip the first iteration to avoid immediate trading
            if not initialized:
                initialized = True
                logging.info("Initialization complete. Waiting for the first valid signal change.")
                time.sleep(60)
                continue

            # Calculate SuperTrend
            supertrend_value, direction = calculate_supertrend(latest_price)
            logging.info(f"Latest Price: {latest_price}")
            logging.info(f"SuperTrend Value: {supertrend_value}")
            logging.info(f"Current Direction: {direction}")

            # Only execute trades on valid direction changes
            if last_signal == "sell" and direction == 1:  # Transition from sell to buy
                logging.info(f"Buy signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "buy")
                last_signal = "buy"
            elif last_signal == "buy" and direction == -1:  # Transition from buy to sell
                logging.info(f"Sell signal detected at price {latest_price}.")
                execute_trade(SYMBOL, accumulated_quantity, "sell")  # Sell all accumulated quantity
                last_signal = "sell"

            # Initialize the first signal but do not trade
            if last_signal is None:
                last_signal = "buy" if direction == 1 else "sell"
                logging.info(f"Initial signal set to {last_signal}. Waiting for the first valid signal change.")

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


