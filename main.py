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
from binance.client import Client
from sklearn.cluster import KMeans  # For k-means clustering

# Binance API Credentials (read from environment variables)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Initialize Binance Client
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

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
TRAINING_DATA_PERIOD = 100
SYMBOL = "BTC/USD"  # Correct cryptocurrency symbol format
BINANCE_SYMBOL = "BTCUSDT"  # Binance uses a different symbol format
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
def fetch_historical_data(symbol=BINANCE_SYMBOL, interval="1m", limit=TRAINING_DATA_PERIOD):
    """
    Fetch historical market data using Binance.
    Returns a DataFrame with high, low, and close prices.
    """
    try:
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
        prices = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        prices = prices[["high", "low", "close"]].astype(float)
        return prices
    except Exception as e:
        logging.error(f"Error fetching historical data from Binance: {e}")
        return pd.DataFrame()

# Calculate Average True Range (ATR)
def calculate_atr(prices, period=ATR_LEN):
    """Calculate the ATR based on historical prices."""
    high = prices["high"]
    low = prices["low"]
    close = prices["close"].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]  # Return the latest ATR value

# Perform k-means clustering to classify volatility levels
def classify_volatility(prices):
    """Classify volatility into high, medium, and low clusters using k-means."""
    try:
        atr_values = prices["close"].rolling(window=ATR_LEN).std().dropna().values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(atr_values)

        # Assign cluster centroids to volatility levels
        centroids = kmeans.cluster_centers_.flatten()
        centroids.sort()  # Ensure low, medium, and high order
        low, medium, high = centroids

        # Determine current volatility cluster
        current_atr = atr_values[-1][0]
        if current_atr <= low:
            return low, "low"
        elif low < current_atr <= medium:
            return medium, "medium"
        else:
            return high, "high"
    except Exception as e:
        logging.error(f"Error in k-means clustering: {e}")
        return 3.0, "medium"  # Default factor

# Calculate SuperTrend
def calculate_supertrend(latest_price, atr_factor):
    """Calculate the SuperTrend value dynamically based on ATR factor."""
    supertrend_value = latest_price - (latest_price * atr_factor / 100)
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

            # Fetch historical data for ATR calculation
            historical_prices = fetch_historical_data()
            if historical_prices.empty:
                logging.warning("Failed to fetch historical data. Skipping this cycle.")
                time.sleep(60)
                continue

            # Classify volatility and calculate dynamic ATR factor
            atr_factor, volatility_level = classify_volatility(historical_prices)
            logging.info(f"Classified volatility as {volatility_level} with ATR factor: {atr_factor}")

            # Calculate SuperTrend
            supertrend_value, direction = calculate_supertrend(latest_price, atr_factor)
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
                execute_trade(SYMBOL, accumulated_quantity, "sell")
                last_signal = "sell"

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

