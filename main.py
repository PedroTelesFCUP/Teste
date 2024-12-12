import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from alpaca_trade_api.rest import REST

# Alpaca API Credentials (stored securely in Render's Environment)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

# Initialize Alpaca API
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# Parameters for SuperTrend
ATR_LEN = 10
FACTOR = 3
SYMBOL = "TSLA"  # Tesla stock
QUANTITY = 1000  # Number of shares to trade

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
)
logging.info("Trading bot initialized.")


def calculate_atr(high, low, close, period):
    """Calculate the Average True Range (ATR)."""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def supertrend(high, low, close, atr, factor):
    """Calculate the SuperTrend."""
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
    """Fetch historical 5-minute market data from Alpaca."""
    logging.info(f"Fetching 5-minute market data for {symbol}...")
    try:
        bars = api.get_bars(symbol, "5Min", limit=limit).df
        bars = bars.tz_convert("America/New_York")  # Convert to your timezone
        logging.info(f"Fetched {len(bars)} data points.")
        print("Market Data:\n", bars.tail())
        return bars
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        print(f"Error fetching market data: {e}")
        return pd.DataFrame()


def execute_trade(symbol, quantity, side):
    """Place a buy or sell order."""
    logging.info(f"Executing {side} order for {quantity} shares of {symbol}...")
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        logging.info(f"{side.capitalize()} order submitted successfully.")
    except Exception as e:
        logging.error(f"Failed to execute {side} order: {e}")
        print(f"Failed to execute {side} order: {e}")


def log_account_activity():
    """Fetch and log recent account activity."""
    try:
        logging.info("Fetching account activity...")
        activities = api.get_activities()
        for activity in activities[:5]:  # Limit to the last 5 activities
            logging.info(f"Activity: {activity}")
    except Exception as e:
        logging.error(f"Error fetching account activity: {e}")


def trading_bot():
    """Main trading bot logic."""
    logging.info("Trading bot started.")
    print("Trading bot started.")
    try:
        account = api.get_account()
        logging.info(f"Alpaca Account Status: {account.status}")
    except Exception as e:
        logging.error(f"Error connecting to Alpaca API: {e}")
        return

    while True:
        logging.info("Heartbeat: Starting a new cycle.")
        try:
            log_account_activity()  # Log account activities at the start of each cycle

            logging.info("Fetching market data...")
            data = fetch_market_data(SYMBOL, limit=ATR_LEN + 1)
            if data.empty:
                logging.warning("No market data available. Retrying in 5 minutes...")
                time.sleep(300)
                continue

            logging.info("Calculating SuperTrend...")
            data["atr"] = calculate_atr(data["high"], data["low"], data["close"], ATR_LEN)
            data["supertrend"], data["direction"] = supertrend(
                data["high"], data["low"], data["close"], data["atr"], FACTOR
            )

            # Log details
            latest_price = data["close"].iloc[-1]
            latest_supertrend = data["supertrend"].iloc[-1]
            atr_value = data["atr"].iloc[-1]
            latest_direction = data["direction"].iloc[-1]
            previous_direction = data["direction"].iloc[-2]

            logging.info(f"Latest Price: {latest_price}")
            logging.info(f"Latest SuperTrend Value: {latest_supertrend}")
            logging.info(f"ATR Value: {atr_value}")
            logging.info(f"Current Direction: {latest_direction}")
            logging.info(f"Previous Direction: {previous_direction}")

            # Trade signals
            if latest_direction == 1 and previous_direction == -1:
                logging.info(f"Buy signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "buy")
            elif latest_direction == -1 and previous_direction == 1:
                logging.info(f"Sell signal detected at price {latest_price}.")
                execute_trade(SYMBOL, QUANTITY, "sell")
            else:
                logging.info("No trade signal detected.")

            logging.info("Sleeping for 5 minutes...\n")
            time.sleep(300)

        except Exception as e:
            logging.error(f"Error in trading bot: {e}")
            time.sleep(300)


# Flask Web Server for Uptime Monitoring
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200


def run_web_server():
    """Run the Flask web server."""
    PORT = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT not set
    app.run(host="0.0.0.0", port=PORT)


# Run Flask Web Server and Trading Bot
if __name__ == "__main__":
    # Start the web server in a separate thread
    thread = Thread(target=run_web_server)
    thread.daemon = True
    thread.start()

    # Start the trading bot
    trading_bot()

