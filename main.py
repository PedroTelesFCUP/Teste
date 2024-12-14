import os
import time
import numpy as np
import pandas as pd
import logging
from flask import Flask
from threading import Thread
from binance.client import Client
from sklearn.cluster import KMeans

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

# Perform K-Means Clustering on volatility
def cluster_volatility(volatility, n_clusters=3):
    if len(volatility) < n_clusters:
        logging.error("Not enough data points for clustering. Skipping.")
        return None, None, None
    try:
        volatility = np.array(volatility).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(volatility)
        centroids = kmeans.cluster_centers_.flatten()
        assigned_cluster = kmeans.predict([[volatility[-1][0]]])[0]
        assigned_centroid = centroids[assigned_cluster]
        return centroids, assigned_cluster, assigned_centroid
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        return None, None, None

# Calculate ATR
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=ATR_LEN).mean().iloc[-1]

# Calculate SuperTrend using cluster-based ATR
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid):
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logging.error("Insufficient market data for SuperTrend calculation. Skipping.")
        return None, None, None, None
    if assigned_centroid is None:
        logging.error("Invalid centroid. Skipping SuperTrend calculation.")
        return None, None, None, None

    hl2 = (high + low) / 2
    upper_band = hl2 + ATR_FACTOR * assigned_centroid
    lower_band = hl2 - ATR_FACTOR * assigned_centroid

    if close.iloc[-1] > upper_band.iloc[-1]:
        direction = -1  # Bearish
    elif close.iloc[-1] < lower_band.iloc[-1]:
        direction = 1  # Bullish
    else:
        direction = 0  # Neutral

    supertrend = lower_band.iloc[-1] if direction == 1 else upper_band.iloc[-1]
    return supertrend, direction, upper_band, lower_band

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

# Execute a Trade
def execute_trade(symbol, quantity, side):
    logging.info(f"Executing {side} order for {quantity} of {symbol}...")

# Main Trading Bot
def trading_bot():
    global last_direction
    volatility = []  # Stores ATR values
    last_direction = 0

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
                logging.warning("ATR calculation failed. Skipping cycle.")
                time.sleep(60)
                continue

            # Add ATR to volatility list
            volatility.append(atr)
            if len(volatility) > 100:  # Limit history to the last 100 values
                volatility.pop(0)

            # Ensure sufficient data points for clustering
            if len(volatility) < 3:  # Require at least 3 points (or n_clusters)
                logging.warning("Insufficient data for clustering. Waiting for more data.")
                time.sleep(60)
                continue

            # Cluster Volatility
            centroids, assigned_cluster, assigned_centroid = cluster_volatility(volatility)
            if assigned_centroid is None:
                logging.warning("Clustering failed. Skipping cycle.")
                time.sleep(60)
                continue

            # Calculate SuperTrend
            supertrend, direction, upper_band, lower_band = calculate_supertrend_with_clusters(
                high, low, close, assigned_centroid
            )

            if supertrend is None:
                logging.warning("SuperTrend calculation failed. Skipping cycle.")
                time.sleep(60)
                continue

            logging.info(f"BTC Price: {close.iloc[-1]}, SuperTrend: {supertrend}, Upper Band: {upper_band.iloc[-1]}, Lower Band: {lower_band.iloc[-1]}, Direction: {direction}")

            # Execute trade if signal changes
            if last_direction == 0 and direction in [1, -1]:
                trade_type = "buy" if direction == 1 else "sell"
                execute_trade(SYMBOL, QUANTITY, trade_type)
            elif last_direction == 1 and direction == -1:
                execute_trade(SYMBOL, QUANTITY, "sell")
            elif last_direction == -1 and direction == 1:
                execute_trade(SYMBOL, QUANTITY, "buy")

            # Update last direction
            last_direction = direction
            time.sleep(60)

        except Exception as e:
            logging.error(f"Error in trading bot: {e}", exc_info=True)
            time.sleep(60)


# Run Flask and Trading Bot
if __name__ == "__main__":
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    trading_bot()



