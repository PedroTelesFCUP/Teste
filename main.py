import os
import sys
import logging
import threading
import asyncio
import time
import numpy as np
from collections import deque
from flask import Flask, request, Response, send_file, jsonify
from binance import AsyncClient, BinanceSocketManager
from binance.enums import SIDE_BUY, SIDE_SELL

# ============== CONFIGURATION ==============
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.log"),  # Save logs to file
        logging.StreamHandler(sys.stdout)    # Output logs to console
    ]
)

# Environment variables / credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY")
TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "YOUR_TESTNET_API_KEY")
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "YOUR_TESTNET_SECRET_KEY")

# Symbol and parameters
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"  # 1-minute candles
QTY = 0.001  # Trade size
ATR_LENGTH = 14
ATR_FACTOR = 2.0
RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Flask server
app = Flask(__name__)
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "password")

# Global Variables for Indicators
high_values = deque(maxlen=100)
low_values = deque(maxlen=100)
close_values = deque(maxlen=100)
position = None
latest_values = {"ATR": None, "RSI": None, "MACD Line": None, "Signal Line": None, "Trend Direction": None, "Latest Close": None}

# ============== AUTHENTICATION ==============
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        "Could not verify your access level for that URL.\n"
        "You have to login with proper credentials", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

@app.before_request
def require_auth():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

# ============== FLASK ROUTES ==============
@app.route("/")
def home():
    return "Bot is running!", 200

@app.route('/logs')
def download_logs():
    try:
        return send_file("bot_logs.log", as_attachment=True)
    except FileNotFoundError:
        return "Log file not found.", 404

@app.route("/orders", methods=["GET"])
def get_orders():
    return jsonify({"message": "Order fetching is not implemented in this async example."})

# ============== INDICATOR CALCULATIONS ==============
def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    true_range = np.maximum(tr1[1:], np.maximum(tr2, tr3))
    true_range = np.concatenate(([tr1[0]], true_range))  # Include the first TR

    atr = np.convolve(true_range, np.ones(period) / period, mode='valid')
    logging.debug(f"True Range: {true_range}, ATR: {atr}")
    return atr

def calculate_rsi(close, period):
    delta = np.diff(close)
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    logging.debug(f"RSI: {rsi}")
    return rsi

def calculate_macd(close, fast, slow, signal):
    if len(close) < slow:
        logging.warning("Not enough data points for MACD calculation.")
        return np.array([]), np.array([])

    ema_fast = np.convolve(close, np.ones(fast) / fast, mode='valid')
    ema_slow = np.convolve(close, np.ones(slow) / slow, mode='valid')

    # Ensure ema_fast and ema_slow align
    macd_line = ema_fast[-len(ema_slow):] - ema_slow
    if len(macd_line) < signal:
        logging.warning("Not enough MACD data points for signal line calculation.")
        return macd_line, np.array([])

    signal_line = np.convolve(macd_line, np.ones(signal) / signal, mode='valid')
    return macd_line, signal_line

def calculate_trend_direction(close, lower_band, upper_band):
    trend_direction = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > lower_band[i]:
            trend_direction[i] = 1  # Bullish
        elif close[i] < upper_band[i]:
            trend_direction[i] = -1  # Bearish
        else:
            trend_direction[i] = trend_direction[i - 1]  # Continue previous trend

    logging.debug(f"Trend Direction: {trend_direction}")
    return trend_direction

def calculate_bands(high, low, atr, factor):
    upper_band = (high + factor * atr)[-len(high):]
    lower_band = (low - factor * atr)[-len(low):]
    return upper_band, lower_band

# ============== CANDLE PROCESSING ==============
def process_candle(high, low, close):
    global high_values, low_values, close_values, position, latest_values

    high_values.append(high)
    low_values.append(low)
    close_values.append(close)

    if len(close_values) < ATR_LENGTH or len(close_values) < max(MACD_SLOW, RSI_LENGTH):
        logging.warning("Not enough data points for calculations.")
        return

    atr = calculate_atr(np.array(high_values), np.array(low_values), np.array(close_values), ATR_LENGTH)
    rsi = calculate_rsi(np.array(close_values), RSI_LENGTH)
    macd_line, signal_line = calculate_macd(np.array(close_values), MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    if len(macd_line) == 0 or len(signal_line) == 0:
        logging.warning("MACD or Signal Line calculation incomplete. Skipping this candle.")
        return

    upper_band, lower_band = calculate_bands(np.array(high_values), np.array(low_values), atr[-1], ATR_FACTOR)
    trend_direction = calculate_trend_direction(np.array(close_values), lower_band, upper_band)

    latest_values = {
        "ATR": atr[-1] if len(atr) > 0 else None,
        "RSI": rsi[-1] if len(rsi) > 0 else None,
        "MACD Line": macd_line[-1] if len(macd_line) > 0 else None,
        "Signal Line": signal_line[-1] if len(signal_line) > 0 else None,
        "Trend Direction": trend_direction[-1] if len(trend_direction) > 0 else None,
        "Latest Close": close
    }

    logging.info(f"Updated ATR: {latest_values['ATR']}")
    logging.info(f"Updated RSI: {latest_values['RSI']}")
    logging.info(f"Updated MACD Line: {latest_values['MACD Line']}, Signal Line: {latest_values['Signal Line']}")
    logging.info(f"Updated Trend Direction: {latest_values['Trend Direction']}")

    evaluate_trading_signals(atr, rsi, macd_line, signal_line, trend_direction, close, upper_band, lower_band)


def evaluate_trading_signals(atr, rsi, macd_line, signal_line, trend_direction, latest_close, upper_band, lower_band):
    global position

    logging.info("Evaluating trading signals...")

    long_condition = trend_direction[-1] == 1 and rsi[-1] > 50 and macd_line[-1] > signal_line[-1]
    short_condition = trend_direction[-1] == -1 and rsi[-1] < 50 and macd_line[-1] < signal_line[-1]

    if long_condition and position != "long":
        logging.info("Buy signal detected.")
        position = "long"
        asyncio.create_task(place_order(SIDE_BUY, QTY))

    elif short_condition and position != "short":
        logging.info("Sell signal detected.")
        position = "short"
        asyncio.create_task(place_order(SIDE_SELL, QTY))

    else:
        logging.info("No trade signal detected.")

async def place_order(side, quantity):
    client = await AsyncClient.create(TESTNET_API_KEY, TESTNET_SECRET_KEY, testnet=True)
    try:
        logging.info(f"Placing {side} order for {quantity} {BINANCE_SYMBOL}...")
        order = await client.create_order(
            symbol=BINANCE_SYMBOL,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        logging.info(f"Order successful: {order}")
    except Exception as e:
        logging.error(f"Error placing order: {e}", exc_info=True)
    finally:
        await client.close_connection()

# ============== HEARTBEAT LOGGING ==============
def heartbeat_logging():
    global latest_values
    while True:
        logging.info(f"Heartbeat - Latest Values: {latest_values}")
        if latest_values.get("Position"):
            logging.info(f"Current Position: {latest_values['Position']} - Latest Close: {latest_values['Latest Close']}")
        else:
            logging.info("Position is None or not set yet.")
        time.sleep(60)  # Log every 60 seconds

# ============== BINANCE WEBSOCKET ==============
async def start_binance_websocket():
    logging.info("Initializing Binance WebSocket...")
    client = await AsyncClient.create(TESTNET_API_KEY, TESTNET_SECRET_KEY, testnet=True)
    bsm = BinanceSocketManager(client)

    logging.info("Binance WebSocket initialized. Starting kline socket...")
    async def handle_message(msg):
        try:
            kline = msg['k']  # Extract kline data
            is_closed = kline['x']  # Check if the candle is closed
            if is_closed:
                high = float(kline['h'])
                low = float(kline['l'])
                close = float(kline['c'])
                logging.debug(f"Raw Kline Data - High: {high}, Low: {low}, Close: {close}")
                process_candle(high, low, close)
        except Exception as e:
            logging.error(f"Error while handling WebSocket message: {e}", exc_info=True)

    try:
        async with bsm.kline_socket(symbol=BINANCE_SYMBOL.lower(), interval=BINANCE_INTERVAL) as stream:
            logging.info(f"Listening for kline data on {BINANCE_SYMBOL} with interval {BINANCE_INTERVAL}.")
            while True:
                msg = await stream.recv()
                await handle_message(msg)
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        logging.info("Closing WebSocket connection...")
        await client.close_connection()

# ============== MAIN THREADS ==============
if __name__ == "__main__":
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080))
    flask_thread.daemon = True
    flask_thread.start()
    logging.info("Flask monitoring started on port 8080.")

    # Start heartbeat logging in a separate thread
    hb_thread = threading.Thread(target=heartbeat_logging, daemon=True)
    hb_thread.start()
    logging.info("Heartbeat logging thread started.")

    # Start asyncio tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_binance_websocket())



