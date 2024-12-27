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
def calculate_atr(high, low, close):
    tr1 = high - low
    tr2 = np.abs(high[1:] - close[:-1])  # High of current candle minus Close of previous candle
    tr3 = np.abs(low[1:] - close[:-1])   # Low of current candle minus Close of previous candle
    tr = np.maximum(tr1[1:], np.maximum(tr2, tr3))  # Take the max of the three TR values
    tr = np.concatenate(([tr1[0]], tr))  # Include the TR of the first candle

    atr = np.convolve(tr, np.ones(ATR_LENGTH) / ATR_LENGTH, mode='valid')
    return atr

def calculate_rsi(close):
    delta = np.diff(close)
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = np.convolve(gain, np.ones(RSI_LENGTH) / RSI_LENGTH, mode='valid')
    avg_loss = np.convolve(loss, np.ones(RSI_LENGTH) / RSI_LENGTH, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close):
    ema_fast = np.convolve(close, np.ones(MACD_FAST) / MACD_FAST, mode='valid')
    ema_slow = np.convolve(close, np.ones(MACD_SLOW) / MACD_SLOW, mode='valid')
    macd_line = ema_fast[-len(ema_slow):] - ema_slow
    signal_line = np.convolve(macd_line, np.ones(MACD_SIGNAL) / MACD_SIGNAL, mode='valid')
    return macd_line, signal_line

# ============== CANDLE PROCESSING ==============
def process_candle(high, low, close):
    global high_values, low_values, close_values, position

    high_values.append(high)
    low_values.append(low)
    close_values.append(close)

    if len(high_values) >= ATR_LENGTH:
        atr = calculate_atr(np.array(high_values), np.array(low_values), np.array(close_values))
        rsi = calculate_rsi(np.array(close_values))
        macd_line, signal_line = calculate_macd(np.array(close_values))

        logging.info(f"Updated ATR: {atr[-1]}")
        logging.info(f"Updated RSI: {rsi[-1]}")
        logging.info(f"Updated MACD Line: {macd_line[-1]}, Signal Line: {signal_line[-1]}")

        evaluate_trading_signals(atr, rsi, macd_line, signal_line, close_values[-1])

def evaluate_trading_signals(atr, rsi, macd_line, signal_line, latest_close):
    global position

    logging.info("Evaluating trading signals...")
    if latest_close > atr[-1] and rsi[-1] > 50 and macd_line[-1] > signal_line[-1] and position != "long":
        logging.info("Buy signal detected.")
        position = "long"
        # Place a buy order here
    elif latest_close < atr[-1] and rsi[-1] < 50 and macd_line[-1] < signal_line[-1] and position != "short":
        logging.info("Sell signal detected.")
        position = "short"
        # Place a sell order here
    else:
        logging.info("No trade signal detected.")

# ============== BINANCE WEBSOCKET ==============
async def start_binance_websocket():
    client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    bsm = BinanceSocketManager(client)

    async def handle_message(msg):
        kline = msg['k']  # Extract kline data
        is_closed = kline['x']  # Check if the candle is closed
        if is_closed:
            high = float(kline['h'])
            low = float(kline['l'])
            close = float(kline['c'])
            process_candle(high, low, close)

    async with bsm.kline_socket(symbol=BINANCE_SYMBOL.lower(), interval=BINANCE_INTERVAL) as stream:
        while True:
            msg = await stream.recv()
            handle_message(msg)

    await client.close_connection()

# ============== MAIN ==============
if __name__ == "__main__":
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080))
    flask_thread.daemon = True
    flask_thread.start()
    logging.info("Flask monitoring started on port 8080.")

    # Start asyncio tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_binance_websocket())

