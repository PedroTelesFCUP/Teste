import os
import sys
import logging
import threading
import time
import numpy as np
from flask import Flask, request, Response, send_file, jsonify
from binance.client import Client
from binance.streams import BinanceSocketManager
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

# Testnet client
TESTNET_BASE_URL = "https://testnet.binance.vision"
client = Client(TESTNET_API_KEY, TESTNET_SECRET_KEY, testnet=True)
client.API_URL = TESTNET_BASE_URL

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
    try:
        orders = client.get_all_orders(symbol=BINANCE_SYMBOL)
        formatted_orders = [
            {
                "Order ID": order["orderId"],
                "Status": order["status"],
                "Side": order["side"],
                "Price": order["price"],
                "Quantity": order["origQty"],
                "Executed Quantity": order["executedQty"],
                "Time": order["time"]
            }
            for order in orders
        ]
        return jsonify(formatted_orders)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============== INDICATOR CALCULATIONS ==============
def calculate_atr(high, low, close):
    tr = np.maximum(high - low, np.maximum(np.abs(high - close[:-1]), np.abs(low - close[:-1])))
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

# ============== SIGNAL CHECKING ==============
def check_signals():
    position = None
    while True:
        try:
            logging.info("Fetching market data...")
            klines = client.get_klines(symbol=BINANCE_SYMBOL, interval=BINANCE_INTERVAL, limit=100)
            data = np.array(klines, dtype=float)
            high = data[:, 2]
            low = data[:, 3]
            close = data[:, 4]

            logging.info("Calculating indicators...")
            atr = calculate_atr(high, low, close)
            rsi = calculate_rsi(close)
            macd_line, signal_line = calculate_macd(close)

            # Log the indicator values for debugging
            logging.info(f"Latest ATR: {atr[-1]}")
            logging.info(f"Latest RSI: {rsi[-1]}")
            logging.info(f"Latest MACD Line: {macd_line[-1]}")
            logging.info(f"Latest Signal Line: {signal_line[-1]}")

            logging.info("Checking trading conditions...")
            if close[-1] > atr[-1] and rsi[-1] > 50 and macd_line[-1] > signal_line[-1] and position != "long":
                logging.info("Buy signal detected.")
                client.order_market_buy(symbol=BINANCE_SYMBOL, quantity=QTY)
                position = "long"
            elif close[-1] < atr[-1] and rsi[-1] < 50 and macd_line[-1] < signal_line[-1] and position != "short":
                logging.info("Sell signal detected.")
                client.order_market_sell(symbol=BINANCE_SYMBOL, quantity=QTY)
                position = "short"
            else:
                logging.info("No trade signal detected.")

            time.sleep(60)  # Wait for the next candle
        except Exception as e:
            logging.error(f"Error in signal checking: {e}", exc_info=True)

# ============== HEARTBEAT LOGGING ==============
def heartbeat_logging():
    while True:
        logging.info("Heartbeat: Bot is running...")
        time.sleep(60)  # Log heartbeat every minute

# ============== BINANCE WEBSOCKET ==============
def start_binance_websocket():
    bsm = BinanceSocketManager(client)
    def handle_message(msg):
        logging.info(f"WebSocket message received: {msg}")
        # Process incoming data here (e.g., trade signals)
    
    conn_key = bsm.start_kline_socket(BINANCE_SYMBOL, handle_message, interval=BINANCE_INTERVAL)
    bsm.start()

# ============== MAIN ==============
if __name__ == "__main__":
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080))
    flask_thread.daemon = True
    flask_thread.start()
    logging.info("Flask monitoring started on port 8080.")

    # Start the heartbeat logging thread
    hb_thread = threading.Thread(target=heartbeat_logging, daemon=True)
    hb_thread.start()
    logging.info("Heartbeat logging thread started.")

    # Start signals checking thread
    signal_thread = threading.Thread(target=check_signals, daemon=True)
    signal_thread.start()
    logging.info("Signal checking thread started.")

    # Start Binance WebSocket
    try:
        start_binance_websocket()
    except Exception as e:
        logging.error(f"Binance WebSocket error: {e}", exc_info=True)
        sys.exit(1)

