import os
import sys
import time
import numpy as np
import pandas as pd
import logging
import datetime
from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from flask import Flask, send_file
from threading import Thread
from binance import ThreadedWebsocketManager
from binance.client import Client
from sklearn.cluster import KMeans
from alpaca_trade_api.rest import REST  # Alpaca API

def restart_program():
    """Restart the current program."""
    try:
        print("Restarting program...")
        time.sleep(2)  # Optional delay
        os.execv(sys.executable, ['python'] + sys.argv)
    except Exception as e:
        print(f"Failed to restart program: {e}")


# Alpaca API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters
ATR_LEN = 10
PRIMARY_ATR_FACTOR = 8
SECONDARY_ATR_FACTOR = 3
ALPACA_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTCUSDT"
QUANTITY = round(0.001, 8)
SIGNAL_INTERVAL = 30  # Seconds

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.log"),  # Save logs to file
        logging.StreamHandler()              # Output logs to console
    ]
)
logging.info("Trading bot initialized.")

# Global Variables
primary_volatility = []
secondary_volatility = []
last_price = None
last_direction = 1 # default as Bullish
last_signal_time = 0
last_heartbeat_time = 0  # Initialize heartbeat time
last_log_time = 0  # New variable for heartbeat logs
high, low, close = [], [], []
upper_band_history = []
lower_band_history = []
upper_band_300_history = []  # Stores only the 300-second upper bands
lower_band_300_history = []  # Stores only the 300-second lower bands
entry_price = None
primary_direction = 1  # Default to Bullish
secondary_direction = 1 # Default to Bullish

# Flask Server
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

@app.route('/logs')
def download_logs():
    try:
        # Use the absolute path to ensure correct file location
        return send_file("bot_logs.log", as_attachment=True)
    except FileNotFoundError:
        return "Log file not found.", 404

from flask import request, Response

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        "Please log in with valid credentials.", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

@app.before_request
def restrict_access():
    auth = request.authorization
    if not auth or not (auth.username == os.getenv("DASH_USER") and auth.password == os.getenv("DASH_PASS")):
        return authenticate()

# Initialize dash app
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/"
)

dash_app.layout = html.Div([
    html.H1("Trading Bot Dashboard", style={'text-align': 'center'}),
    dcc.Graph(id='price-chart', style={'height': '50vh'}),
    html.H3("Metrics Table", style={'margin-top': '20px'}),
    html.Table(id='metrics-table', style={'width': '100%', 'border': '1px solid black'}),
    dcc.Interval(
        id='update-interval',
        interval=30 * 1000,  # Refresh every 30 seconds
        n_intervals=0
    )
])

# Perform K-Means Clustering on volatility
def cluster_volatility(volatility, n_clusters=3):
    """
    Perform K-Means clustering on volatility data.

    :param volatility: List of volatility values.
    :param n_clusters: Number of clusters to create.
    :return: Centroids, assigned cluster, assigned centroid, cluster sizes, and dominant cluster.
    """
    try:
        # Input validation
        if len(volatility) < n_clusters:
            return None, None, None, None, None

        # Prepare data for clustering
        volatility = np.array(volatility).reshape(-1, 1)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(volatility)

        # Extract clustering results
        centroids = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]

        # Identify the cluster of the latest volatility value
        latest_volatility = volatility[-1][0]
        assigned_cluster = kmeans.predict([[latest_volatility]])[0]
        assigned_centroid = centroids[assigned_cluster]

        # Find the dominant cluster
        dominant_cluster = np.argmax(cluster_sizes)  # Cluster with the largest size

        return centroids, assigned_cluster, assigned_centroid, cluster_sizes, dominant_cluster

    except Exception:
        return None, None, None, None, None


# Calculate ATR
def calculate_atr(high, low, close, factor=1):
    """
    Calculate the Average True Range (ATR) with an optional multiplication factor.

    :param high: Series of high prices.
    :param low: Series of low prices.
    :param close: Series of close prices.
    :param factor: Multiplication factor for ATR (default is 1).
    :return: Latest ATR value or None if calculation fails.
    """
    try:
        # Input validation
        if len(high) < ATR_LEN or len(low) < ATR_LEN or len(close) < ATR_LEN:
            logging.warning("Insufficient data for ATR calculation.")
            return None

        # Calculate True Range (TR) components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR with the specified factor
        atr = tr.rolling(window=ATR_LEN).mean() * factor

        # Return the latest ATR value if available
        return atr.iloc[-1] if len(atr.dropna()) > 0 else None

    except Exception as e:
        logging.error(f"Error calculating ATR: {e}", exc_info=True)
        return None

# Initialize Volatility and Historical Data
def initialize_historical_data():
    """
    Initializes historical market data for high, low, close prices, and volatility.
    """
    global high, low, close, volatility, primary_volatility, secondary_volatility
    try:
        # Fetch historical data from Binance
        logging.info(f"Fetching historical data for {BINANCE_SYMBOL} with interval 1m.")
        klines = binance_client.get_klines(symbol=BINANCE_SYMBOL, interval="1m", limit=100 + ATR_LEN)
        
        # Convert to DataFrame and process
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume",
                                             "close_time", "quote_asset_volume", "number_of_trades",
                                             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        
        # Convert relevant columns to floats
        high = data["high"].astype(float).tolist()
        low = data["low"].astype(float).tolist()
        close = data["close"].astype(float).tolist()

        # Validate that we have enough data for ATR calculations
        if len(high) < ATR_LEN or len(low) < ATR_LEN or len(close) < ATR_LEN:
            logging.warning("Insufficient data for ATR calculation. Resetting values.")
            high, low, close, primary_volatility, secondary_volatility = [], [], [], [], []
            return

        # Calculate ATR for primary and secondary volatilities
        primary_atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close), factor=8)
        secondary_atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close), factor=3)

        if primary_atr is not None:
            primary_volatility = [primary_atr] * 100  # Initialize with consistent data
        else:
            primary_volatility = []

        if secondary_atr is not None:
            secondary_volatility = [secondary_atr] * 100  # Initialize with consistent data
        else:
            secondary_volatility = []

        # Update volatility (generic list for legacy compatibility)
        volatility = primary_volatility

        # Log initialization success or warning
        if not primary_volatility or not secondary_volatility:
            logging.warning("ATR calculation resulted in insufficient data for volatility.")
        else:
            logging.info(f"Initialized historical data for {BINANCE_SYMBOL} with {len(close)} close entries "
                         f"and valid ATR values for both signals.")
    except Exception as e:
        logging.error(f"Error initializing historical data for {BINANCE_SYMBOL}: {e}")
        # Reset to empty lists on failure
        high, low, close, volatility, primary_volatility, secondary_volatility = [], [], [], [], [], []


def initialize_direction(high, low, close, atr_factor, assigned_centroid):
    """
    Initialize the trading direction based on SuperTrend bands and historical trends.

    :param high: List of high prices.
    :param low: List of low prices.
    :param close: List of close prices.
    :param atr_factor: ATR multiplier for calculating bands.
    :param assigned_centroid: Volatility level determined by clustering.
    :return: Initial direction (1 for bullish, -1 for bearish).
    """
    try:
        # Convert inputs to Pandas Series for vectorized operations
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # Ensure sufficient data for calculation
        if len(high) < ATR_LEN or len(low) < ATR_LEN or len(close) < ATR_LEN:
            logging.warning("Insufficient data to initialize direction.")
            return 1  # Default to bullish if data is insufficient

        # Calculate HL2 (average of high and low) and bands
        hl2 = (high + low) / 2
        upper_band = hl2 + atr_factor * assigned_centroid
        lower_band = hl2 - atr_factor * assigned_centroid

        # Check the latest close price against the bands to determine direction
        if close.iloc[-1] < lower_band.iloc[-1]:
            direction = 1  # Bullish
        elif close.iloc[-1] > upper_band.iloc[-1]:
            direction = -1  # Bearish
        else:
            # Fallback: Compare to midpoint of bands for neutrality
            midpoint = (upper_band.iloc[-1] + lower_band.iloc[-1]) / 2
            direction = 1 if close.iloc[-1] < midpoint else -1

        # Log the initialization details
        logging.info(
            f"Initialized direction: {'Bullish (1)' if direction == 1 else 'Bearish (-1)'}\n"
            f"Last Close: {close.iloc[-1]:.2f}, Upper Band: {upper_band.iloc[-1]:.2f}, Lower Band: {lower_band.iloc[-1]:.2f}"
        )

        return direction

    except Exception as e:
        logging.error(f"Error in initialize_direction: {e}", exc_info=True)
        return 1  # Default to bullish if an error occurs


# Calculate SuperTrend
def calculate_supertrend_with_clusters(high, low, close, assigned_centroid, atr_factor, previous_direction):
    """
    Calculate SuperTrend and determine the trading direction with persistence logic.
    """
    try:
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # Calculate HL2 (average of high and low)
        hl2 = (high + low) / 2
        upper_band = hl2 + atr_factor * assigned_centroid
        lower_band = hl2 - atr_factor * assigned_centroid

        # Ensure band continuity
        upper_band = upper_band.where(upper_band < upper_band.shift(1), upper_band.shift(1))
        lower_band = lower_band.where(lower_band > lower_band.shift(1), lower_band.shift(1))

        # Determine direction
        if close.iloc[-1] > upper_band.iloc[-1]:
            direction = -1  # Bearish
        elif close.iloc[-1] < lower_band.iloc[-1]:
            direction = 1   # Bullish
        else:
            direction = previous_direction  # Persist direction

        return direction, upper_band, lower_band

    except Exception as e:
        logging.error(f"Error in calculate_supertrend_with_clusters: {e}", exc_info=True)
        return previous_direction, None, None



# Execute Trades
def execute_trade(symbol, quantity, side):
    """
    Executes a trade order with Alpaca and handles errors.
    
    :param symbol: The trading symbol (e.g., BTC/USD).
    :param quantity: The quantity to trade.
    :param side: The side of the trade ("buy" or "sell").
    """
    try:
        logging.info(f"Attempting to {side} {quantity} units of {symbol}.")
        
        # Submit the order to Alpaca
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        
        # Confirm order status
        logging.info(f"{side.capitalize()} order submitted successfully. Order ID: {order.id}")
        
        # Retrieve and log order details
        order_details = alpaca_api.get_order(order.id)
        logging.info(
            f"Order Details:\n"
            f"Symbol: {order_details.symbol}\n"
            f"Quantity: {order_details.qty}\n"
            f"Filled Quantity: {order_details.filled_qty}\n"
            f"Side: {order_details.side}\n"
            f"Status: {order_details.status}\n"
            f"Submitted At: {order_details.submitted_at}"
        )
        
        # Handle partial fills
        if order_details.status not in ["filled", "partially_filled"]:
            logging.warning(f"Order status: {order_details.status}. Please check manually.")
    
    except Exception as e:
        logging.error(f"Error executing {side} order for {symbol}: {e}", exc_info=True)

# Function for Heartbeat Logging
def heartbeat_logging():
    """
    Logs current status and data every 30 seconds for monitoring purposes.
    Updates band history to ensure data is current.
    """
    global primary_direction, secondary_direction
    if not volatility or len(volatility) < 3:
        logging.info("Heartbeat: Insufficient data for detailed logging.")
        return

    try:
        # Perform clustering for primary and secondary signals
        primary_centroids, primary_assigned_cluster, primary_assigned_centroid, primary_cluster_sizes, primary_dominant_cluster = cluster_volatility(primary_volatility)
        secondary_centroids, secondary_assigned_cluster, secondary_assigned_centroid, secondary_cluster_sizes, secondary_dominant_cluster = cluster_volatility(secondary_volatility)

        # Calculate SuperTrend for both signals
        new_primary_direction, primary_upper_band, primary_lower_band = calculate_supertrend_with_clusters(
            pd.Series(high), pd.Series(low), pd.Series(close), primary_assigned_centroid, PRIMARY_ATR_FACTOR, primary_direction
        )
        new_secondary_direction, secondary_upper_band, secondary_lower_band = calculate_supertrend_with_clusters(
            pd.Series(high), pd.Series(low), pd.Series(close), secondary_assigned_centroid, SECONDARY_ATR_FACTOR, secondary_direction
        )

        # Update directions globally
        primary_direction = new_primary_direction
        secondary_direction = new_secondary_direction

        # Stop-Loss and Take-Profit Logic
        stop_loss = None
        take_profit = None
        if entry_price is not None:
            stop_loss = secondary_lower_band.iloc[-1]
            take_profit = stop_loss * 1.5

        # Log Heartbeat Information
        logging.info(
            f"\n=== Heartbeat Logging ===\n"
            f"Price: {last_price:.2f}\n"
            f"Primary Cluster Centroids: {', '.join(f'{x:.2f}' for x in primary_centroids)}\n"
            f"Primary Cluster Sizes: {', '.join(str(size) for size in primary_cluster_sizes)}\n"
            f"Primary Dominant Cluster: {primary_dominant_cluster}\n"
            f"Secondary Cluster Centroids: {', '.join(f'{x:.2f}' for x in secondary_centroids)}\n"
            f"Secondary Cluster Sizes: {', '.join(str(size) for size in secondary_cluster_sizes)}\n"
            f"Secondary Dominant Cluster: {secondary_dominant_cluster}\n"
            f"Primary Direction: {'Bullish (1)' if primary_direction == 1 else 'Bearish (-1)'}\n"
            f"Secondary Direction: {'Bullish (1)' if secondary_direction == 1 else 'Bearish (-1)'}\n"
            f"Entry Price: {entry_price if entry_price else 'None'}\n"
            f"Stop Loss: {stop_loss if stop_loss else 'None'}\n"
            f"Take Profit: {take_profit if take_profit else 'None'}\n"
            f"=========================="
        )

    except Exception as e:
        logging.error(f"Error during heartbeat logging: {e}", exc_info=True)



    except Exception as e:
        logging.error(f"Error during heartbeat logging: {e}", exc_info=True)


# Signal Processing
def calculate_and_execute(price, primary_direction, secondary_direction):
    """
    Process trading signals and execute trades based on price and direction logic.

    :param price: Current price.
    :param primary_direction: Current direction of the primary signal.
    :param secondary_direction: Current direction of the secondary signal.
    :return: Updated primary_direction and secondary_direction.
    """
    global entry_price # Ensure this global variable is properly used

    # Perform clustering separately for primary and secondary signals
    primary_centroids, _, primary_assigned_centroid, primary_cluster_sizes, primary_dominant_cluster = cluster_volatility(primary_volatility)
    secondary_centroids, _, secondary_assigned_centroid, secondary_cluster_sizes, secondary_dominant_cluster = cluster_volatility(secondary_volatility)




    # Calculate ATR and SuperTrend for primary and secondary signals
    new_primary_direction, _, _ = calculate_supertrend_with_clusters(
        high, low, close, primary_assigned_centroid, PRIMARY_ATR_FACTOR, primary_direction
    )
    new_secondary_direction, secondary_upper_band, secondary_lower_band = calculate_supertrend_with_clusters(
        high, low, close, secondary_assigned_centroid, SECONDARY_ATR_FACTOR, secondary_direction
    )

    # Handle Stop-Loss and Take-Profit Logic
    stop_loss = None
    take_profit = None
    if entry_price is not None:
        stop_loss = secondary_lower_band.iloc[-1]
        take_profit = stop_loss * 1.5

        if price <= stop_loss:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
            logging.info(f"Stop-loss triggered. Exiting trade. Price: {price:.2f}")
            entry_price = None  # Reset entry price

        elif price >= take_profit:
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
            logging.info(f"Take-profit triggered. Exiting trade. Price: {price:.2f}")
            entry_price = None  # Reset entry price

    # Execute Buy Signal if conditions are met
    buy_signal = (
        primary_dominant_cluster == 3 and  # Primary clustering shows high volatility
        primary_direction == 1 and  # Primary signal is bullish
        secondary_direction == 1 and  # Secondary signal is bullish
        secondary_dominant_cluster == 3 and  # Secondary clustering shows high volatility
        entry_price is None  # No active trade
    )
    if buy_signal:
        execute_trade(ALPACA_SYMBOL, QUANTITY, "buy")
        entry_price = price
        logging.info(f"Buy signal triggered. Entry price: {price:.2f}")

    # Logging
    stop_loss_display = f"{stop_loss:.2f}" if stop_loss else "None"
    take_profit_display = f"{take_profit:.2f}" if take_profit else "None"
    logging.info(
        f"\n=== Combined Logging ===\n"
        f"Price: {last_price:.2f}\n"
        f"Primary Clustering Centroids: {', '.join(f'{x:.2f}' for x in primary_centroids)}\n"
        f"Primary Dominant Cluster: {primary_dominant_cluster}\n"
        f"Secondary Clustering Centroids: {', '.join(f'{x:.2f}' for x in secondary_centroids)}\n"
        f"Secondary Dominant Cluster: {secondary_dominant_cluster}\n"
        f"Cluster Sizes (Primary): {', '.join(str(size) for size in primary_cluster_sizes)}\n"
        f"Cluster Sizes (Secondary): {', '.join(str(size) for size in secondary_cluster_sizes)}\n"
        f"Primary ATR (Current): {atr_primary.iloc[-1]:.2f}\n"
        f"Secondary ATR (Current): {atr_secondary.iloc[-1]:.2f}\n"
        f"Primary Direction: {'Bullish (1)' if primary_direction == 1 else 'Bearish (-1)'}\n"
        f"Secondary Direction: {'Bullish (1)' if secondary_direction == 1 else 'Bearish (-1)'}\n"
        f"Entry Price: {entry_price if entry_price else 'None'}\n"
        f"Stop Loss: {stop_loss_display}\n"
        f"Take Profit: {take_profit_display}\n"
        f"Primary Upper Band (Current): {primary_upper_band.iloc[-1]:.2f}\n"
        f"Primary Lower Band (Current): {primary_lower_band.iloc[-1]:.2f}\n"
        f"Secondary Upper Band (Current): {secondary_upper_band.iloc[-1]:.2f}\n"
        f"Secondary Lower Band (Current): {secondary_lower_band.iloc[-1]:.2f}\n"
        f"=========================="
    )


    return new_primary_direction, new_secondary_direction



# Update dash
def update_dashboard():
    global high, low, close, upper_band_history, lower_band_history, last_price, primary_direction, secondary_direction

    # Create price and bands chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=close[-10:], mode='lines', name='Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=upper_band_history[-10:], mode='lines', name='Primary Upper Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=lower_band_history[-10:], mode='lines', name='Primary Lower Band', line=dict(color='red')))

    # Create table rows for metrics
    metrics = [
        ["Last Price", f"{last_price:.2f}" if last_price else "N/A"],
        ["Primary Direction", f"{'Bullish (1)' if primary_direction == 1 else 'Bearish (-1)'}"],
        ["Secondary Direction", f"{'Bullish (1)' if secondary_direction == 1 else 'Bearish (-1)'}"],
        ["Primary Upper Band", f"{upper_band_history[-1]:.2f}" if upper_band_history else "N/A"],
        ["Primary Lower Band", f"{lower_band_history[-1]:.2f}" if lower_band_history else "N/A"],
        ["High (last)", f"{high[-1]:.2f}" if high else "N/A"],
        ["Low (last)", f"{low[-1]:.2f}" if low else "N/A"]
    ]
    rows = [html.Tr([html.Th(metric[0]), html.Td(metric[1])]) for metric in metrics]

    return fig, rows

# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close, primary_volatility, secondary_volatility

    if 'k' not in msg:
        return

    candle = msg['k']
    last_price = float(candle['c'])
    high.append(float(candle['h']))
    low.append(float(candle['l']))
    close.append(float(candle['c']))

    # Limit the length of historical data
    if len(high) > ATR_LEN + 1:
        high.pop(0)
    if len(low) > ATR_LEN + 1:
        low.pop(0)
    if len(close) > ATR_LEN + 1:
        close.pop(0)

    # Update ATR for both signals
    if len(high) >= ATR_LEN:
        primary_atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close), factor=8)
        secondary_atr = calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close), factor=3)

        if primary_atr:
            primary_volatility.append(primary_atr)
            if len(primary_volatility) > 100:
                primary_volatility.pop(0)

        if secondary_atr:
            secondary_volatility.append(secondary_atr)
            if len(secondary_volatility) > 100:
                secondary_volatility.pop(0)


# WebSocket Manager
def start_websocket():
    """Starts the Binance WebSocket for real-time price data."""
    while True:  # Keep the WebSocket running
        try:
            logging.info("Starting WebSocket connection...")
            # Initialize ThreadedWebsocketManager
            twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            twm.start()

            # Start streaming 5-minute candles
            twm.start_kline_socket(callback=on_message, symbol=BINANCE_SYMBOL.lower(), interval="1m")
            twm.join()  # Keep the WebSocket connection open
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            logging.info(f"Reconnecting in 30 seconds...")
            time.sleep(30)  # Wait before reconnecting

def process_signals():
    global last_signal_time, last_heartbeat_time, primary_direction, secondary_direction

    # Align to the next 30-second mark minus a few seconds (e.g., 3 seconds before)
    now = datetime.datetime.now()
    next_heartbeat = now + datetime.timedelta(seconds=(30 - now.second % 30))
    sync_offset = 3  # Start a few seconds earlier
    next_heartbeat_delay = max(0, (next_heartbeat - now).total_seconds() - sync_offset)
    logging.info(f"Synchronizing to next heartbeat in {next_heartbeat_delay:.2f} seconds.")
    time.sleep(next_heartbeat_delay)

    while True:
        current_time = time.time()

        try:
            # Signal processing every SIGNAL_INTERVAL seconds
            if current_time - last_signal_time >= SIGNAL_INTERVAL:
                if last_price is not None:
                    # Perform clustering separately for primary and secondary volatilities
                    primary_centroids, _, primary_assigned_centroid, _, primary_dominant_cluster = cluster_volatility(primary_volatility)
                    secondary_centroids, _, secondary_assigned_centroid, _, secondary_dominant_cluster = cluster_volatility(secondary_volatility)

                    # Execute trading logic
                    primary_direction, secondary_direction = calculate_and_execute(
                        last_price, primary_direction, secondary_direction
                    )
                last_signal_time = current_time  # Update the last signal time

            # Heartbeat logging every 30 seconds
            if current_time - last_heartbeat_time >= 30:
                heartbeat_logging()
                last_heartbeat_time = current_time  # Update the last heartbeat time

        except Exception as e:
            logging.error(f"Error in process_signals loop: {e}", exc_info=True)

        # Sleep for a small interval to avoid excessive CPU usage
        time.sleep(1)

# Main Script Entry Point
if __name__ == "__main__":
    # Initialize historical data
    initialize_historical_data()
    # Perform clustering for both signals
    primary_centroids, _, primary_assigned_centroid, _, primary_dominant_cluster = cluster_volatility(primary_volatility)
    secondary_centroids, _, secondary_assigned_centroid, _, secondary_dominant_cluster = cluster_volatility(secondary_volatility)


    # Validate clustering results before proceeding
    if primary_assigned_centroid is None or secondary_assigned_centroid is None:
        raise ValueError("Clustering failed for one or both signals. Check volatility data.")

    # Initialize directions
    primary_direction = initialize_direction(high, low, close, PRIMARY_ATR_FACTOR, primary_assigned_centroid)
    secondary_direction = initialize_direction(high, low, close, SECONDARY_ATR_FACTOR, secondary_assigned_centroid)


    logging.info(f"Primary Direction: {'Bullish (1)' if primary_direction == 1 else 'Bearish (-1)'}")
    logging.info(f"Secondary Direction: {'Bullish (1)' if secondary_direction == 1 else 'Bearish (-1)'}")

    # Start Flask app in a separate thread
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    time.sleep(10)

    # Start Signal Processing Loop in a separate thread
    Thread(target=process_signals, daemon=True).start()

    # Start WebSocket for real-time data
    try:
        start_websocket()
    except Exception as e:
        logging.error(f"Critical failure in WebSocket connection: {e}. Restarting...")
        restart_program()
