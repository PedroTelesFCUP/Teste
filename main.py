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
PRIMARY_ATR_FACTOR = 7
SECONDARY_ATR_FACTOR = 2.5
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
primary_upper_band = []
primary_lower_band = []
secondary_upper_band = []
secondary_lower_band = []
last_secondary_directions = []
entry_price = None
primary_direction = 1  # Default to Bullish
secondary_direction = 1 # Default to Bullish
primary_dominant_cluster = None
secondary_dominant_cluster = None
primary_centroids = []
secondary_centroids = []
primary_cluster_sizes = []
secondary_cluster_sizes = []
last_centroids = None 
volatility_threshold = 0.99  # Define a threshold for significant volatility changes
RECALC_INTERVAL = 100  # Fixed recalculation interval for clustering
buy_signal = {}
sell_signal = {}

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

# Dashboard layout update
dash_app.layout = html.Div([
    html.H1("Trading Bot Dashboard", style={'text-align': 'center'}),
    dcc.Graph(id='primary-chart', style={'height': '50vh'}),
    dcc.Graph(id='secondary-chart', style={'height': '50vh'}),
    html.H3("Metrics Table", style={'margin-top': '20px'}),
    html.Table(id='metrics-table', style={'width': '100%', 'border': '1px solid black'}),
    dcc.Interval(
        id='update-interval',
        interval=30 * 1000,  # Refresh every 30 seconds
        n_intervals=0
    )
])

@dash_app.callback(
    [Output('primary-chart', 'figure'),
     Output('secondary-chart', 'figure'),
     Output('metrics-table', 'children')],
    [Input('update-interval', 'n_intervals')]
)
@dash_app.callback(
    [Output('primary-chart', 'figure'),
     Output('secondary-chart', 'figure'),
     Output('metrics-table', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_dashboard_callback(n):
    try:
        # Validate necessary variables
        if len(high) < ATR_LEN or len(low) < ATR_LEN or len(close) < ATR_LEN:
            raise ValueError("Insufficient data for dashboard update.")

        # Primary Chart
        fig_primary = go.Figure()
        fig_primary.add_trace(go.Scatter(y=close[-10:], mode='lines', name='Close Price', line=dict(color='blue')))
        fig_primary.add_trace(go.Scatter(y=primary_upper_band[-10:], mode='lines', name='Primary Upper Band', line=dict(color='green', dash='dash')))
        fig_primary.add_trace(go.Scatter(y=primary_lower_band[-10:], mode='lines', name='Primary Lower Band', line=dict(color='red', dash='dash')))

        # Secondary Chart
        fig_secondary = plot_signals_with_markers(
            high, low, close, secondary_direction, secondary_upper_band, secondary_lower_band, buy_sell_signals
        )

        # Metrics Table
        metrics = [
            ["Price", f"{last_price:.2f}" if last_price else "N/A"],
            ["Primary Clustering Centroids", ", ".join(f"{x:.2f}" for x in primary_centroids)],
            ["Primary Cluster Sizes", ", ".join(str(size) for size in primary_cluster_sizes)],
            ["Primary Dominant Cluster", f"{primary_dominant_cluster}"],
            ["Secondary Clustering Centroids", ", ".join(f"{x:.2f}" for x in secondary_centroids)],
            ["Secondary Cluster Sizes", ", ".join(str(size) for size in secondary_cluster_sizes)],
            ["Secondary Dominant Cluster", f"{secondary_dominant_cluster}"],
            ["Primary ATR", f"{primary_volatility[-1]:.2f}"],
            ["Secondary ATR", f"{secondary_volatility[-1]:.2f}"]
        ]
        rows = [html.Tr([html.Th(metric[0]), html.Td(metric[1])]) for metric in metrics]

        return fig_primary, fig_secondary, rows

    except Exception as e:
        logging.error(f"Error updating dashboard: {e}", exc_info=True)
        return (
            go.Figure().update_layout(title="Primary Chart (Error)", template="plotly_white"),
            go.Figure().update_layout(title="Secondary Chart (Error)", template="plotly_white"),
            [html.Tr([html.Th("Error"), html.Td("An error occurred while updating the dashboard.")])]
        )


# Perform K-Means Clustering on volatility
def cluster_volatility(volatility, n_clusters=3):
    """
    Perform clustering on volatility using the last 100 points for stability.

    Parameters:
    - volatility: List of volatility values.
    - n_clusters: Number of clusters.

    Returns:
    - centroids: Stabilized centroids of the clusters.
    - assigned_cluster: Cluster index (1, 2, 3) for the most recent volatility value.
    - assigned_centroid: Centroid value of the assigned cluster.
    - cluster_sizes: Sizes of each cluster.
    - dominant_cluster: Index (1, 2, 3) of the cluster with the highest size.
    """
    global last_centroids
    try:
        # Ensure sufficient data for clustering


        # Use the last 100 points
        window_volatility = volatility[-100:]

        # Initialize centroids if not already set
        if not last_centroids:
            centroids = [
                np.percentile(window_volatility, 25),
                np.percentile(window_volatility, 50),
                np.percentile(window_volatility, 75)
            ]
        else:
            centroids = last_centroids

        # Assign values to clusters
        clusters = {i: [] for i in range(n_clusters)}
        for v in window_volatility:
            distances = [abs(v - c) for c in centroids]
            cluster = distances.index(min(distances))
            clusters[cluster].append(v)

        # Calculate cluster sizes
        cluster_sizes = [len(clusters[i]) for i in range(n_clusters)]

        # Update centroids based on cluster averages
        centroids = [np.mean(clusters[i]) for i in range(n_clusters)]
        last_centroids = centroids

        # Assign the most recent volatility value
        latest_volatility = volatility[-1]
        distances = [abs(latest_volatility - c) for c in centroids]
        assigned_cluster = distances.index(min(distances)) + 1
        assigned_centroid = centroids[assigned_cluster - 1]

        # Determine the dominant cluster
        dominant_cluster = cluster_sizes.index(max(cluster_sizes)) + 1

        return centroids, assigned_cluster, assigned_centroid, cluster_sizes, dominant_cluster

    except Exception as e:
        logging.error(f"Error in cluster_volatility: {e}", exc_info=True)
        return [None] * 5


# Calculate ATR
def calculate_atr(high, low, close, factor=3):
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
    global high, low, close, primary_volatility, secondary_volatility
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


        # Log initialization success or warning
        if not primary_volatility or not secondary_volatility:
            logging.warning("ATR calculation resulted in insufficient data for volatility.")
        else:
            logging.info(f"Initialized historical data for {BINANCE_SYMBOL} with {len(close)} close entries "
                         f"and valid ATR values for both signals.")
    except Exception as e:
        logging.error(f"Error initializing historical data for {BINANCE_SYMBOL}: {e}")
        # Reset to empty lists on failure
        high, low, close, primary_volatility, secondary_volatility = [], [], [], [], []


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
def execute_trade(symbol, quantity, side, stop_loss=None, take_profit=None):
    """
    Executes a trade order with Alpaca and handles errors.
    
    :param symbol: The trading symbol (e.g., BTC/USD).
    :param quantity: The quantity to trade.
    :param side: The side of the trade ("buy" or "sell").
    :param stop_loss: The stop-loss price.
    :param take_profit: The take-profit price.
    """
    try:
        logging.info(f"Attempting to {side} {quantity} units of {symbol}.")
        
        # Submit the order to Alpaca with stop-loss and take-profit
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc",
            stop_loss={"stop_price": stop_loss} if stop_loss else None,
            take_profit={"limit_price": take_profit} if take_profit else None
        )
        
        # Confirm order status
        logging.info(f"{side.capitalize()} order submitted successfully. Order ID: {order.id}")
        
    except Exception as e:
        logging.error(f"Error executing {side} trade for {symbol}: {e}", exc_info=True)

# Function for Heartbeat Logging
def heartbeat_logging():
    """
    Logs current status and data every 30 seconds for monitoring purposes.
    Updates band history to ensure data is current.
    """
    global primary_direction, secondary_direction

    try:
        # Perform clustering for primary and secondary signals
        primary_centroids, primary_assigned_cluster, primary_assigned_centroid, primary_cluster_sizes, primary_dominant_cluster = cluster_volatility(primary_volatility, n_clusters=3)
        secondary_centroids, secondary_assigned_cluster, secondary_assigned_centroid, secondary_cluster_sizes, secondary_dominant_cluster = cluster_volatility(secondary_volatility, n_clusters=3)

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
            f"Primary Clustering Centroids: {', '.join(f'{x:.2f}' for x in primary_centroids)}\n"
            f"Primary Cluster Sizes: {', '.join(str(size) for size in primary_cluster_sizes)}\n"
            f"Primary Dominant Cluster: {primary_dominant_cluster}\n"
            f"Secondary Clustering Centroids: {', '.join(f'{x:.2f}' for x in secondary_centroids)}\n"
            f"Secondary Cluster Sizes: {', '.join(str(size) for size in secondary_cluster_sizes)}\n"
            f"Secondary Dominant Cluster: {secondary_dominant_cluster}\n"
            f"Primary Direction: {'Bullish (1)' if primary_direction == 1 else 'Bearish (-1)'}\n"
            f"Secondary Direction: {'Bullish (1)' if secondary_direction == 1 else 'Bearish (-1)'}\n"
            f"Entry Price: {entry_price if entry_price else 'None'}\n"
            f"Stop Loss (Current): {stop_loss if stop_loss else 'None'}\n"
            f"Take Profit (Current): {take_profit if take_profit else 'None'}\n"
            f"Primary ATR (Current): {primary_volatility[-1]:.2f}\n"
            f"Secondary ATR (Current): {secondary_volatility[-1]:.2f}\n"
            f"Primary Upper Band (Current): {primary_upper_band.iloc[-1]:.2f}\n"
            f"Primary Lower Band (Current): {primary_lower_band.iloc[-1]:.2f}\n"
            f"Secondary Upper Band (Current): {secondary_upper_band.iloc[-1]:.2f}\n"
            f"Secondary Lower Band (Current): {secondary_lower_band.iloc[-1]:.2f}\n"
            f"=========================="
        )


    except Exception as e:
        logging.error(f"Error during heartbeat logging: {e}", exc_info=True)



    except Exception as e:
        logging.error(f"Error during heartbeat logging: {e}", exc_info=True)


# Signal Processing
def calculate_and_execute(price, primary_direction, secondary_direction, 
                          primary_cluster_sizes, secondary_cluster_sizes, 
                          primary_dominant_cluster, secondary_dominant_cluster):
    """
    Process trading signals and execute trades based on price and direction logic.

    :param price: Current price.
    :param primary_direction: Current direction of the primary signal.
    :param secondary_direction: Current direction of the secondary signal.
    :param primary_cluster_sizes: Sizes of the primary clusters.
    :param secondary_cluster_sizes: Sizes of the secondary clusters.
    :param primary_dominant_cluster: Dominant cluster for the primary signal.
    :param secondary_dominant_cluster: Dominant cluster for the secondary signal.
    :return: Updated primary_direction and secondary_direction.
    """
    global entry_price  # Ensure this global variable is properly used

    # Ensure ATR values are available
    if not primary_volatility or not secondary_volatility:
        logging.error("ATR values are missing.")
        return primary_direction, secondary_direction

    # Perform clustering to fetch fresh results
    primary_centroids, _, primary_assigned_centroid, primary_cluster_sizes, primary_dominant_cluster = cluster_volatility(primary_volatility, n_clusters=3)
    secondary_centroids, _, secondary_assigned_centroid, secondary_cluster_sizes, secondary_dominant_cluster = cluster_volatility(secondary_volatility,  n_clusters=3)


                              
    # Calculate SuperTrend
    new_primary_direction, primary_upper_band, primary_lower_band = calculate_supertrend_with_clusters(
        high, low, close, primary_volatility[-1], PRIMARY_ATR_FACTOR, primary_direction
    )
    new_secondary_direction, secondary_upper_band, secondary_lower_band = calculate_supertrend_with_clusters(
        high, low, close, secondary_volatility[-1], SECONDARY_ATR_FACTOR, secondary_direction
    )

    # Check for specific transition patterns in secondary signal
    bearish_bullish_bearish = (
        last_secondary_directions[-3:] == [-1, 1, -1]  # Bearish → Bullish → Bearish
    )
    bullish_bearish_bullish = (
        last_secondary_directions[-3:] == [1, -1, 1]  # Bullish → Bearish → Bullish
    )

    # Relaxed mode: At least two bearish or bullish signals in the last 10 cycles
    relaxed_mode = (
        last_secondary_directions.count(-1) > 1 or
        last_secondary_directions.count(1) > 1
    )

    # Handle Stop-Loss and Take-Profit Logic
    stop_loss = None
    take_profit = None
    if entry_price is not None:
        stop_loss = secondary_lower_band.iloc[-1]
        take_profit = stop_loss * 1.5

        if price <= stop_loss:
            profit_loss = price - entry_price
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
            logging.info(f"Stop-loss triggered. Exiting trade. Price: {price:.2f}, Profit/Loss: {profit_loss:.2f}")
            entry_price = None  # Reset entry price

        elif price >= take_profit:
            profit_loss = price - entry_price
            execute_trade(ALPACA_SYMBOL, QUANTITY, "sell")
            logging.info(f"Take-profit triggered. Exiting trade. Price: {price:.2f}, Profit/Loss: {profit_loss:.2f}")
            entry_price = None  # Reset entry price

    # Buy signal
    buy_signal = (
        primary_direction == 1 and  # Primary signal is bullish
        bullish_bearish_bullish and
        primary_dominant_cluster == 3 and  # High volatility for primary
        secondary_dominant_cluster in [2, 3] and  # Medium or high volatility for secondary in relaxed mode
        entry_price is None  # No active trade
    )
    if buy_signal:
        # Calculate stop-loss and take-profit
        stop_loss = secondary_lower_band.iloc[-1] if secondary_lower_band is not None else None
        take_profit = stop_loss * 1.5 if stop_loss else None

        if stop_loss is None or take_profit is None:
            logging.error("Missing stop-loss or take-profit. Skipping buy trade.")
        else:
            try:
                execute_trade(ALPACA_SYMBOL, QUANTITY, "buy", stop_loss=stop_loss, take_profit=take_profit)
                entry_price = price  # Record entry price
                logging.info(
                    f"Buy signal triggered. Entry price: {price:.2f}, "
                    f"Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}"
                )
            except Exception as e:
                logging.error(f"Error executing buy trade: {e}")


    # Sell signal
    sell_signal = (
        primary_direction == -1 and  # Primary signal is bearish
        bearish_bullish_bearish and
        primary_dominant_cluster == 3 and  # High volatility for primary
        secondary_dominant_cluster in [2, 3] and  # Medium or high volatility for secondary in relaxed mode
        entry_price is None  # No active trade
    )
    if sell_signal:
        # Calculate stop-loss and take-profit
        stop_loss = secondary_upper_band.iloc[-1] if secondary_upper_band is not None else None
        take_profit = stop_loss * 1.5 if stop_loss else None

        if stop_loss is None or take_profit is None:
            logging.error("Missing stop-loss or take-profit. Skipping sell trade.")
        else:
            try:
                execute_trade(ALPACA_SYMBOL, QUANTITY, "sell", stop_loss=stop_loss, take_profit=take_profit)
                entry_price = price  # Record entry price
                logging.info(
                    f"Sell signal triggered. Entry price: {price:.2f}, "
                    f"Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}"
                )
            except Exception as e:
                logging.error(f"Error executing sell trade: {e}")
    # Logging
    stop_loss_display = f"{stop_loss:.2f}" if stop_loss else "None"
    take_profit_display = f"{take_profit:.2f}" if take_profit else "None"
    logging.info(
        f"\n=== Combined Logging ===\n"
        f"Price: {price:.2f}\n"
        f"Primary Clustering Centroids: {', '.join(f'{x:.2f}' for x in primary_centroids)}\n"
        f"Primary Cluster Sizes: {', '.join(str(size) for size in primary_cluster_sizes)}\n"
        f"Primary Dominant Cluster: {primary_dominant_cluster}\n"
        f"Secondary Clustering Centroids: {', '.join(f'{x:.2f}' for x in secondary_centroids)}\n"
        f"Secondary Cluster Sizes: {', '.join(str(size) for size in secondary_cluster_sizes)}\n"
        f"Secondary Dominant Cluster: {secondary_dominant_cluster}\n"
        f"Primary ATR (Current): {primary_volatility[-1]:.2f}\n"
        f"Secondary ATR (Current): {secondary_volatility[-1]:.2f}\n"
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



def plot_signals_with_markers(highs, lows, closes, directions, upper_band, lower_band, signals):
    fig = go.Figure()

    # Plot close prices
    fig.add_trace(go.Scatter(x=list(range(len(closes))), y=closes, mode='lines', name='Close Price', line=dict(color='blue', width=2)))

    # Plot upper and lower bands
    fig.add_trace(go.Scatter(x=list(range(len(upper_band))), y=upper_band, mode='lines', name='Upper Band', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=list(range(len(lower_band))), y=lower_band, mode='lines', name='Lower Band', line=dict(color='red', dash='dash')))

    # Add direction-based markers
    for i, (price, direction) in enumerate(zip(closes, directions)):
        color = 'green' if direction == 1 else 'red'
        fig.add_trace(go.Scatter(x=[i], y=[price], mode='markers', marker=dict(color=color, size=8), name='Direction'))

    # Add Buy/Sell signals
    for i, signal in signals.items():
        text = "Buy" if signal == "buy" else "Sell"
        color = 'green' if signal == "buy" else 'red'
        fig.add_trace(go.Scatter(x=[i], y=[closes[i]], mode='markers+text',
                                 marker=dict(size=10, color=color),
                                 text=text, textposition="top center"))

    # Layout improvements
    fig.update_layout(
        title="Trading Signals with SuperTrend Logic",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        legend=dict(orientation="h"),
        template="plotly_white"
    )
    return fig

# Update the dashboard to include combined logging data
@dash_app.callback(
    [Output('primary-chart', 'figure'),
     Output('secondary-chart', 'figure'),
     Output('metrics-table', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_dashboard_callback(n):
    try:
        # Create primary chart
        fig_primary = go.Figure()
        fig_primary.add_trace(go.Scatter(y=close[-10:], mode='lines', name='Close Price', line=dict(color='blue')))
        fig_primary.add_trace(go.Scatter(y=primary_upper_band[-10:], mode='lines', name='Primary Upper Band', line=dict(color='green', dash='dash')))
        fig_primary.add_trace(go.Scatter(y=primary_lower_band[-10:], mode='lines', name='Primary Lower Band', line=dict(color='red', dash='dash')))

        # Create secondary chart
        fig_secondary = plot_signals_with_markers(
            high, low, close, secondary_direction, secondary_upper_band, secondary_lower_band, buy_sell_signals
        )

        # Create table rows for metrics
        metrics = [
            ["Price", f"{last_price:.2f}" if last_price else "N/A"],
            ["Primary Clustering Centroids", ", ".join(f"{x:.2f}" for x in primary_centroids)],
            ["Primary Cluster Sizes", ", ".join(str(size) for size in primary_cluster_sizes)],
            ["Primary Dominant Cluster", f"{primary_dominant_cluster}"],
            ["Secondary Clustering Centroids", ", ".join(f"{x:.2f}" for x in secondary_centroids)],
            ["Secondary Cluster Sizes", ", ".join(str(size) for size in secondary_cluster_sizes)],
            ["Secondary Dominant Cluster", f"{secondary_dominant_cluster}"],
            ["Primary ATR", f"{primary_volatility[-1]:.2f}"],
            ["Secondary ATR", f"{secondary_volatility[-1]:.2f}"]
        ]
        rows = [html.Tr([html.Th(metric[0]), html.Td(metric[1])]) for metric in metrics]

        return fig_primary, fig_secondary, rows

    except Exception as e:
        logging.error(f"Error updating dashboard: {e}", exc_info=True)
        return go.Figure(), go.Figure(), []


# WebSocket Handler
def on_message(msg):
    global last_price, high, low, close, primary_volatility, secondary_volatility
    global last_secondary_directions, secondary_direction  # Ensure these are global

    if 'k' not in msg:
        return

    candle = msg['k']
    last_price = float(candle['c'])
    high.append(float(candle['h']))
    low.append(float(candle['l']))
    close.append(float(candle['c']))

    # Limit the length of historical data for high, low, close
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

    # Update secondary direction history
    if secondary_direction is not None:
        last_secondary_directions.append(secondary_direction)

        # Maintain only the last 10 cycles
        if len(last_secondary_directions) > 10:
            last_secondary_directions.pop(0)



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
                    primary_centroids, _, primary_assigned_centroid, primary_cluster_sizes, primary_dominant_cluster = cluster_volatility(primary_volatility, n_clusters=3)
                    secondary_centroids, _, secondary_assigned_centroid, secondary_cluster_sizes, secondary_dominant_cluster = cluster_volatility(secondary_volatility, n_clusters=3)

                    # Execute trading logic
                    primary_direction, secondary_direction = calculate_and_execute(
                        last_price, primary_direction, secondary_direction, 
                        primary_cluster_sizes, secondary_cluster_sizes, 
                        primary_dominant_cluster, secondary_dominant_cluster
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
    primary_centroids, _, primary_assigned_centroid, _, primary_dominant_cluster = cluster_volatility(primary_volatility, n_clusters=3)
    secondary_centroids, _, secondary_assigned_centroid, _, secondary_dominant_cluster = cluster_volatility(secondary_volatility, n_clusters=3)


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
