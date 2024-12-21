import os
import sys
import time
import logging
import threading
from datetime import datetime

from binance import ThreadedWebsocketManager
from binance.client import Client
from alpaca_trade_api import REST as AlpacaREST
from sklearn.cluster import KMeans
import numpy as np

from logging.handlers import RotatingFileHandler

##########################################
# CONFIGURATION & LOGGING
##########################################
print("Main script start: Initializing logging...")

# Define log directory and ensure it exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define log filename with timestamp to avoid overwriting
log_filename = os.path.join(log_dir, f"trading_bot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs during debugging

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Adjust as needed

# Create a rotating file handler to prevent large log files
file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5)  # 5MB per file, keep last 5
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Logging initialized. Logs will be saved to console and file.")
logger.info(f"Log file: {log_filename}")

# Fetch API keys from environment variables
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Validate that all necessary API keys are present
missing_keys = []
if not BINANCE_API_KEY:
    missing_keys.append("BINANCE_API_KEY")
if not BINANCE_SECRET_KEY:
    missing_keys.append("BINANCE_SECRET_KEY")
if not ALPACA_API_KEY:
    missing_keys.append("ALPACA_API_KEY")
if not ALPACA_SECRET_KEY:
    missing_keys.append("ALPACA_SECRET_KEY")

if missing_keys:
    logger.error(f"Missing environment variables: {', '.join(missing_keys)}. Exiting.")
    sys.exit(1)

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading URL
SYMBOL_ALPACA = "BTCUSD"
QTY = 0.001  # Adjust as per your trading preferences

# Binance symbol & timeframe
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"

# Strategy / logic parameters
ATR_LEN = 10
PRIMARY_FACTOR = 3.0
SECONDARY_FACTOR = 8.0
TRAINING_DATA_PERIOD = 90  # Must match the number of ATRs available
MAX_CANDLES = 200  # Increased to fetch sufficient data for both SuperTrends
PRIMARY_TRAINING_PERIOD = 120  # Longer training period for Primary
SECONDARY_TRAINING_PERIOD = 60  # Shorter training period for Secondary

# Heartbeat & signal intervals
HEARTBEAT_INTERVAL = 30  # seconds
SIGNAL_CHECK_INTERVAL = 1  # seconds

# Globals
alpaca_api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# Data arrays
time_array = []
high_array = []
low_array = []
close_array = []
atr_array = []

# Cluster assignments for Primary and Secondary SuperTrends
cluster_assignments_primary = []
cluster_assignments_secondary = []

# Cluster centroids for Primary and Secondary SuperTrends
hv_new_primary = None
mv_new_primary = None
lv_new_primary = None

hv_new_secondary = None
mv_new_secondary = None
lv_new_secondary = None

# SuperTrend indicators
primary_supertrend = []
primary_direction = []
primary_upperBand = []
primary_lowerBand = []

secondary_supertrend = []
secondary_direction = []
secondary_upperBand = []
secondary_lowerBand = []

# For pattern detection
last_secondary_directions = []

# Position tracking
in_position = False
position_side = None
entry_price = None

buy_signals = []
sell_signals = []

# Heartbeat tracking
last_heartbeat_time = 0

# Threading lock for thread-safe operations
lock = threading.Lock()

print("Global variables initialized. (Running as a background worker or standalone script.)")

##########################################
# HELPER FUNCTIONS
##########################################
def wilder_smoothing(values, period):
    """Applies Wilder's smoothing technique to a list of values."""
    result = [None] * len(values)
    if len(values) < period:
        return result

    # Start smoothing after the first 'period' values
    initial_sum = sum(values[:period])
    result[period - 1] = initial_sum / period

    for i in range(period, len(values)):
        if values[i] is None or result[i - 1] is None:
            result[i] = None
        else:
            result[i] = ((result[i - 1] * (period - 1)) + values[i]) / period

    return result

def compute_atr(highs, lows, closes, period):
    """Computes the Average True Range (ATR) for given high, low, and close prices."""
    tr_list = []

    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        tr_list.append(tr)

    # Apply Wilder's smoothing (similar to Pine Script's ATR)
    atr = [None] * len(tr_list)
    if len(tr_list) >= period:
        atr[period - 1] = sum(tr_list[:period]) / period
        for i in range(period, len(tr_list)):
            if atr[i - 1] is None:
                atr[i] = tr_list[i]
            else:
                atr[i] = (atr[i - 1] * (period - 1) + tr_list[i]) / period
    else:
        # Not enough data to compute ATR
        pass

    return atr

def run_kmeans(vol_data, n_clusters=3, random_state=0):
    """
    Runs K-Means clustering on the volatility data and returns sorted centroids,
    labels, and cluster sizes.

    Parameters:
    - vol_data (list or array): The volatility data to cluster.
    - n_clusters (int): Number of clusters.
    - random_state (int): Determines random number generation for centroid initialization.

    Returns:
    - hv (float): High Volatility centroid.
    - mv (float): Medium Volatility centroid.
    - lv (float): Low Volatility centroid.
    - sorted_labels (list): Cluster assignments for each data point.
    - cluster_sizes (dict): Number of data points in each cluster.
    """
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(np.array(vol_data).reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        # Sort centroids and re-map labels accordingly
        sorted_indices = np.argsort(centroids)
        sorted_centroids = centroids[sorted_indices]
        label_mapping = {original: sorted_idx for sorted_idx, original in enumerate(sorted_indices)}
        sorted_labels = [label_mapping[label] for label in labels]

        # Calculate cluster sizes
        cluster_sizes = {
            'Low': sorted_labels.count(0),
            'Medium': sorted_labels.count(1),
            'High': sorted_labels.count(2)
        }

        hv = sorted_centroids[2]  # High Volatility
        mv = sorted_centroids[1]  # Medium Volatility
        lv = sorted_centroids[0]  # Low Volatility

        logger.debug(
            f"K-Means Centroids: HV={hv:.4f}, MV={mv:.4f}, LV={lv:.4f} | "
            f"Counts: HV={cluster_sizes['High']}, MV={cluster_sizes['Medium']}, LV={cluster_sizes['Low']}"
        )

        return hv, mv, lv, sorted_labels, cluster_sizes

    except Exception as e:
        logger.error(f"K-Means failed: {e}", exc_info=True)
        return None, None, None, [], {}

def compute_supertrend(i, factor, assigned_atr, st_array, dir_array, ub_array, lb_array):
    with lock:
        length = len(close_array)
        if i < 0 or i >= length:
            logger.error(f"compute_supertrend called with invalid index: {i}")
            return

        if assigned_atr is None:
            st_array[i] = st_array[i-1] if i > 0 else None
            dir_array[i] = dir_array[i-1] if i > 0 else 1
            ub_array[i] = ub_array[i-1] if i > 0 else None
            lb_array[i] = lb_array[i-1] if i > 0 else None
            return

        hl2 = (high_array[i] + low_array[i]) / 2.0
        upBand = hl2 + factor * assigned_atr
        downBand = hl2 - factor * assigned_atr

        if i == 0:
            dir_array[i] = 1  # Initial direction
            ub_array[i] = upBand
            lb_array[i] = downBand
            st_array[i] = upBand
            return

        # Ensure previous index is within bounds
        if i - 1 < 0 or i - 1 >= len(dir_array):
            logger.error(f"SuperTrend computation at index {i} has invalid previous index {i-1}")
            return

        prevDir = dir_array[i - 1]
        prevUB = ub_array[i - 1] if ub_array[i - 1] is not None else upBand
        prevLB = lb_array[i - 1] if lb_array[i - 1] is not None else downBand

        # Adjust bands based on previous bands
        if downBand > prevLB or close_array[i - 1] < prevLB:
            downBand = downBand
        else:
            downBand = prevLB

        if upBand < prevUB or close_array[i - 1] > prevUB:
            upBand = upBand
        else:
            upBand = prevUB

        # Determine direction based on standard SuperTrend logic
        if prevDir == 1:  # Previously Bullish
            if close_array[i] < downBand:
                dir_array[i] = -1  # Change to Bearish
            else:
                dir_array[i] = 1   # Continue Bullish
        elif prevDir == -1:  # Previously Bearish
            if close_array[i] > upBand:
                dir_array[i] = 1   # Change to Bullish
            else:
                dir_array[i] = -1  # Continue Bearish
        else:
            # Default to previous direction if undefined
            dir_array[i] = prevDir

        # Set SuperTrend value based on current direction
        st_array[i] = downBand if dir_array[i] == 1 else upBand
        ub_array[i] = upBand
        lb_array[i] = downBand

        logger.debug(
            f"SuperTrend computed for index {i}: Dir={dir_array[i]}, ST={st_array[i]}, UB={ub_array[i]}, LB={lb_array[i]}"
        )

def execute_trade(side, qty, symbol, stop_loss=None, take_profit=None):
    """Executes a trade via Alpaca API."""
    logger.info(f"Executing {side.upper()} {qty} {symbol} (SL={stop_loss}, TP={take_profit})")
    try:
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
            stop_loss={"stop_price": stop_loss} if stop_loss else None,
            take_profit={"limit_price": take_profit} if take_profit else None
        )
        logger.info(f"Order submitted: {order}")
        with lock:
            if side.lower() == "buy":
                buy_signals.append((len(close_array)-1, close_array[-1]))
            elif side.lower() == "sell":
                sell_signals.append((len(close_array)-1, close_array[-1]))
    except Exception as e:
        logger.error(f"Alpaca order failed: {e}", exc_info=True)

##########################################
# INITIALIZE HISTORICAL DATA
##########################################
def initialize_historical_data():
    """
    Fetches historical data, computes ATR, runs K-Means for both SuperTrends,
    assigns clusters, and computes SuperTrend indicators.
    """
    logger.info("Initializing historical data from Binance REST API...")

    try:
        bars = binance_client.get_klines(
            symbol=BINANCE_SYMBOL,
            interval=BINANCE_INTERVAL,
            limit=MAX_CANDLES  # Increased to 200 to ensure enough data
        )
        for bar in bars:
            open_time = bar[0]
            high_val = float(bar[2])
            low_val = float(bar[3])
            close_val = float(bar[4])

            time_array.append(open_time)
            high_array.append(high_val)
            low_array.append(low_val)
            close_array.append(close_val)

        logger.info(f"Fetched {len(close_array)} historical bars.")

        # Compute ATR
        atr = compute_atr(high_array, low_array, close_array, ATR_LEN)
        atr_array.clear()
        atr_array.extend(atr)
        logger.debug(f"Computed ATR: {atr_array[-5:]}")

        # Initialize cluster assignments
        cluster_assignments_primary.clear()
        cluster_assignments_secondary.clear()
        cluster_assignments_primary.extend([None] * len(close_array))
        cluster_assignments_secondary.extend([None] * len(close_array))

        # Initialize SuperTrend arrays
        primary_supertrend.clear()
        primary_direction.clear()
        primary_upperBand.clear()
        primary_lowerBand.clear()

        secondary_supertrend.clear()
        secondary_direction.clear()
        secondary_upperBand.clear()
        secondary_lowerBand.clear()

        # Pre-populate SuperTrend arrays with None
        for _ in close_array:
            primary_supertrend.append(None)
            primary_direction.append(None)
            primary_upperBand.append(None)
            primary_lowerBand.append(None)

            secondary_supertrend.append(None)
            secondary_direction.append(None)
            secondary_upperBand.append(None)
            secondary_lowerBand.append(None)

        logger.info("SuperTrend arrays initialized.")
        ##########################################
        # RUN K-MEANS FOR PRIMARY SUPERTREND
        ##########################################
        if len(atr_array) >= PRIMARY_TRAINING_PERIOD:
            vol_data_primary = atr_array[-PRIMARY_TRAINING_PERIOD:]
            if len(vol_data_primary) == PRIMARY_TRAINING_PERIOD:
                logger.info("Running K-Means for Primary SuperTrend on historical ATR data...")
                hv_primary, mv_primary, lv_primary, labels_primary, cluster_sizes_primary = run_kmeans(
                    vol_data_primary,
                    n_clusters=3,
                    random_state=0  # Primary uses 0
                )
                if hv_primary and mv_primary and lv_primary:
                    global hv_new_primary, mv_new_primary, lv_new_primary
                    hv_new_primary, mv_new_primary, lv_new_primary = hv_primary, mv_primary, lv_primary
                    logger.info(
                        f"K-Means for Primary SuperTrend: HV={hv_new_primary:.4f}, MV={mv_new_primary:.4f}, LV={lv_new_primary:.4f} | "
                        f"Counts: HV={cluster_sizes_primary['High']}, MV={cluster_sizes_primary['Medium']}, LV={cluster_sizes_primary['Low']}"
                    )

                    # Assign clusters and compute SuperTrend for Primary SuperTrend
                    start_idx = len(close_array) - PRIMARY_TRAINING_PERIOD
                    for idx in range(start_idx, len(close_array)):
                        vol = atr_array[idx]
                        distances = [abs(vol - hv_new_primary), abs(vol - mv_new_primary), abs(vol - lv_new_primary)]
                        c_idx_primary = distances.index(min(distances))
                        cluster_assignments_primary[idx] = c_idx_primary
                        assigned_centroid_primary = [lv_new_primary, mv_new_primary, hv_new_primary][c_idx_primary]
                        primary_direction[idx] = 1 if close_array[idx] > (high_array[idx] + low_array[idx]) / 2 + PRIMARY_FACTOR * assigned_centroid_primary else -1
                        compute_supertrend(
                            i=idx,
                            factor=PRIMARY_FACTOR,
                            assigned_atr=assigned_centroid_primary,
                            st_array=primary_supertrend,
                            dir_array=primary_direction,
                            ub_array=primary_upperBand,
                            lb_array=primary_lowerBand
                        )
                        logger.debug(f"Primary SuperTrend updated for index {idx}")
                else:
                    logger.warning("K-Means for Primary SuperTrend returned invalid centroids.")
            else:
                logger.warning("Insufficient ATR data for Primary SuperTrend K-Means.")
        else:
            logger.warning("Not enough ATR data to run K-Means for Primary SuperTrend.")

        ##########################################
        # RUN K-MEANS FOR SECONDARY SUPERTREND
        ##########################################
        if len(atr_array) >= (PRIMARY_TRAINING_PERIOD + SECONDARY_TRAINING_PERIOD):
            vol_data_secondary = atr_array[-(PRIMARY_TRAINING_PERIOD + SECONDARY_TRAINING_PERIOD):-PRIMARY_TRAINING_PERIOD]
            if len(vol_data_secondary) == SECONDARY_TRAINING_PERIOD:
                logger.info("Running K-Means for Secondary SuperTrend on historical ATR data...")
                hv_secondary, mv_secondary, lv_secondary, labels_secondary, cluster_sizes_secondary = run_kmeans(
                    vol_data_secondary,
                    n_clusters=3,
                    random_state=1  # Secondary uses 1
                )
                if hv_secondary and mv_secondary and lv_secondary:
                    global hv_new_secondary, mv_new_secondary, lv_new_secondary
                    hv_new_secondary, mv_new_secondary, lv_new_secondary = hv_secondary, mv_secondary, lv_secondary
                    logger.info(
                        f"K-Means for Secondary SuperTrend: HV={hv_new_secondary:.4f}, MV={mv_new_secondary:.4f}, LV={lv_new_secondary:.4f} | "
                        f"Counts: HV={cluster_sizes_secondary['High']}, MV={cluster_sizes_secondary['Medium']}, LV={cluster_sizes_secondary['Low']}"
                    )

                    # Assign clusters and compute SuperTrend for Secondary SuperTrend
                    start_idx = len(close_array) - PRIMARY_TRAINING_PERIOD
                    for idx in range(start_idx, len(close_array)):
                        vol = atr_array[idx]
                        distances = [abs(vol - hv_new_secondary), abs(vol - mv_new_secondary), abs(vol - lv_new_secondary)]
                        c_idx_secondary = distances.index(min(distances))
                        cluster_assignments_secondary[idx] = c_idx_secondary
                        assigned_centroid_secondary = [lv_new_secondary, mv_new_secondary, hv_new_secondary][c_idx_secondary]
                        secondary_direction[idx] = 1 if close_array[idx] > (high_array[idx] + low_array[idx]) / 2 + SECONDARY_FACTOR * assigned_centroid_secondary else -1
                        compute_supertrend(
                            i=idx,
                            factor=SECONDARY_FACTOR,
                            assigned_atr=assigned_centroid_secondary,
                            st_array=secondary_supertrend,
                            dir_array=secondary_direction,
                            ub_array=secondary_upperBand,
                            lb_array=secondary_lowerBand
                        )
                        logger.debug(f"Secondary SuperTrend updated for index {idx}")
                else:
                    logger.warning("K-Means for Secondary SuperTrend returned invalid centroids.")
            else:
                logger.warning("Insufficient ATR data for Secondary SuperTrend K-Means.")
        else:
            logger.warning("Not enough ATR data to run K-Means for Secondary SuperTrend.")

        logger.info("Historical data initialization complete.")
    except Exception as e:
        logger.error(f"Historical data initialization failed: {e}", exc_info=True)
        
##########################################
# THREADS: HEARTBEAT & SIGNAL CHECK
##########################################
def heartbeat_logging():
    global last_heartbeat_time
    logger.info("Heartbeat thread started...")

    while True:
        try:
            now = time.time()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                with lock:
                    if len(close_array) == 0:
                        logger.info("No data yet.")
                    else:
                        i = len(close_array) - 1
                        p_dir = primary_direction[i] if i < len(primary_direction) else None
                        s_dir = secondary_direction[i] if i < len(secondary_direction) else None
                        c_idx_primary = cluster_assignments_primary[i] if i < len(cluster_assignments_primary) else None
                        c_idx_secondary = cluster_assignments_secondary[i] if i < len(cluster_assignments_secondary) else None

                        assigned_centroid_primary = None
                        if c_idx_primary is not None and hv_new_primary is not None and mv_new_primary is not None and lv_new_primary is not None:
                            assigned_centroid_primary = [lv_new_primary, mv_new_primary, hv_new_primary][c_idx_primary]

                        assigned_centroid_secondary = None
                        if c_idx_secondary is not None and hv_new_secondary is not None and mv_new_secondary is not None and lv_new_secondary is not None:
                            assigned_centroid_secondary = [lv_new_secondary, mv_new_secondary, hv_new_secondary][c_idx_secondary]

                        pri_st = primary_supertrend[i] if i < len(primary_supertrend) else None
                        sec_st = secondary_supertrend[i] if i < len(secondary_supertrend) else None

                        # Calculate ATR values
                        primary_atr_val = assigned_centroid_primary * PRIMARY_FACTOR if assigned_centroid_primary else 'N/A'
                        secondary_atr_val = assigned_centroid_secondary * SECONDARY_FACTOR if assigned_centroid_secondary else 'N/A'

                        msg = "\n=== Heartbeat ===\n"
                        msg += f"Last Price: {close_array[i]:.2f}\n"
                        msg += f"Primary Dir: {p_dir}\n"
                        msg += f"Secondary Dir: {s_dir}\n"
                        msg += f"Primary Cluster: {c_idx_primary if c_idx_primary is not None else 'None'} (0=Low,1=Med,2=High)\n"
                        msg += f"Primary Cluster Sizes: Low={cluster_assignments_primary.count(0)}, Med={cluster_assignments_primary.count(1)}, High={cluster_assignments_primary.count(2)}\n"
                        msg += f"Secondary Cluster: {c_idx_secondary if c_idx_secondary is not None else 'None'} (0=Low,1=Med,2=High)\n"      
                        msg += f"Secondary Cluster Sizes: Low={cluster_assignments_secondary.count(0)}, Med={cluster_assignments_secondary.count(1)}, High={cluster_assignments_secondary.count(2)}\n"
                        msg += f"Primary Base ATR (Assigned Centroid): {assigned_centroid_primary if assigned_centroid_primary else 'N/A'}\n"
                        msg += f"Secondary Base ATR (Assigned Centroid): {assigned_centroid_secondary if assigned_centroid_secondary else 'N/A'}\n"
                        msg += f"Primary ATR: {primary_atr_val}\n"
                        msg += f"Secondary ATR: {secondary_atr_val}\n"
                        msg += f"PriST: {pri_st if pri_st else 'N/A'}\n"
                        msg += f"SecST: {sec_st if sec_st else 'N/A'}\n"
                        msg += f"Primary Upper Band: {primary_upperBand[i] if i < len(primary_upperBand) else 'N/A'}\n"
                        msg += f"Primary Lower Band: {primary_lowerBand[i] if i < len(primary_lowerBand) else 'N/A'}\n"
                        msg += f"Secondary Upper Band: {secondary_upperBand[i] if i < len(secondary_upperBand) else 'N/A'}\n"
                        msg += f"Secondary Lower Band: {secondary_lowerBand[i] if i < len(secondary_lowerBand) else 'N/A'}\n"
                        msg += f"In Position: {in_position} ({position_side})\n"
                        msg += f"Entry Price: {entry_price}\n"
                        msg += f"Primary Cluster Sizes: Low={cluster_assignments_primary.count(0)}, Med={cluster_assignments_primary.count(1)}, High={cluster_assignments_primary.count(2)}\n"
                        msg += f"Secondary Cluster Sizes: Low={cluster_assignments_secondary.count(0)}, Med={cluster_assignments_secondary.count(1)}, High={cluster_assignments_secondary.count(2)}\n"
                        msg += "=============="
                        logger.info(msg)

                    last_heartbeat_time = now
        except Exception as e:
            logger.error(f"Error in heartbeat: {e}", exc_info=True)
        time.sleep(1)  # Check every second

def check_signals():
    global in_position, position_side, entry_price
    logger.info("Signal checking thread started...")

    while True:
        try:
            with lock:
                length = len(close_array)
                if length > 0:
                    i = length - 1
                    p_dir = primary_direction[i] if i < len(primary_direction) else None
                    s_dir = secondary_direction[i] if i < len(secondary_direction) else None
                    c_idx_primary = cluster_assignments_primary[i] if i < len(cluster_assignments_primary) else None
                    c_idx_secondary = cluster_assignments_secondary[i] if i < len(cluster_assignments_secondary) else None

                    if (p_dir is not None
                        and s_dir is not None
                        and c_idx_primary is not None
                        and c_idx_secondary is not None
                        and len(last_secondary_directions) >= 3):

                        recent_3 = last_secondary_directions[-3:]
                        bullish_bearish_bullish = (recent_3 == [1, -1, 1])
                        bearish_bearish_bearish = (recent_3 == [-1, 1, -1])
                        current_price = close_array[i]

                        # LONG ENTRY
                        if (not in_position
                            and bullish_bearish_bullish
                            and p_dir == 1
                            and c_idx_primary == 2  # High Volatility for Primary
                            and c_idx_secondary == 2):  # High Volatility for Secondary
                            sl = low_array[i]
                            dist = current_price - sl
                            tp = current_price + (1.5 * dist)
                            logger.info("Pullback BUY triggered!")
                            execute_trade(
                                side="buy",
                                qty=QTY,
                                symbol=SYMBOL_ALPACA,
                                stop_loss=round(sl, 2),
                                take_profit=round(tp, 2)
                            )
                            in_position = True
                            position_side = "long"
                            entry_price = current_price

                        # SHORT ENTRY
                        if (not in_position
                            and bearish_bearish_bearish
                            and p_dir == -1
                            and c_idx_primary == 2  # High Volatility for Primary
                            and c_idx_secondary == 2):  # High Volatility for Secondary
                            sl = high_array[i]
                            dist = sl - current_price
                            tp = current_price - (1.5 * dist)
                            logger.info("Pullback SHORT triggered!")
                            execute_trade(
                                side="sell",
                                qty=QTY,
                                symbol=SYMBOL_ALPACA,
                                stop_loss=round(sl, 2),
                                take_profit=round(tp, 2)
                            )
                            in_position = True
                            position_side = "short"
                            entry_price = current_price

                        # EXIT LOGIC for LONG
                        if in_position and position_side == "long" and p_dir == -1:
                            logger.info("Primary turned bearish. Closing LONG.")
                            execute_trade("sell", QTY, SYMBOL_ALPACA)
                            in_position = False
                            position_side = None
                            entry_price = None

                        # EXIT LOGIC for SHORT
                        if in_position and position_side == "short" and p_dir == 1:
                            logger.info("Primary turned bullish. Closing SHORT.")
                            execute_trade("buy", QTY, SYMBOL_ALPACA)
                            in_position = False
                            position_side = None
                            entry_price = None

                    # Update last_secondary_directions based on current secondary direction
                    if s_dir is not None:
                        last_secondary_directions.append(s_dir)
                        if len(last_secondary_directions) > 10:
                            last_secondary_directions.pop(0)

        except Exception as e:
            logger.error(f"Error in check_signals: {e}", exc_info=True)
        time.sleep(SIGNAL_CHECK_INTERVAL)  # Check every second

##########################################
# WEBSOCKET CALLBACK
##########################################
def on_message_candle(msg):
    """
    Callback function for Binance WebSocket kline messages.
    """
    try:
        if 'k' not in msg:
            logger.debug("Received message without 'k' key. Ignoring.")
            return

        k = msg['k']
        is_final = k['x']
        close_price = float(k['c'])
        high_price = float(k['h'])
        low_price = float(k['l'])
        open_time = k['t']

        if is_final:
            logger.info("on_message_candle: Candle is final. Appending new bar...")

            with lock:
                # Append new candle data
                time_array.append(open_time)
                high_array.append(high_price)
                low_array.append(low_price)
                close_array.append(close_price)

                # Trim arrays to maintain MAX_CANDLES
                while len(time_array) > MAX_CANDLES:
                    time_array.pop(0)
                    high_array.pop(0)
                    low_array.pop(0)
                    close_array.pop(0)

                # Compute ATR
                if len(close_array) >= ATR_LEN:
                    new_atr = compute_atr(high_array, low_array, close_array, ATR_LEN)
                    atr_array.clear()
                    atr_array.extend(new_atr)
                    logger.debug(f"Computed ATR: {atr_array[-5:]}")
                else:
                    atr_array.clear()
                    atr_array.extend([None] * len(close_array))  # Use None for insufficient data
                    logger.debug("Not enough data to compute ATR.")

                # Trim ATR array to match close_array length
                while len(atr_array) > len(close_array):
                    atr_array.pop(0)

                # Sync cluster_assignments_primary and cluster_assignments_secondary with close_array
                while len(cluster_assignments_primary) < len(close_array):
                    cluster_assignments_primary.append(None)
                while len(cluster_assignments_primary) > len(close_array):
                    cluster_assignments_primary.pop(0)

                while len(cluster_assignments_secondary) < len(close_array):
                    cluster_assignments_secondary.append(None)
                while len(cluster_assignments_secondary) > len(close_array):
                    cluster_assignments_secondary.pop(0)

                # Ensure SuperTrend arrays are synchronized
                def fix_arrays(st, di, ub, lb):
                    needed_len = len(close_array)
                    while len(st) < needed_len:
                        st.append(None)
                    while len(st) > needed_len:
                        st.pop(0)
                    while len(di) < needed_len:
                        di.append(None)
                    while len(di) > needed_len:
                        di.pop(0)
                    while len(ub) < needed_len:
                        ub.append(None)
                    while len(ub) > needed_len:
                        ub.pop(0)
                    while len(lb) < needed_len:
                        lb.append(None)
                    while len(lb) > needed_len:
                        lb.pop(0)

                fix_arrays(primary_supertrend, primary_direction,
                           primary_upperBand, primary_lowerBand)
                fix_arrays(secondary_supertrend, secondary_direction,
                           secondary_upperBand, secondary_lowerBand)

                data_count = len(close_array)

                # Assign cluster and compute SuperTrend for Primary SuperTrend
                if hv_new_primary and mv_new_primary and lv_new_primary and atr_array[-1] is not None:
                    vol_primary = atr_array[-1]
                    distances_primary = [abs(vol_primary - hv_new_primary), abs(vol_primary - mv_new_primary), abs(vol_primary - lv_new_primary)]
                    c_idx_primary = distances_primary.index(min(distances_primary))
                    cluster_assignments_primary[-1] = c_idx_primary
                    assigned_centroid_primary = [lv_new_primary, mv_new_primary, hv_new_primary][c_idx_primary]
                    primary_direction[-1] = 1 if close_price > (high_array[-1] + low_array[-1]) / 2 + PRIMARY_FACTOR * assigned_centroid_primary else -1
                    compute_supertrend(
                        i=data_count - 1,
                        factor=PRIMARY_FACTOR,
                        assigned_atr=assigned_centroid_primary,
                        st_array=primary_supertrend,
                        dir_array=primary_direction,
                        ub_array=primary_upperBand,
                        lb_array=primary_lowerBand
                    )
                    logger.debug(f"Primary SuperTrend updated for index {data_count - 1}")

                # Assign cluster and compute SuperTrend for Secondary SuperTrend
                if hv_new_secondary and mv_new_secondary and lv_new_secondary and atr_array[-1] is not None:
                    vol_secondary = atr_array[-1]
                    distances_secondary = [abs(vol_secondary - hv_new_secondary), abs(vol_secondary - mv_new_secondary), abs(vol_secondary - lv_new_secondary)]
                    c_idx_secondary = distances_secondary.index(min(distances_secondary))
                    cluster_assignments_secondary[-1] = c_idx_secondary
                    assigned_centroid_secondary = [lv_new_secondary, mv_new_secondary, hv_new_secondary][c_idx_secondary]
                    secondary_direction[-1] = 1 if close_price > (high_array[-1] + low_array[-1]) / 2 + SECONDARY_FACTOR * assigned_centroid_secondary else -1
                    compute_supertrend(
                        i=data_count - 1,
                        factor=SECONDARY_FACTOR,
                        assigned_atr=assigned_centroid_secondary,
                        st_array=secondary_supertrend,
                        dir_array=secondary_direction,
                        ub_array=secondary_upperBand,
                        lb_array=secondary_lowerBand
                    )
                    logger.debug(f"Secondary SuperTrend updated for index {data_count - 1}")
    except Exception as e:
        logger.error(f"SuperTrend calculation missed : {e}", exc_info=True)
    ##########################################
    # START THE BINANCE WEBSOCKET
    ##########################################
def start_binance_websocket():
    """
    Starts the Binance WebSocket for receiving real-time candle data.
    """
    while True:
        try:
            logger.info("Starting Binance WebSocket (background worker).")
            twm = ThreadedWebsocketManager(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY
            )
            twm.start()

            # Start the kline socket for the specified symbol/interval
            logger.info(f"Subscribing to {BINANCE_SYMBOL} {BINANCE_INTERVAL} kline stream.")
            twm.start_kline_socket(
                callback=on_message_candle,
                symbol=BINANCE_SYMBOL,
                interval=BINANCE_INTERVAL
            )

            # Keep the WebSocket alive
            twm.join()

        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in 30 seconds...", exc_info=True)
            time.sleep(30)  # Wait before reconnecting

##########################################
# MAIN
##########################################
def main():
    logger.info("Fetching initial historical data for warmup...")
    initialize_historical_data()  # Populate arrays from Binance REST

    logger.info("Main function start: Starting threads...")
    # Start heartbeat thread
    hb_thread = threading.Thread(target=heartbeat_logging, daemon=True)
    hb_thread.start()

    # Start signal checking thread
    signal_thread = threading.Thread(target=check_signals, daemon=True)
    signal_thread.start()

    # Start the Binance WebSocket in the main thread (blocking)
    start_binance_websocket()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)




