import os
import sys
import time
import math
import statistics
import logging
import threading
import pandas as pd
from flask import Flask, send_file
# 3rd-party libraries
from binance import ThreadedWebsocketManager
from binance.client import Client
from alpaca_trade_api import REST as AlpacaREST
from sklearn.cluster import KMeans
import numpy as np

# ============== CONFIGURATION ==============
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.log"),  # Save logs to file
        logging.StreamHandler(sys.stdout)              # Output logs to console
    ]
)

# Environment variables / credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

# Alpaca endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper trading
SYMBOL_ALPACA = "BTCUSD"  # e.g. "BTCUSD" on Alpaca
QTY = 0.001               # Example trade size

# Binance symbol & timeframe
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"  # 1-minute bars

# Strategy / logic parameters
ATR_LEN = 15
PRIMARY_FACTOR = 8.0      # SuperTrend factor for primary
SECONDARY_FACTOR = 3.0    # SuperTrend factor for secondary
TRAINING_DATA_PERIOD = 100  # Increased from 1 to 3
HIGHVOL_PERCENTILE = 0.75
MIDVOL_PERCENTILE = 0.5
LOWVOL_PERCENTILE = 0.25

# Heartbeat intervals
HEARTBEAT_INTERVAL = 61   # seconds
SIGNAL_CHECK_INTERVAL = 17 # check signals every 1 second

# Keep only the last MAX_CANDLES in memory
MAX_CANDLES = 200

# Number of candles to determine pullback pattern
MAX_PULLBACK_CANDLES =30

# only once?
CLUSTER_RUN_ONCE = False       # If True, run K-Means only once after we have the training period data



# ============== ALPACA & BINANCE CLIENTS ==============
alpaca_api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# ============== GLOBAL DATA STRUCTURES ==============
time_array = []
high_array = []
low_array = []
close_array = []
atr_array = []
cluster_assignments = []

# K-Means centroids for High/Medium/Low once stable
hv_new = None
mv_new = None
lv_new = None

# Two signals
primary_supertrend = []
primary_direction = []
primary_upperBand = []
primary_lowerBand = []

secondary_supertrend = []
secondary_direction = []
secondary_upperBand = []
secondary_lowerBand = []

# Keep track of secondary directions to detect patterns
last_secondary_directions = []

# Position management
in_position = False    # True if in a trade
position_side = None   # 'long' or 'short'
entry_price = None

# For logging
last_heartbeat_time = 0

# Flask Server
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!", 200

@app.route('/logs')
def download_logs():
    try:
        # Assuming logs are stored in the current directory
        return send_file("bot_logs.log", as_attachment=True)
    except FileNotFoundError:
        return "Log file not found.", 404

# ============== HELPER FUNCTIONS ==============
def wilder_smoothing(values, period):
    """
    ATR uses Wilder's smoothing:
      atr[i] = ((atr[i-1]*(period-1)) + tr[i]) / period
    """
    result = [None]*len(values)
    if len(values) < period:
        return result
    # Ensure no None values in the initial period
    initial_values = [v for v in values[:period] if v is not None]
    if len(initial_values) < period:
        # Not enough valid values to compute ATR
        return result
    initial = sum(initial_values) / period
    result[period-1] = initial
    for i in range(period, len(values)):
        if values[i] is None or result[i-1] is None:
            result[i] = result[i-1] if i > 0 else None
        else:
            current = ((result[i-1]*(period-1)) + values[i]) / period
            result[i] = current
    return result

def compute_atr(h_array, l_array, c_array, period):
    """Compute Average True Range (ATR)"""
    tr_list = []
    for i in range(len(c_array)):
        if i == 0:
            # Initialize with the first True Range
            tr = h_array[i] - l_array[i]
            tr_list.append(tr)
            continue
        t1 = h_array[i] - l_array[i]
        t2 = abs(h_array[i] - c_array[i-1])
        t3 = abs(l_array[i] - c_array[i-1])
        true_range = max(t1, t2, t3)
        tr_list.append(true_range)
    smooth = wilder_smoothing(tr_list, period)
    return smooth

def run_kmeans(vol_data, hv_init, mv_init, lv_init):
    """
    Replicates the K-Means style clustering from Pine.
    """
    if len(vol_data) < 3:
        logging.warning("Not enough data points for K-Means clustering.")
        return None, None, None, 0, 0, 0

    try:
        amean = [hv_init]
        bmean = [mv_init]
        cmean = [lv_init]

        def means_stable(m):
            if len(m) < 2:
                return False
            return math.isclose(m[-1], m[-2], rel_tol=1e-9, abs_tol=1e-9)

        while True:
            hv_cluster = []
            mv_cluster = []
            lv_cluster = []

            cur_a = amean[-1]
            cur_b = bmean[-1]
            cur_c = cmean[-1]

            for v in vol_data:
                da = abs(v - cur_a)
                db = abs(v - cur_b)
                dc = abs(v - cur_c)
                m_dist = min(da, db, dc)
                if m_dist == da:
                    hv_cluster.append(v)
                elif m_dist == db:
                    mv_cluster.append(v)
                else:
                    lv_cluster.append(v)

            new_a = statistics.mean(hv_cluster) if hv_cluster else cur_a
            new_b = statistics.mean(mv_cluster) if mv_cluster else cur_b
            new_c = statistics.mean(lv_cluster) if lv_cluster else cur_c

            amean.append(new_a)
            bmean.append(new_b)
            cmean.append(new_c)

            stable_a = means_stable(amean)
            stable_b = means_stable(bmean)
            stable_c = means_stable(cmean)

            if stable_a and stable_b and stable_c:
                return new_a, new_b, new_c, len(hv_cluster), len(mv_cluster), len(lv_cluster)
    except Exception as e:
        logging.error(f"K-Means clustering failed: {e}", exc_info=True)
        return None, None, None, 0, 0, 0

def compute_supertrend(i, factor, assigned_atr, st_array, dir_array, ub_array, lb_array):
    if assigned_atr is None:
        # Carry forward previous values
        st_array[i] = st_array[i-1] if i > 0 else None
        dir_array[i] = dir_array[i-1] if i > 0 else 1
        ub_array[i] = ub_array[i-1] if i > 0 else None
        lb_array[i] = lb_array[i-1] if i > 0 else None
        return

    hl2 = (high_array[i] + low_array[i]) / 2.0
    upBand = hl2 + factor * assigned_atr
    downBand = hl2 - factor * assigned_atr

    if i == 0:
        dir_array[i] = 1
        ub_array[i] = upBand
        lb_array[i] = downBand
        st_array[i] = upBand
        logging.debug(f"Initial supertrend at index {i}: dir=1, st={upBand}, ub={upBand}, lb={downBand}")
        return

    prevDir = dir_array[i-1]
    prevUB = ub_array[i-1] if ub_array[i-1] is not None else upBand
    prevLB = lb_array[i-1] if lb_array[i-1] is not None else downBand

    # Band continuity

        if (downBand > prevLB or close_array[i-1] < prevLB):
        downBand = downBand
    else:
        downBand = prevLB
    if (upBand < prevUB or close_array[i-1] > prevUB):
        upBand = upBand
    else:
        upBand = prevUB

    # Direction determination
    if prevDir == 1:
        if close_array[i] < downBand:
            newDir = -1  # Switch to Downtrend
            logging.debug(f"Index {i}: Close price {close_array[i]:.2f} < downBand {downBand:.2f} ⇒ Switch to Downtrend (-1)")
        else:
            newDir = 1   # Remain Uptrend
            logging.debug(f"Index {i}: Close price {close_array[i]:.2f} >= downBand {downBand:.2f} ⇒ Remain Uptrend (1)")
    elif prevDir == -1:
        if close_array[i] > upBand:
            newDir = 1   # Switch to Uptrend
            logging.debug(f"Index {i}: Close price {close_array[i]:.2f} > upBand {upBand:.2f} ⇒ Switch to Uptrend (1)")
        else:
            newDir = -1  # Remain Downtrend
            logging.debug(f"Index {i}: Close price {close_array[i]:.2f} <= upBand {upBand:.2f} ⇒ Remain Downtrend (-1)")

    # Assign new direction and SuperTrend
    dir_array[i] = newDir
    st_array[i] = upBand if newDir == -1 else downBand
    ub_array[i] = upBand
    lb_array[i] = downBand

    # Detect and log direction shift
    if i > 0 and newDir != prevDir:
        trend = "Uptrend" if newDir == 1 else "Downtrend"
        prev_trend = "Uptrend" if prevDir == 1 else "Downtrend"
        logging.info(
            f"Direction Shift Detected at index {i}: {prev_trend} -> {trend} | "
            f"Close: {close_array[i]:.2f} | ST: {st_array[i]:.2f} | "
            f"Upper Band: {upBand:.2f} | Lower Band: {downBand:.2f}"
        )

    # Optional: Detailed debug logging (can be disabled to reduce verbosity)
    # logging.debug(
    #     f"Supertrend at index {i}: prevDir={prevDir}, close={close_array[i]}, "
    #     f"upBand={upBand}, downBand={downBand}, newDir={newDir}, st={st_array[i]}"
    # )


# ============== ORDER EXECUTION (WITH STOP/TAKE PROFIT) ==============
def execute_trade(side, qty, symbol, stop_loss=None, take_profit=None):
    """
    Places a market order on Alpaca.
    """
    try:
        logging.info(f"Executing {side.upper()} {qty} {symbol} (SL={stop_loss}, TP={take_profit})")
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
            stop_loss={"stop_price": stop_loss} if stop_loss else None,
            take_profit={"limit_price": take_profit} if take_profit else None
        )
        logging.info(f"Order submitted: {order}")
    except Exception as e:
        logging.error(f"Alpaca order failed: {e}", exc_info=True)

# ============== LOGGING (HEARTBEAT) ==============
def heartbeat_logging():
    global last_heartbeat_time
    while True:
        now = time.time()
        if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
            if len(close_array) > 0:
                i = len(close_array)-1
                p_dir = primary_direction[i] if i < len(primary_direction) else 'N/A'
                s_dir = secondary_direction[i] if i < len(secondary_direction) else 'N/A'
                c_idx = cluster_assignments[i] if i < len(cluster_assignments) else None
                atr = atr_array[i] if i < len(atr_array) else 'N/A'
                pri_st = primary_supertrend[i] if i < len(primary_supertrend) else 'N/A'
                sec_st = secondary_supertrend[i] if i < len(secondary_supertrend) else 'N/A'

                cluster_str = f"{c_idx} (0=High,1=Med,2=Low)" if c_idx is not None else "None (0=High,1=Med,2=Low)"
                msg = "\n=== Heartbeat ===\n"
                msg += f"Last Price: {close_array[i]:.2f}\n"
                msg += f"Primary Dir: {p_dir}\n"
                msg += f"Secondary Dir: {s_dir}\n"
                msg += f"Cluster: {cluster_str}\n"
                msg += f"ATR: {atr}\n"
                msg += f"PriST: {pri_st}\n"
                msg += f"SecST: {sec_st}\n"
                msg += f"In Position: {in_position} ({position_side})\n"
                msg += f"Entry Price: {entry_price}\n"
                msg += "=============="
                logging.info(msg)
            last_heartbeat_time = now
        time.sleep(1)

# ============== SIGNAL CHECKS FOR LONG & SHORT ==============
def check_signals():
    """Check for trade signals based on SuperTrend indicators"""
    global in_position, position_side, entry_price

    while True:
        try:
            if len(close_array) == 0:
                time.sleep(SIGNAL_CHECK_INTERVAL)
                continue

            i = len(close_array)-1
            t = time_array[i]


            # Gather signals
            p_dir = primary_direction[i]
            s_dir = secondary_direction[i]
            c_idx = cluster_assignments[i]
            current_price = close_array[i]

            if p_dir is None or s_dir is None or c_idx is None:
                time.sleep(SIGNAL_CHECK_INTERVAL)
                continue

            # Check last 3 secondary directions for pullback pattern
            if len(last_secondary_directions) >= 3:
                # Extract last 3 directions
                recent_3 = last_secondary_directions[-3:]
                # Compute corresponding indices from the last 3 elements in close_array
                indices = list(range(len(close_array) - 3, len(close_array)))

                # LONG pattern: [1, -1, 1] within MAX_PULLBACK_CANDLES
                bullish_bearish_bullish = (recent_3 == [1, -1, 1] and (indices[-1] - indices[0] <= MAX_PULLBACK_CANDLES))
    
                # SHORT pattern: [-1, 1, -1] within MAX_PULLBACK_CANDLES
                bearish_bearish_bearish = (recent_3 == [-1, 1, -1] and (indices[-1] - indices[0] <= MAX_PULLBACK_CANDLES))



                # ============ LONG ENTRY ============
                if (not in_position) and bullish_bearish_bullish and p_dir == 1 and c_idx == 0:
                    # Stop-loss = current bar's low
                    sl = low_array[i]
                    # Distance from entry to SL
                    dist = current_price - sl
                    # Take-profit = entry + 1.5 * dist
                    tp = current_price + (1.5 * dist)
                    logging.info("Pullback BUY triggered!")
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

                # ============ SHORT ENTRY ============
                if (not in_position) and bearish_bearish_bearish and p_dir == -1 and c_idx == 0:
                    # Stop-loss = current bar's high
                    sl = high_array[i]
                    dist = sl - current_price
                    # Take-profit = entry - 1.5 * dist
                    tp = current_price - (1.5 * dist)
                    logging.info("Pullback SHORT triggered!")
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


        except Exception as e:
            logging.error(f"Error in check_signals loop: {e}", exc_info=True)

        time.sleep(SIGNAL_CHECK_INTERVAL)

# ============== WEBSOCKET CALLBACK ==============
def on_message_candle(msg):
    global hv_new, mv_new, lv_new  # Declare globals at the start of the function

    if 'k' not in msg:
        return

    k = msg['k']
    is_final = k['x']
    close_price = float(k['c'])
    high_price = float(k['h'])
    low_price = float(k['l'])
    open_time = k['t']

    if is_final:
        # Append new candle
        time_array.append(open_time)
        high_array.append(high_price)
        low_array.append(low_price)
        close_array.append(close_price)

        # Trim to MAX_CANDLES
        if len(time_array) > MAX_CANDLES:
            time_array.pop(0)
            high_array.pop(0)
            low_array.pop(0)
            close_array.pop(0)

        # Recompute ATR
        new_atr = compute_atr(high_array, low_array, close_array, ATR_LEN)
        atr_array.clear()
        atr_array.extend(new_atr)

        # Trim atr_array if necessary
        while len(atr_array) > len(close_array):
            atr_array.pop(0)

        # Adjust cluster_assignments length
        while len(cluster_assignments) < len(close_array):
            cluster_assignments.append(None)
        while len(cluster_assignments) > len(close_array):
            cluster_assignments.pop(0)

        # Adjust primary/secondary arrays
        needed_len = len(close_array)
        def fix_arrays(st, di, ub, lb):
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

        fix_arrays(primary_supertrend, primary_direction, primary_upperBand, primary_lowerBand)
        fix_arrays(secondary_supertrend, secondary_direction, secondary_upperBand, secondary_lowerBand)

        # Possibly run K-Means
        data_count = len(close_array)
        if data_count >= TRAINING_DATA_PERIOD and (not CLUSTER_RUN_ONCE or (CLUSTER_RUN_ONCE and hv_new is None)):
            start_idx = data_count - TRAINING_DATA_PERIOD
            vol_data = [x for x in atr_array[start_idx:] if x is not None]
            if len(vol_data) == TRAINING_DATA_PERIOD:
                upper_val = max(vol_data)
                lower_val = min(vol_data)
                hv_init = lower_val + (upper_val - lower_val)*HIGHVOL_PERCENTILE
                mv_init = lower_val + (upper_val - lower_val)*MIDVOL_PERCENTILE
                lv_init = lower_val + (upper_val - lower_val)*LOWVOL_PERCENTILE

                hvf, mvf, lvf, _, _, _ = run_kmeans(vol_data, hv_init, mv_init, lv_init)
                if hvf and mvf and lvf:
                    hv_new, mv_new, lv_new = hvf, mvf, lvf
                    logging.info(f"K-Means Finalized: HV={hv_new:.4f}, MV={mv_new:.4f}, LV={lv_new:.4f}")
                else:
                    logging.warning("K-Means did not finalize due to insufficient data or errors.")

        # Assign cluster & compute supertrend for this bar
        i = len(close_array)-1
        assigned_centroid = None
        vol = atr_array[i]

        if hv_new is not None and mv_new is not None and lv_new is not None and vol is not None:
            dA = abs(vol - hv_new)
            dB = abs(vol - mv_new)
            dC = abs(vol - lv_new)
            distances = [dA, dB, dC]
            c_idx = distances.index(min(distances))  # 0=High,1=Med,2=Low
            cluster_assignments[i] = c_idx
            assigned_centroid = [hv_new, mv_new, lv_new][c_idx]
        else:
            logging.warning("Assigned centroid is None. Skipping SuperTrend computation for this bar.")

        if assigned_centroid is not None:
            compute_supertrend(
                i, PRIMARY_FACTOR, assigned_centroid,
                primary_supertrend, primary_direction,
                primary_upperBand, primary_lowerBand
            )
            compute_supertrend(
                i, SECONDARY_FACTOR, assigned_centroid,
                secondary_supertrend, secondary_direction,
                secondary_upperBand, secondary_lowerBand
            )
        else:
            # Assign default SuperTrend values if centroid is None
            compute_supertrend(
                i, PRIMARY_FACTOR, None,
                primary_supertrend, primary_direction,
                primary_upperBand, primary_lowerBand
            )
            compute_supertrend(
                i, SECONDARY_FACTOR, None,
                secondary_supertrend, secondary_direction,
                secondary_upperBand, secondary_lowerBand
            )

        # Update last_secondary_directions
        if secondary_direction[i] is not None:
            last_secondary_directions.append(secondary_direction[i])
            if len(last_secondary_directions) > 35:
                last_secondary_directions.pop(0)

# ============== BINANCE WEBSOCKET ==============
def start_binance_websocket():
    logging.info("Starting Binance WebSocket...")
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    twm.start()
    twm.start_kline_socket(
        callback=on_message_candle,
        symbol=BINANCE_SYMBOL,
        interval=BINANCE_INTERVAL
    )
    twm.join()  # Block here

# ============== MAIN ==============
if __name__ == "__main__":
    logging.info("Starting dual SuperTrend with long & short pullback logic...")

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
