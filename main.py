import os
import sys
import time
import math
import statistics
import logging
import threading
import pandas as pd
from datetime import datetime

# 3rd-party libraries
from binance import ThreadedWebsocketManager
from binance.client import Client
from alpaca_trade_api import REST as AlpacaREST

# ============== CONFIGURATION ==============
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Environment variables / credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

# Alpaca endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # For paper trading
SYMBOL_ALPACA = "BTCUSD"  # e.g. "BTCUSD" on Alpaca
QTY = 0.001               # Example size

# Binance symbol & timeframe
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"  # Candle interval: 1m, 5m, 15m, etc.

# Strategy / logic parameters
ATR_LEN = 10
PRIMARY_FACTOR = 3.0      # SuperTrend factor for primary signal
SECONDARY_FACTOR = 8.0    # SuperTrend factor for secondary signal
TRAINING_DATA_PERIOD = 100
MAX_PULLBACK_CANDLES = 30
HIGHVOL_PERCENTILE = 0.75
MIDVOL_PERCENTILE = 0.5
LOWVOL_PERCENTILE = 0.25

# Time window for valid trades (example)
# We'll use Python's datetime -> timestamp in ms
START_TIME = datetime(2024, 12, 1, 9, 30).timestamp() * 1000  # 9:30
END_TIME   = datetime(2024, 12, 17, 16, 0).timestamp() * 1000 # 16:00

# Trading signals refresh intervals
SIGNAL_CHECK_INTERVAL = 1.0  # seconds
CLUSTER_RUN_ONCE = True       # If True, run K-Means only once after we have the training period data

# ============== ALPACA & BINANCE CLIENTS ==============
alpaca_api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# ============== GLOBAL DATA STRUCTURES ==============
time_array = []
high_array = []
low_array = []
close_array = []

# We store final ATR(10) used for clustering
atr_array = []

# We'll store the stable K-Means centroids for High/Medium/Low once they are computed
hv_new = None
mv_new = None
lv_new = None

# For each candle, we store cluster assignment (0,1,2) or None
cluster_assignments = []

# Two sets of arrays for each signal:
# PRIMARY
primary_supertrend = []
primary_direction = []
primary_upperBand = []
primary_lowerBand = []

# SECONDARY
secondary_supertrend = []
secondary_direction = []
secondary_upperBand = []
secondary_lowerBand = []

# We track direction flips in secondary to detect bullish->bearish->bullish
last_secondary_directions = []

# Simple position tracking
in_position = False
entry_price = None

# ============== HELPER FUNCTIONS ==============

def wilder_smoothing(values, period):
    """
    Implements Wilder's smoothing. 
    ATR in TradingView Pine is typically: 
      atr[ i ] = ((atr[i-1]*(period-1)) + tr[i]) / period
    """
    result = [None]*len(values)
    if len(values) < period:
        return result
    # First valid ATR uses simple average
    initial = sum(values[:period]) / period
    result[period-1] = initial
    for i in range(period, len(values)):
        prev = result[i-1]
        current = ((prev*(period-1)) + values[i]) / period
        result[i] = current
    return result

def compute_atr(h_array, l_array, c_array, period):
    """
    Candle-by-candle ATR( period ) with Wilder's smoothing.
    """
    tr_list = []
    for i in range(len(c_array)):
        if i == 0:
            tr_list.append(None)
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
    Replicates the K-Means style clustering from Pine:
    1) Start with hv_init, mv_init, lv_init
    2) Assign each vol_data point to the nearest centroid
    3) Recompute centroids
    4) Repeat until stable
    Returns final (hv, mv, lv) and cluster sizes.
    """
    amean = [hv_init]
    bmean = [mv_init]
    cmean = [lv_init]

    def means_stable(m):
        # returns True if the last two values of m are nearly identical
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
            dist_a = abs(v - cur_a)
            dist_b = abs(v - cur_b)
            dist_c = abs(v - cur_c)
            min_dist = min(dist_a, dist_b, dist_c)
            if min_dist == dist_a:
                hv_cluster.append(v)
            elif min_dist == dist_b:
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

def compute_supertrend(
    i, 
    factor, 
    assigned_atr, 
    st_array, 
    dir_array, 
    ub_array, 
    lb_array
):
    """
    Candle-by-candle SuperTrend replication from Pine code:
      upperBand = hl2 + factor * ATR
      lowerBand = hl2 - factor * ATR
      Then "band persistence" logic:
        lowerBand := lowerBand > prevLowerBand or close[1]<prevLowerBand ? lowerBand : prevLowerBand
        upperBand := upperBand < prevUpperBand or close[1]>prevUpperBand ? upperBand : prevUpperBand
      direction logic:
        if na(atr[1]):
            direction=1
        else if prevSuperTrend==prevUpperBand:
            direction= close>upperBand ? -1 : 1
        else:
            direction= close<lowerBand ? 1 : -1
      superTrend= direction == -1 ? lowerBand : upperBand
    We store results in st_array, dir_array, ub_array, lb_array.
    """
    if assigned_atr is None:
        # Before we have assigned_atr, just carry forward
        st_array[i] = st_array[i-1] if i>0 else None
        dir_array[i] = dir_array[i-1] if i>0 else 1
        ub_array[i] = ub_array[i-1] if i>0 else None
        lb_array[i] = lb_array[i-1] if i>0 else None
        return

    hl2 = (high_array[i] + low_array[i]) / 2.0
    upBand = hl2 + factor * assigned_atr
    downBand = hl2 - factor * assigned_atr

    if i == 0:
        # First candle with valid assigned_atr
        dir_array[i] = 1
        ub_array[i] = upBand
        lb_array[i] = downBand
        # superTrend = lowerBand if direction==-1 else upperBand if direction==-1 else ?
        # Actually from the Pine code, if na(atr[1]) => direction=1 => superTrend= upperBand
        st_array[i] = upBand
        return

    # We have a previous candle
    prevST = st_array[i-1]
    prevDir = dir_array[i-1]
    prevUB = ub_array[i-1] if ub_array[i-1] is not None else upBand
    prevLB = lb_array[i-1] if lb_array[i-1] is not None else downBand

    # Pine band continuity
    # lowerBand = lowerBand > prevLB or close[i-1]<prevLB ? lowerBand : prevLB
    if (downBand > prevLB or close_array[i-1]<prevLB):
        downBand = downBand
    else:
        downBand = prevLB

    # upperBand = upperBand < prevUB or close[i-1]>prevUB ? upperBand : prevUB
    if (upBand < prevUB or close_array[i-1]>prevUB):
        upBand = upBand
    else:
        upBand = prevUB

    # direction logic
    # if na(atr[1]) => direction=1 (handled on i=0)
    # else if prevSuperTrend==prevUpperBand => direction= close>upperBand ? -1 : 1
    # else => direction= close<lowerBand ? 1 : -1
    # But in Pine, we check if prevST was the upperBand or lowerBand:
    # If direction was -1 => previous ST was lowerBand, else upperBand
    wasUpper = (prevDir != -1)  # If direction != -1 => ST was upperBand
    if wasUpper:
        # direction = -1 if close>upBand else 1
        if close_array[i] > upBand:
            dir_array[i] = -1
        else:
            dir_array[i] = 1
    else:
        # direction = 1 if close<downBand else -1
        if close_array[i] < downBand:
            dir_array[i] = 1
        else:
            dir_array[i] = -1

    # superTrend = direction == -1 ? lowerBand : upperBand
    if dir_array[i] == -1:
        st_array[i] = downBand
    else:
        st_array[i] = upBand

    ub_array[i] = upBand
    lb_array[i] = downBand

# ============== REAL-TIME DATA (BINANCE WEBSOCKET) ==============
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
                hv_new, mv_new, lv_new = hvf, mvf, lvf
                logging.info(f"K-Means Finalized: HV={hv_new:.4f}, MV={mv_new:.4f}, LV={lv_new:.4f}")

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

        # Update last_secondary_directions
        if secondary_direction[i] is not None:
            last_secondary_directions.append(secondary_direction[i])
            if len(last_secondary_directions) > 30:
                last_secondary_directions.pop(0)


def start_binance_websocket():
    """
    Starts Binance WebSocket to stream kline data for BINANCE_SYMBOL on BINANCE_INTERVAL.
    """
    logging.info("Starting Binance WebSocket...")
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    twm.start()
    twm.start_kline_socket(
        callback=on_message_candle,
        symbol=BINANCE_SYMBOL.lower(),
        interval=BINANCE_INTERVAL
    )
    twm.join()  # Keep the socket open

# ============== TRADING LOGIC ==============
def execute_trade(side, qty, symbol):
    """
    Places a market order on Alpaca. You can enhance with stop-loss/take-profit if you wish.
    """
    try:
        logging.info(f"Executing {side} {qty} {symbol}")
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        logging.info(f"Order submitted: {order}")
    except Exception as e:
        logging.error(f"Alpaca order failed: {e}", exc_info=True)

def check_signals():
    """
    Periodically checks for signals based on:
    - Primary direction is bullish
    - Secondary direction pattern: bullish(-1)->bearish(1)->bullish(-1) ??? 
      Actually you mentioned "bullish->bearish->bullish" = direction transitions from +1->-1->+1 
      But let's clarify:
      Typically, direction=1 means bullish, direction=-1 means bearish in many supertrend scripts. 
      We'll assume that's the case here.
    - Volatility = cluster 0 (i.e. High)
    - Must be within trading time window
    - If in no position, we buy. If in a position, we might exit. 
      Adjust logic to your preference.
    """
    global in_position, entry_price

    while True:
        try:
            if len(close_array) == 0:
                time.sleep(SIGNAL_CHECK_INTERVAL)
                continue

            idx = len(close_array)-1  # last bar index
            t = time_array[idx]
            if t < START_TIME or t > END_TIME:
                # Outside trading window, flatten any position
                if in_position:
                    logging.info("Outside trading window. Closing position.")
                    execute_trade("sell", QTY, SYMBOL_ALPACA)
                    in_position = False
                    entry_price = None
                time.sleep(SIGNAL_CHECK_INTERVAL)
                continue

            # We only check signals if the last bar is fully formed:
            # Pine signals typically trigger on bar close, which is what we do after is_final in on_message_candle.

            # 1) Confirm we have directions for both signals
            p_dir = primary_direction[idx]
            s_dir = secondary_direction[idx]
            c_idx = cluster_assignments[idx]

            if p_dir is None or s_dir is None or c_idx is None:
                time.sleep(SIGNAL_CHECK_INTERVAL)
                continue

            # 2) Check pullback pattern in secondary:
            #    bullish->bearish->bullish means direction= +1-> -1-> +1
            #    We'll see if last_secondary_directions includes that pattern in the last 3 bars: [1, -1, 1]
            #    But your Pine code had different logic, so let's replicate your snippet:
            #    "One with ATR=3 and one with ATR=8. Make decisions on buy/sell with pullback
            #     when the primary signal is bullish and there is a bullish->bearish->bullish trend in the secondary 
            #     and cluster=0 (highest)."
            #    We'll do exactly that:
            if len(last_secondary_directions) >= 3:
                recent_3 = last_secondary_directions[-3:]
                indices = list(range(len(close_array) - 3, len(close_array)))

                # LONG pattern: [1, -1, 1] within MAX_PULLBACK_CANDLES
                bullish_bearish_bullish = (recent_3 == [1, -1, 1] and (indices[-1] - indices[0] <= MAX_PULLBACK_CANDLES))

                # SHORT pattern: [-1, 1, -1] within MAX_PULLBACK_CANDLES
                bearish_bearish_bearish = (recent_3 == [-1, 1, -1] and (indices[-1] - indices[0] <= MAX_PULLBACK_CANDLES))

                # ============ LONG ENTRY ============
                if (not in_position) and bullish_bearish_bullish and p_dir == 1:  # and c_idx == 0:
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
                if (not in_position) and bearish_bearish_bearish and p_dir == -1:  # and c_idx == 0:
                    # Stop-loss = current bar's high
                    sl = high_array[i]
                    # Distance from entry to SL
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

# ============== MAIN ==============
if __name__ == "__main__":
    logging.info("Starting the dual SuperTrend strategy with K-Means clustering...")

    # Start a thread for signals checking
    signal_thread = threading.Thread(target=check_signals, daemon=True)
    signal_thread.start()

    # Start binance websocket for price data
    try:
        start_binance_websocket()
    except Exception as e:
        logging.error(f"Binance WebSocket error: {e}", exc_info=True)
        sys.exit(1)


