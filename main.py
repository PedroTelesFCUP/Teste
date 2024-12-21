import os
import sys
import time
import math
import statistics
import logging
import threading
from datetime import datetime

from binance import ThreadedWebsocketManager
from binance.client import Client
from alpaca_trade_api import REST as AlpacaREST
from flask import Flask

##########################################
# CONFIGURATION & LOGGING
##########################################
print("Main script start: Initializing logging...")
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.info("Logging initialized.")

# Env variables or replace with your keys
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL_ALPACA = "BTCUSD"
QTY = 0.001

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"

ATR_LEN = 10
PRIMARY_FACTOR = 3.0
SECONDARY_FACTOR = 8.0
TRAINING_DATA_PERIOD = 100
HIGHVOL_PERCENTILE = 0.75
MIDVOL_PERCENTILE = 0.5
LOWVOL_PERCENTILE = 0.25

HEARTBEAT_INTERVAL = 30
SIGNAL_CHECK_INTERVAL = 1
START_TIME = datetime(2024, 12, 1, 9, 30).timestamp() * 1000
END_TIME = datetime(2024, 12, 17, 16, 0).timestamp() * 1000
MAX_CANDLES = 100
CLUSTER_RUN_ONCE = True

# Globals
alpaca_api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

time_array = []
high_array = []
low_array = []
close_array = []
atr_array = []
cluster_assignments = []

hv_new = None
mv_new = None
lv_new = None

primary_supertrend = []
primary_direction = []
primary_upperBand = []
primary_lowerBand = []

secondary_supertrend = []
secondary_direction = []
secondary_upperBand = []
secondary_lowerBand = []

last_secondary_directions = []

in_position = False
position_side = None
entry_price = None

buy_signals = []
sell_signals = []

last_heartbeat_time = 0
lock = threading.Lock()

print("Global variables initialized.")

##########################################
# HELPER FUNCTIONS
##########################################
def wilder_smoothing(values, period):
    clean_values = [v for v in values if v is not None]
    if len(clean_values) < period:
        return [None]*len(values)
    start_index = 0
    for idx,v in enumerate(values):
        if v is not None:
            start_index = idx
            break
    length = len(values)
    result = [None]*length
    if length - start_index < period:
        return result
    window = values[start_index:start_index+period]
    if any(x is None for x in window):
        return result
    atr_val = sum(window)/period
    result[start_index+period-1] = atr_val
    for i in range(start_index+period, length):
        if values[i] is None or result[i-1] is None:
            result[i] = None
        else:
            prev = result[i-1]
            current = ((prev*(period-1)) + values[i]) / period
            result[i] = current
    return result

def compute_atr(h_array, l_array, c_array, period):
    length = len(c_array)
    if length <= period:
        return [None]*length
    tr_list = [None]*length
    for i in range(length):
        if i == 0:
            tr_list[i] = None
        else:
            t1 = h_array[i] - l_array[i]
            t2 = abs(h_array[i]-c_array[i-1])
            t3 = abs(l_array[i]-c_array[i-1])
            tr_list[i] = max(t1,t2,t3)
    return wilder_smoothing(tr_list, period)

def run_kmeans(vol_data, hv_init, mv_init, lv_init):
    amean = [hv_init]
    bmean = [mv_init]
    cmean = [lv_init]

    def means_stable(m):
        if len(m)<2:
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

def compute_supertrend(i, factor, assigned_atr, st_array, dir_array, ub_array, lb_array):
    with lock:
        length = len(close_array)
        if i<0 or i>=length:
            return
        if assigned_atr is None:
            st_array[i] = st_array[i-1] if i>0 else None
            dir_array[i] = dir_array[i-1] if i>0 else 1
            ub_array[i] = ub_array[i-1] if i>0 else None
            lb_array[i] = lb_array[i-1] if i>0 else None
            return

        hl2 = (high_array[i] + low_array[i]) / 2.0
        upBand = hl2 + factor * assigned_atr
        downBand = hl2 - factor * assigned_atr

        if i == 0:
            dir_array[i] = 1
            ub_array[i] = upBand
            lb_array[i] = downBand
            st_array[i] = upBand
            return

        prevDir = dir_array[i-1]
        prevUB = ub_array[i-1] if ub_array[i-1] is not None else upBand
        prevLB = lb_array[i-1] if lb_array[i-1] is not None else downBand

        if (downBand > prevLB or close_array[i-1]<prevLB):
            downBand = downBand
        else:
            downBand = prevLB

        if (upBand < prevUB or close_array[i-1]>prevUB):
            upBand = upBand
        else:
            upBand = prevUB

        wasUpper = (prevDir != -1)
        if wasUpper:
            if close_array[i] > upBand:
                dir_array[i] = -1
            else:
                dir_array[i] = 1
        else:
            if close_array[i] < downBand:
                dir_array[i] = 1
            else:
                dir_array[i] = -1

        st_array[i] = downBand if dir_array[i] == -1 else upBand
        ub_array[i] = upBand
        lb_array[i] = downBand

def execute_trade(side, qty, symbol, stop_loss=None, take_profit=None):
    logging.info(f"Executing {side.upper()} {qty} {symbol} (SL={stop_loss}, TP={take_profit})")
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
        logging.info(f"Order submitted: {order}")
        with lock:
            if side == "buy":
                buy_signals.append((len(close_array)-1, close_array[-1]))
            else:
                sell_signals.append((len(close_array)-1, close_array[-1]))
    except Exception as e:
        logging.error(f"Alpaca order failed: {e}", exc_info=True)

def heartbeat_logging():
    global last_heartbeat_time
    print("Heartbeat thread starting...")
    logging.info("Heartbeat thread started...")
    while True:
        try:
            now = time.time()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                with lock:
                    if len(close_array) == 0:
                        logging.info("No data yet.")
                    else:
                        i = len(close_array)-1
                        p_dir = primary_direction[i] if i<len(primary_direction) else None
                        s_dir = secondary_direction[i] if i<len(secondary_direction) else None
                        c_idx = cluster_assignments[i] if i<len(cluster_assignments) else None
                        assigned_centroid = None
                        if c_idx is not None and hv_new is not None and mv_new is not None and lv_new is not None:
                            assigned_centroid = [hv_new, mv_new, lv_new][c_idx]
                        pri_st = primary_supertrend[i] if i<len(primary_supertrend) else None
                        sec_st = secondary_supertrend[i] if i<len(secondary_supertrend) else None

                        msg = "\n=== Heartbeat ===\n"
                        msg += f"Last Price: {close_array[i]:.2f}\n"
                        msg += f"Primary Dir: {p_dir}\n"
                        msg += f"Secondary Dir: {s_dir}\n"
                        msg += f"Cluster: {c_idx if c_idx is not None else 'None'} (0=High,1=Med,2=Low)\n"
                        msg += f"Base ATR (Assigned Centroid): {assigned_centroid if assigned_centroid else 'N/A'}\n"
                        if assigned_centroid is not None:
                            primary_atr_val = assigned_centroid * PRIMARY_FACTOR
                            secondary_atr_val = assigned_centroid * SECONDARY_FACTOR
                        else:
                            primary_atr_val = 'N/A'
                            secondary_atr_val = 'N/A'
                        msg += f"Primary ATR: {primary_atr_val}\n"
                        msg += f"Secondary ATR: {secondary_atr_val}\n"
                        msg += f"PriST: {pri_st if pri_st else 'N/A'}\n"
                        msg += f"SecST: {sec_st if sec_st else 'N/A'}\n"
                        msg += f"In Position: {in_position} ({position_side})\n"
                        msg += f"Entry Price: {entry_price}\n"
                        msg += "=============="
                        logging.info(msg)

                    last_heartbeat_time = now
        except Exception as e:
            logging.error(f"Error in heartbeat: {e}", exc_info=True)
        time.sleep(1)

def check_signals():
    global in_position, position_side, entry_price
    print("Signal checking thread starting...")
    logging.info("Signal checking thread started...")
    while True:
        try:
            with lock:
                length = len(close_array)
                if length > 0:
                    i = length-1
                    t = time_array[i]
                    if t < START_TIME or t > END_TIME:
                        if in_position:
                            logging.info("Outside trading window. Closing position.")
                            execute_trade("sell", QTY, SYMBOL_ALPACA)
                            in_position = False
                            position_side = None
                            entry_price = None
                    else:
                        p_dir = primary_direction[i] if i<len(primary_direction) else None
                        s_dir = secondary_direction[i] if i<len(secondary_direction) else None
                        c_idx = cluster_assignments[i] if i<len(cluster_assignments) else None

                        if p_dir is not None and s_dir is not None and c_idx is not None and len(last_secondary_directions)>=3:
                            recent_3 = last_secondary_directions[-3:]
                            bullish_bearish_bullish = (recent_3 == [1, -1, 1])
                            bearish_bullish_bearish = (recent_3 == [-1, 1, -1])
                            current_price = close_array[i]

                            # LONG ENTRY
                            if (not in_position) and bullish_bearish_bullish and p_dir == 1 and c_idx == 0:
                                sl = low_array[i]
                                dist = current_price - sl
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

                            # SHORT ENTRY
                            if (not in_position) and bearish_bullish_bearish and p_dir == -1 and c_idx == 0:
                                sl = high_array[i]
                                dist = sl - current_price
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

                            # EXIT LOGIC
                            if in_position and position_side == "long" and p_dir == -1:
                                logging.info("Primary turned bearish. Closing LONG.")
                                execute_trade("sell", QTY, SYMBOL_ALPACA)
                                in_position = False
                                position_side = None
                                entry_price = None

                            if in_position and position_side == "short" and p_dir == 1:
                                logging.info("Primary turned bullish. Closing SHORT.")
                                execute_trade("buy", QTY, SYMBOL_ALPACA)
                                in_position = False
                                position_side = None
                                entry_price = None

        except Exception as e:
            logging.error(f"Error in check_signals: {e}", exc_info=True)
        time.sleep(SIGNAL_CHECK_INTERVAL)

def on_message_candle(msg):
    global hv_new, mv_new, lv_new
    try:
        if 'k' not in msg:
            return
        k = msg['k']
        is_final = k['x']
        close_price = float(k['c'])
        high_price = float(k['h'])
        low_price = float(k['l'])
        open_time = k['t']

        if is_final:
            with lock:
                time_array.append(open_time)
                high_array.append(high_price)
                low_array.append(low_price)
                close_array.append(close_price)

                while len(time_array) > MAX_CANDLES:
                    time_array.pop(0)
                    high_array.pop(0)
                    low_array.pop(0)
                    close_array.pop(0)

                if len(close_array) > ATR_LEN:
                    new_atr = compute_atr(high_array, low_array, close_array, ATR_LEN)
                    atr_array.clear()
                    atr_array.extend(new_atr)
                else:
                    atr_array.clear()
                    atr_array.extend([None]*len(close_array))

                while len(atr_array) > len(close_array):
                    atr_array.pop(0)

                while len(cluster_assignments) < len(close_array):
                    cluster_assignments.append(None)
                while len(cluster_assignments) > len(close_array):
                    cluster_assignments.pop(0)

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

                fix_arrays(primary_supertrend, primary_direction, primary_upperBand, primary_lowerBand)
                fix_arrays(secondary_supertrend, secondary_direction, secondary_upperBand, secondary_lowerBand)

                data_count = len(close_array)
                if data_count >= TRAINING_DATA_PERIOD and (not CLUSTER_RUN_ONCE or (CLUSTER_RUN_ONCE and hv_new is None)):
                    vol_data = [x for x in atr_array[data_count-TRAINING_DATA_PERIOD:] if x is not None]
                    if len(vol_data) == TRAINING_DATA_PERIOD:
                        upper_val = max(vol_data)
                        lower_val = min(vol_data)
                        hv_init = lower_val + (upper_val - lower_val)*HIGHVOL_PERCENTILE
                        mv_init = lower_val + (upper_val - lower_val)*MIDVOL_PERCENTILE
                        lv_init = lower_val + (upper_val - lower_val)*LOWVOL_PERCENTILE

                        hvf, mvf, lvf, _, _, _ = run_kmeans(vol_data, hv_init, mv_init, lv_init)
                        hv_new, mv_new, lv_new = hvf, mvf, lvf
                        logging.info(f"K-Means Finalized: HV={hv_new:.4f}, MV={mv_new:.4f}, LV={lv_new:.4f}")

                i = len(close_array)-1
                assigned_centroid = None
                vol = atr_array[i] if i<len(atr_array) else None
                if hv_new is not None and mv_new is not None and lv_new is not None and vol is not None:
                    dA = abs(vol - hv_new)
                    dB = abs(vol - mv_new)
                    dC = abs(vol - lv_new)
                    distances = [dA, dB, dC]
                    c_idx = distances.index(min(distances))
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

                if i<len(secondary_direction) and secondary_direction[i] is not None:
                    last_secondary_directions.append(secondary_direction[i])
                    if len(last_secondary_directions) > 10:
                        last_secondary_directions.pop(0)

    except Exception as e:
        logging.error(f"Error in on_message_candle: {e}", exc_info=True)

def start_binance_websocket():
    print("Binance WebSocket thread starting...")
    logging.info("Starting Binance WebSocket...")
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    twm.start()
    twm.start_kline_socket(
        callback=on_message_candle,
        symbol=BINANCE_SYMBOL.lower(),
        interval=BINANCE_INTERVAL
    )
    twm.join()

##########################################
# MAIN
##########################################
if __name__ == "__main__":
    print("Main function start: Starting threads...")
    logging.info("Starting heartbeat and signals threads.")

    hb_thread = threading.Thread(target=heartbeat_logging, daemon=True)
    hb_thread.start()

    signal_thread = threading.Thread(target=check_signals, daemon=True)
    signal_thread.start()

    # Additionally start a minimal Flask server so that Render sees an open port.
    # If you don't want this, convert to a background worker on Render.
    server = Flask(__name__)

    @server.route("/")
    def home():
        return "Bot is running!", 200

    def run_server():
        print("Starting minimal web server on port 8080...")
        server.run(host="0.0.0.0", port=8080)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Start binance websocket (blocking)
    try:
        start_binance_websocket()
    except Exception as e:
        logging.error(f"Binance WebSocket error: {e}", exc_info=True)
        sys.exit(1)


