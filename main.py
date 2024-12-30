import os
from decimal import Decimal, ROUND_DOWN
from binance.client import Client

# Testnet API credentials
TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "YOUR_TESTNET_API_KEY")  
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "YOUR_TESTNET_SECRET_KEY")
TESTNET_BASE_URL = "https://testnet.binance.vision"

# Initialize Binance Testnet Client
testnet_api = Client(api_key=TESTNET_API_KEY, api_secret=TESTNET_SECRET_KEY)
testnet_api.API_URL = TESTNET_BASE_URL + "/api"

# Placeholder for logging function
def log_message(message, level=None):
    """Simple logging function."""
    print(f"[LOG] {message}")

# Placeholder for bot_status dictionary
bot_status = {
    "in_position": False,
    "position_size": 0,
    "last_trade_price": 0,
    "current_price": 0,
}

def format_price(price, tick_size):
    """
    Convert `price` into a Decimal quantized to match `tick_size` precision.
    e.g. if tick_size=0.01, we keep 2 decimals; if 0.0001, we keep 4 decimals.
    """
    p = Decimal(str(price))
    t = Decimal(str(tick_size))
    return str(p.quantize(t, rounding=ROUND_DOWN))

def get_last_price(symbol):
    """Fetch the last price for the given symbol from Binance Testnet."""
    try:
        ticker = testnet_api.get_ticker(symbol=symbol)
        last_price = float(ticker['lastPrice'])
        return last_price
    except Exception as e:
        log_message(f"Error fetching last price for {symbol}: {e}", level="ERROR")
        return None

def place_binance_oco_order(symbol, qty, side, stop_loss, stop_loss_limit, take_profit):
    """
    Places an OCO order on Binance Testnet.

    Args:
        symbol (str): Trading pair (e.g., "BTCUSDT").
        qty (float): Quantity to trade.
        side (str): "BUY" or "SELL".
        stop_loss (float): Stop-loss activation price.
        stop_loss_limit (float): Stop-loss limit price.
        take_profit (float): Take-profit limit price.

    Returns:
        dict or None: Response from the OCO order if successful, otherwise None.
    """
    try:
        # Fetch precision rules for the symbol
        symbol_info = testnet_api.get_symbol_info(symbol)
        price_filter = next(f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")
        tick_size = float(price_filter["tickSize"])

        # Format prices with correct precision
        stop_loss_str = format_price(stop_loss, tick_size)
        stop_loss_limit_str = format_price(stop_loss_limit, tick_size)
        take_profit_str = format_price(take_profit, tick_size)

        # Define OCO parameters
        oco_params = {
            "symbol": symbol,
            "side": side,  # Use the original side 
            "quantity": str(qty),
            "timestamp": int(testnet_api.get_server_time()['serverTime']), # Add timestamp
            "aboveType": "TAKE_PROFIT",      # take-profit type
            "aboveStopPrice": stop_loss_str,   # Stop-loss limit price
            "belowType": "STOP_LOSS",          # Take-profit type
            "belowStopPrice":  take_profit_str       # Take-profit price
        }

        # Place the OCO order 
        oco_order = testnet_api.create_oco_order(**oco_params)
        log_message(f"OCO order placed: SL={stop_loss_str}, SLL={stop_loss_limit_str}, TP={take_profit_str}")

        return oco_order

    except Exception as e:
        log_message(f"Error placing OCO order: {e}", level="ERROR")
        return None

# Main execution
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    QUANTITY = 0.001
    LAST_PRICE = get_last_price(SYMBOL)

    if LAST_PRICE:
        # Define stop-loss and take-profit levels
        STOP_LOSS = LAST_PRICE - 200
        STOP_LOSS_LIMIT = LAST_PRICE - 210
        TAKE_PROFIT = LAST_PRICE + 200

        # Place the OCO order (use the correct side "BUY")
        response = place_binance_oco_order(SYMBOL, QUANTITY, "BUY", STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT)
        print("OCO Order Response:", response)
    else:
        print("Failed to fetch the last price.")

