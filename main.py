import os
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.enums import *

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

def place_binance_testnet_order(symbol, qty, side, stop_loss=0.0, take_profit=0.0):
    """
    Places a market order and optionally an OCO order on Binance Testnet.

    Args:
        symbol (str): Trading pair (e.g., "BTCUSDT").
        qty (float): Quantity to trade.
        side (str): "BUY" or "SELL".
        stop_loss (float): Stop-loss activation price (optional).
        take_profit (float): Take-profit limit price (optional).

    Returns:
        dict or None: Response from main market order if successful, otherwise None.
    """
    side_lower = side.lower()
    try:
        # 1) Fetch precision rules for the symbol
        symbol_info = testnet_api.get_symbol_info(symbol)
        price_filter = next(f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")
        tick_size = float(price_filter["tickSize"])

        # Format stop_loss & take_profit with correct precision
        stop_loss_str = format_price(stop_loss, tick_size)
        take_profit_str = format_price(take_profit, tick_size)

        # Place the main market order in its own try
        try:
            if side_lower == "buy":
                order = testnet_api.order_market_buy(symbol=symbol, quantity=qty)
            elif side_lower == "sell":
                order = testnet_api.order_market_sell(symbol=symbol, quantity=qty)
            else:
                raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

            log_message(f"Main market order placed. side={side}, qty={qty}, symbol={symbol}")

            # Indicate we are in position if order is not None
            bot_status["in_position"] = True
            bot_status["position_size"] = qty
            bot_status["last_trade_price"] = bot_status["current_price"]

        except Exception as e_main:
            log_message(f"Error placing main {side} market order: {e_main}", level="ERROR")
            return None

        # Place OCO only if both SL & TP > 0
        if float(stop_loss) > 0.0 and float(take_profit) > 0.0:
            try:
                # side for OCO is the opposite of the direction we traded
                # If we bought, we want to place a SELL OCO to close that position.
                # If we sold, we want a BUY OCO to close that short, etc.
                oco_side = "SELL" if side_lower == "buy" else "BUY"

                oco_params = {
                    "symbol": symbol,
                    "side": oco_side,
                    "quantity": str(qty),
                    "aboveType": "STOP_LOSS_LIMIT",     # The "stop-loss-limit" portion
                    "aboveTimeInForce": "GTC",
                    "belowType": "LIMIT_MAKER",         # The "take-profit-limit" or limit_maker portion
                }

                # For a BUY -> OCO SELL scenario
                # Typical assumption: stop_loss < currentPrice < take_profit
                # => stop_loss -> STOP_LOSS_LIMIT, take_profit -> LIMIT_MAKER
                if side_lower == "buy":
                    oco_params.update({
                        "abovePrice": take_profit_str,     # STOP_LOSS_LIMIT limit price
                        "aboveStopPrice": take_profit_str, # the stop price
                        "belowPrice": stop_loss_str,   # LIMIT_MAKER price
                    })
                else:
                    # For a SELL -> OCO BUY scenario (less common),
                    # you might do the reverse logic if you want a buy stop-limit above the market
                    # and a buy limit below the market for the "take profit" (in a short scenario).
                    # Adjust if needed:
                    oco_params.update({
                        "abovePrice": stop_loss_str,
                        "aboveStopPrice": stop_loss_str,
                        "belowPrice": take_profit_str,
                    })

                # Attempt placing the OCO
                oco_order = testnet_api.create_oco_order(**oco_params)
                log_message(f"OCO order placed for SL={stop_loss_str}, TP={take_profit_str}, side={oco_side}")

            except Exception as e_oco:
                log_message(f"Error placing OCO order: {e_oco}", level="ERROR")
        else:
            log_message("No OCO order placed (stop_loss or take_profit <= 0).")

        return order

    except Exception as e:
        log_message(f"Error in place_binance_testnet_order: {e}", level="ERROR")
        return None

# Main execution
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    QUANTITY = 0.001  # Example quantity

    # Fetch the last price for the symbol
    last_price = get_last_price(SYMBOL)
    if last_price is not None:
        TAKE_PROFIT = last_price + 200
        STOP_LOSS = last_price - 200

        # Call the function
        response = place_binance_testnet_order(SYMBOL, QUANTITY, "BUY", STOP_LOSS, TAKE_PROFIT)
        print("Order Response:", response)
    else:
        print("Failed to fetch the last price. Order not placed.")


