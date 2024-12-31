import os
from binance.client import Client
from binance.enums import SIDE_SELL, ORDER_TYPE_MARKET

# Load API key and secret from environment variables
api_key = os.getenv("TESTNET_API_KEY")
api_secret = os.getenv("TESTNET_SECRET_KEY")

# Initialize Binance client for Testnet
client = Client(api_key, api_secret, testnet=True)

def get_balance(asset):
    """Fetch the balance of a specific asset."""
    account_info = client.get_account()
    for balance in account_info['balances']:
        if balance['asset'] == asset:
            free_balance = float(balance['free'])
            return free_balance
    return 0.0

def sell_asset(symbol, asset, percentage):
    """Sell a percentage of the available balance of an asset."""
    # Get the current balance
    balance = get_balance(asset)
    if balance <= 0:
        print(f"Insufficient balance for {asset}. Current balance: {balance}")
        return

    # Calculate the amount to sell (95% of the balance)
    amount_to_sell = balance * percentage
    # Binance enforces precision; truncate to 8 decimal places
    amount_to_sell = float(f"{amount_to_sell:.8f}")

    # Check if the amount to sell meets the minimum lot size
    if amount_to_sell < 0.0001:  # Example minimum lot size for BTC
        print(f"Amount to sell ({amount_to_sell}) is below the minimum lot size.")
        return

    try:
        # Place a market sell order
        order = client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=amount_to_sell
        )
        print("Sell order placed successfully:")
        print(order)
    except Exception as e:
        print(f"Error placing sell order: {e}")

# Main logic
if __name__ == "__main__":
    asset = "BTC"  # Asset to sell
    symbol = "BTCUSDT"  # Trading pair
    percentage_to_sell = 0.95  # 95%

    # Attempt to sell 95% of the BTC balance
    sell_asset(symbol, asset, percentage_to_sell)
