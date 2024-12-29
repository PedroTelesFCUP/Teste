import os
from binance.client import Client
from binance.enums import SIDE_SELL, ORDER_TYPE_MARKET

# Load API key and secret from environment variables
api_key = os.getenv("TESTNET_API_KEY")
api_secret = os.getenv("TESTNET_SECRET_KEY")

# Initialize the Binance client for Testnet
client = Client(api_key, api_secret, testnet=True)

# Specify the trading pair and amount to sell
symbol = "BTCUSDT"  # Trading pair
quantity = 1.10500000  # Amount of BTC to sell

try:
    # Place a market sell order
    order = client.create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=quantity
    )

    # Print order details
    print("Order placed successfully:")
    print(order)

except Exception as e:
    print(f"Error placing sell order: {e}")
