import os
from binance.client import Client

# Load API key and secret from environment variables
api_key = os.getenv("TESTNET_API_KEY")
api_secret = os.getenv("TESTNET_SECRET_KEY")

# Initialize the Binance client for Testnet
client = Client(api_key, api_secret, testnet=True)

# Fetch account balances
account_info = client.get_account()

# Display the asset balances
print("Asset Balances:")
for balance in account_info['balances']:
    asset = balance['asset']
    free_balance = balance['free']
    locked_balance = balance['locked']
    if float(free_balance) > 0 or float(locked_balance) > 0:  # Show only non-zero balances
        print(f"{asset}: Free: {free_balance}, Locked: {locked_balance}")

