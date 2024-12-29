from binance.client import Client

# Replace these with your Testnet API keys
api_key = 'your_testnet_api_key'
api_secret = 'your_testnet_secret_key'

# Initialize the client for Testnet
client = Client(api_key, api_secret, testnet=True)

# Fetch account information
account_info = client.get_account()

# Print balances
print("Asset Balances:")
for balance in account_info['balances']:
    asset = balance['asset']
    free_balance = balance['free']
    locked_balance = balance['locked']
    print(f"{asset}: Free: {free_balance}, Locked: {locked_balance}")


