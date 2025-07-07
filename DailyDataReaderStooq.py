import pandas_datareader as web
import time

symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']  # Add your 100+ symbols here
results = {}

for symbol in symbols:
    try:
        data = web.stooq.StooqDailyReader(
            symbols=symbol,
            start='1/1/10',
            end='2/1/10',
            retry_count=3,
            pause=0.1
        ).read()
        results[symbol] = data
        print(f"{symbol} loaded.")
        time.sleep(1)  # ⏳ Pause to avoid being blocked
    except Exception as e:
        print(f"Error loading {symbol}: {e}")

print(results)
