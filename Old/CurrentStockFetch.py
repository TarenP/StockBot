import pandas as pd
import yfinance as yf

# Load the CSV file
df = pd.read_csv("analyst_ratings_clean.csv")

# Drop rows where the 'stock' column is missing
df = df.dropna(subset=["stock"])

# Extract unique tickers
tickers = df['stock'].unique().tolist()

# Print how many were found and preview
print(f"Total unique tickers: {len(tickers)}")

all_stats = []

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        stats = {
            "symbol": ticker,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "book_value": info.get("bookValue"),
            "total_revenue": info.get("totalRevenue"),
            "debt_to_equity": info.get("debtToEquity"),
        }

        all_stats.append(stats)
    except Exception as e:
        print(f"Failed to fetch for {ticker}: {e}")

# Convert to DataFrame and save
df_stats = pd.DataFrame(all_stats)
df_stats.to_csv("fundamentals.csv", index=False)
print(df_stats.head())