import pandas as pd
import requests
import time
import os

# Output file
OUTPUT_FILE = "sec_fundamentals_analyst_tickers.csv"
# OUTPUT_FILE = "sec_fundamentals_all_tickers.csv"

# Load date range from your ratings CSV
df = pd.read_csv("Sentiment/analyst_ratings_with_sentiment.csv")

df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df = df.dropna(subset=['date'])
start_date = df['date'].min()
end_date = df['date'].max()
print(f"Using date range: {start_date.date()} to {end_date.date()}")

# GAAP tags to fetch
gaap_tags = {
    "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncome": "NetIncomeLoss",
    "EPS_Basic": "EarningsPerShareBasic",
    "EPS_Diluted": "EarningsPerShareDiluted",
    "Assets": "Assets",
    "Liabilities": "Liabilities",
    "Equity": "StockholdersEquity",
    "GrossProfit": "GrossProfit",
    "OperatingIncome": "OperatingIncomeLoss"
}

# Get SEC Ticker → CIK mapping
def get_all_cik_mappings():
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'hedgefundbot (Org: BlueJacketIndustries, Email: Tarenpatel1013@gmail.com)'}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    return {item["ticker"].upper(): str(item["cik_str"]).zfill(10) for item in data.values()}

# Fetch data for one tag
def fetch_tag_data(cik, tag):
    """
    Fetch XBRL facts for a given CIK and tag, preferring USD but falling back
    to the first non-empty unit if USD is not available.
    """
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    headers = {'User-Agent': 'hedgefundbot (Org: BlueJacketIndustries, Email: Tarenpatel1013@gmail.com)'}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return []
    data = r.json().get("units", {})
    
    for unit_name, facts in data.items():
        if facts:
            return facts
    
    return []

# Load completed tickers if file exists
if os.path.exists(OUTPUT_FILE):
    try:
        existing = pd.read_csv(OUTPUT_FILE, usecols=["ticker"])
        completed_tickers = set(existing["ticker"].unique())
        print(f"Resuming: {len(completed_tickers)} tickers already fetched.")
    except Exception as e:
        print(f"Couldn't read existing file: {e}")
        completed_tickers = set()
else:
    pd.DataFrame(columns=["ticker", "cik", "metric", "date", "value"]).to_csv(OUTPUT_FILE, index=False)
    completed_tickers = set()

# Main loop
#Fetch all tickers
# cik_map = get_all_cik_mappings()

# Only fetch CIKs for tickers in your ratings dataset
all_ciks = get_all_cik_mappings()
valid_tickers = df['stock'].dropna().unique()
cik_map = {ticker: all_ciks[ticker] for ticker in valid_tickers if ticker in all_ciks}
total = len(cik_map)

for i, (ticker, cik) in enumerate(cik_map.items(), start=1):
    if ticker in completed_tickers:
        # print(f"[{i}/{total}] Skipping {ticker}")
        continue

    print(f"[{i}/{total}] Fetching {ticker} (CIK: {cik})")
    rows = []

    for label, tag in gaap_tags.items():
        try:
            data_points = fetch_tag_data(cik, tag)
            # print(f"   -> {label}: {len(data_points)} data points pulled")
            for dp in data_points:
                try:
                    d = pd.to_datetime(dp.get("end"), utc=True)
                    if start_date <= d <= end_date:
                        rows.append({
                            "ticker": ticker,
                            "cik": cik,
                            "metric": label,
                            "date": d.date(),
                            "value": dp.get("val")
                        })
                except Exception as e:
                    print(f"    X Error parsing date for {ticker}/{label}: {e}")
                    continue
            time.sleep(0.2)
        except Exception as e:
            print(f"  X Failed to fetch {label} for {ticker} (CIK: {cik}): {e}")
            with open("failed_fetches.log", "a") as log_file:
                log_file.write(f"{ticker},{cik},{label}\n")
            continue

    # print(f"Total valid rows: {len(rows)}")

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        # print(f"Appended {len(rows)} rows for {ticker}")
    else:
        print(f"No rows written for {ticker}")

    completed_tickers.add(ticker)

print("All Data Gathered")
