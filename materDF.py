import pandas as pd

df_prices = pd.read_parquet("stooq_panel.parquet")



# 1) Read your CSV *without* parse_dates, so pandas doesn’t try to infer tz info:
df_sent = pd.read_csv("Sentiment/analyst_ratings_with_sentiment.csv")

# 2) Force a clean datetime conversion, then drop any time component:
df_sent["date"] = (
    pd.to_datetime(df_sent["date"], utc=True, errors="coerce")  # parse, force UTC
      .dt.tz_convert(None)                                      # strip tz info
      .dt.normalize()                                           # set all times to 00:00
)

# 3) Upper-case your tickers and set the multi-index:
df_sent["ticker"] = df_sent["stock"].str.upper()
df_sent.set_index(["date", "ticker"], inplace=True)


# Join
df_all = df_prices.join(df_sent, how="left")
print(df_all.head())