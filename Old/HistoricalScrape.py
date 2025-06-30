# -----------------------------  INSTALLS  ----------------------------------
# pip install sec-edgar-downloader yfinance ta pandas lxml python-dateutil

import os, glob, re, datetime as dt, pandas as pd, numpy as np
import yfinance as yf
import ta
from lxml import etree
from dateutil import parser as dparse
from sec_edgar_downloader import Downloader

# ---------------------------------------------------------------------------
# 0.  CONFIG
# ---------------------------------------------------------------------------
TICKER   = "AAPL"
FORMS    = ("10-K", "10-Q")          # quarterly + annual
CSV_SENT = "Sentiment/analyst_ratings_with_sentiment.csv"

# ---------------------------------------------------------------------------
# 1.  FIND DATE RANGE FROM SENTIMENT CSV
# ---------------------------------------------------------------------------
def csv_date_range(csv_path: str):
    min_dt, max_dt = None, None
    with open(csv_path, encoding="utf-8") as f:
        hdr = next(f)                # header
        for line in f:
            # assumes the column is literally named 'date'
            date_str = line.split(",")[0]
            dt_parsed = dparse.parse(date_str)
            if min_dt is None or dt_parsed < min_dt: min_dt = dt_parsed
            if max_dt is None or dt_parsed > max_dt: max_dt = dt_parsed
    return min_dt.date(), max_dt.date()

date_start, date_end = csv_date_range(CSV_SENT)
print("Sentiment file covers", date_start, "→", date_end)

# ---------------------------------------------------------------------------
# 2.  GAAP TAGS
# ---------------------------------------------------------------------------
REVENUE_TAGS = [
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap:SalesRevenueNet",
    "us-gaap:NetSales",
    "us-gaap:Revenues",
]
GAAP_TAGS = {
    "NetIncome"      : "us-gaap:NetIncomeLoss",
    "EPS_Basic"      : "us-gaap:EarningsPerShareBasic",
    "EPS_Diluted"    : "us-gaap:EarningsPerShareDiluted",
    "Assets"         : "us-gaap:Assets",
    "Liabilities"    : "us-gaap:Liabilities",
    "Equity"         : "us-gaap:StockholdersEquity",
    "GrossProfit"    : "us-gaap:GrossProfit",
    "OperatingIncome": "us-gaap:OperatingIncomeLoss",
}
IX_NS = {"ix":"http://www.xbrl.org/2013/inlineXBRL"}

# ---------------------------------------------------------------------------
# 3.  HELPERS FOR XBRL EXTRACTION
# ---------------------------------------------------------------------------
def find_fact(root, tag_list):
    for tag in (tag_list if isinstance(tag_list, list) else [tag_list]):
        el = root.xpath(f".//ix:*[@name='{tag}']", namespaces=IX_NS)
        if el: return el[0]
        lname = tag.split(":",1)[1]
        el = root.xpath(f"//*[local-name()='{lname}']")
        if el: return el[0]
    return None

def numeric(el):
    if el is None: return None
    txt = el.text.replace(",", "").strip()
    try: val = float(txt)
    except ValueError: return txt
    unit = (el.get("unitRef") or "").lower()
    if unit.endswith(("usdm","usdmm")):     return val * 1_000_000
    if unit.endswith(("usdth","usd000")):   return val * 1_000
    if el.tag.startswith("{http://www.xbrl.org/2013/inlineXBRL}"):
        dec = el.get("decimals")
        if dec in ("-6","-3"):
            return val * (1_000_000 if dec=="-6" else 1_000)
    return val

# ---------------------------------------------------------------------------
# 4.  DOWNLOAD & PARSE 10-K + 10-Q
# ---------------------------------------------------------------------------
dl = Downloader(None, "my_email@example.com")
years = range(date_start.year, date_end.year+1)

filing_rows = []
for yr in years:
    for form in FORMS:
        try:
            dl.get(form, TICKER, after=f"{yr}-01-01", before=f"{yr}-12-31", limit=1)
        except Exception: pass

root_dir = f"sec-edgar-filings/{TICKER}"
for form in FORMS:
    for folder in glob.glob(os.path.join(root_dir, form, "*")):
        txt_files = glob.glob(os.path.join(folder,"*.txt"))
        if not txt_files: continue
        txt_path = txt_files[0]
        text     = open(txt_path,"r",encoding="utf-8",errors="ignore").read()

        # filing date from header
        m_date = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", text)
        if not m_date: continue
        filing_dt = dt.datetime.strptime(m_date.group(1), "%Y%m%d%H%M%S").date()

        # XBRL instance
        m_x = re.search(r"(?is)<\s*(?:xbrli:)?xbrl\b.*?</\s*(?:xbrli:)?xbrl>", text)
        if not m_x: continue
        root = etree.fromstring(m_x.group(0).encode("utf-8"),
                                parser=etree.XMLParser(recover=True, ns_clean=True))

        row = {"FilingDate": filing_dt,
               "Form": form,
               "Revenue": numeric(find_fact(root, REVENUE_TAGS))}
        for lbl, tag in GAAP_TAGS.items():
            row[lbl] = numeric(find_fact(root, tag))
        filing_rows.append(row)

df_filings = pd.DataFrame(filing_rows).dropna(subset=["FilingDate"])
df_filings.sort_values("FilingDate", inplace=True)

# ---------------------------------------------------------------------------
# 5.  DAILY CALENDAR, MERGE, FORWARD-FILL
# ---------------------------------------------------------------------------
biz_days = pd.date_range(date_start, date_end, freq="B")
daily = pd.DataFrame(index=biz_days)

daily = daily.merge(df_filings.set_index("FilingDate"),
                    how="left", left_index=True, right_index=True).ffill()

# ---------------------------------------------------------------------------
# 6.  ADD MARKET DATA & TECHNICALS
# ---------------------------------------------------------------------------
px = yf.download(TICKER, start=str(date_start), end=str(date_end+dt.timedelta(days=1)),
                 auto_adjust=True)
daily = daily.join(px[["Adj Close","Volume"]], how="left")

daily["Return_1d"]      = daily["Adj Close"].pct_change()
daily["SMA_20"]         = ta.trend.sma_indicator(daily["Adj Close"], window=20)
daily["Volatility_20"]  = daily["Return_1d"].rolling(20).std()

# ---------------------------------------------------------------------------
# 7.  SENTIMENT AGGREGATION
# ---------------------------------------------------------------------------
sent = (pd.read_csv(CSV_SENT, parse_dates=["date"])
          .assign(date=lambda d: d["date"].dt.date)
          .groupby("date")
          .agg(art_cnt=("sentiment","size"),
               pos_avg=("pos_score","mean"),
               neg_avg=("neg_score","mean"),
               neu_avg=("neutral_score","mean")))
sent.index = pd.to_datetime(sent.index)
daily = daily.join(sent, how="left").fillna({"art_cnt":0, "pos_avg":0, "neg_avg":0, "neu_avg":0})

# ---------------------------------------------------------------------------
# 8.  FUNDAMENTAL RATIOS & GROWTH
# ---------------------------------------------------------------------------
daily["ROA"]            = daily["NetIncome"]      / daily["Assets"]
daily["Gross_Margin"]   = daily["GrossProfit"]    / daily["Revenue"]
daily["Op_Margin"]      = daily["OperatingIncome"]/ daily["Revenue"]
daily["Debt_to_Equity"] = daily["Liabilities"]    / daily["Equity"]

for col in ["Revenue","NetIncome"]:
    daily[col+"_YoY"] = daily[col].pct_change(252)

# ---------------------------------------------------------------------------
# 9.  FINAL FEATURE FRAME
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # market
    "Adj Close","Volume","Return_1d","SMA_20","Volatility_20",
    # sentiment
    "art_cnt","pos_avg","neg_avg","neu_avg",
]
df_features = daily[FEATURE_COLS].dropna()
pd.set_option("display.float_format", "{:,.3f}".format)
print(df_features.head())
print(df_features.tail())