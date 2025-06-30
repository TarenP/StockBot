"""
aapl_quarterly.py
-----------------
Scrape Apple 10-Q (Q1-Q3) and 10-K (Q4) filings from the SEC, parse GAAP facts,
and produce one clean row per fiscal quarter 2009-2020.

pip install sec-edgar-downloader lxml pandas
"""

import os, glob, re, pandas as pd
from lxml import etree
from sec_edgar_downloader import Downloader

# ─────────────────────────────────────────────────────────────────────────────
# GAAP tags & helpers
# ─────────────────────────────────────────────────────────────────────────────
REVENUE_TAGS = [
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap:SalesRevenueNet",
    "us-gaap:NetSales",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueGoodsNet",
    "us-gaap:RevenueFromSaleOfGoods",
    "us-gaap:SalesRevenueServicesNet",
    "aapl:ProductSales",
    "aapl:ServicesSales",
    "aapl:NetSales",
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
IX_NS = {"ix": "http://www.xbrl.org/2013/inlineXBRL"}

def find_fact(root, tag_or_list):
    """Return the first matching element among tag_or_list (inline first)."""
    for tag in (tag_or_list if isinstance(tag_or_list, list) else [tag_or_list]):
        el = root.xpath(f".//ix:*[@name='{tag}']", namespaces=IX_NS)
        if el:
            return el[0]
        lname = tag.split(":", 1)[1]
        el = root.xpath(f"//*[local-name()='{lname}']")
        if el:
            return el[0]
    return None

def numeric(el):
    """Convert text to float and scale if unit/decimals indicate thousands/millions."""
    if el is None or el.text is None:
        return None
    txt = el.text.replace(",", "").strip()
    try:
        val = float(txt)
    except ValueError:
        return None

    unit = (el.get("unitRef") or "").lower()
    if unit.endswith(("usdm", "usdmm")):
        return val * 1_000_000
    if unit.endswith(("usdth", "usd000")):
        return val * 1_000

    if el.tag.startswith("{http://www.xbrl.org/2013/inlineXBRL}"):
        dec = el.get("decimals")
        if dec == "-6": return val * 1_000_000
        if dec == "-3": return val * 1_000
    return val

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Download Apple 10-Q (Q1-Q3) + 10-K (Q4)  for FY-2009-2020
# ─────────────────────────────────────────────────────────────────────────────
dl = Downloader(None, "Amazingmemc@gmail.com")   # insert your address

# FIXED: Start from 2007 to capture FY2009 Q1 (which occurs in calendar Oct-Dec 2008)
for yr in range(2008, 2021):                      # changed from 2008 to 2007
    print(f"Downloading {yr}...")
    dl.get("10-Q", "AAPL", after=f"{yr}-01-01", before=f"{yr}-12-31", limit=4)
    dl.get("10-K", "AAPL", after=f"{yr}-01-01", before=f"{yr}-12-31", limit=1)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parse every downloaded filing to rows
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for form in ("10-Q", "10-K"):
    for folder in glob.glob(f"sec-edgar-filings/AAPL/{form}/*"):
        # the side-car .txt
        txt_files = glob.glob(os.path.join(folder, "*.txt"))
        if not txt_files:
            continue
        txt = open(txt_files[0], encoding="utf-8", errors="ignore").read()

        # (a) inline <xbrl> fragment
        m = re.search(r"(?is)<\s*(?:xbrli:)?xbrl\b.*?</\s*(?:xbrli:)?xbrl>", txt)
        if m:
            xbrl_bytes = m.group(0).encode("utf-8")
        else:
            # (b) any standalone instance *recursively* (older 10-Qs)
            xmls = (glob.glob(os.path.join(folder, "**/*.xml"),  recursive=True) +
                    glob.glob(os.path.join(folder, "**/*.xbrl"), recursive=True))
            if not xmls:
                continue
            xbrl_bytes = open(xmls[0], "rb").read()

        root = etree.fromstring(
            xbrl_bytes,
            parser=etree.XMLParser(recover=True, ns_clean=True)
        )

        # ── period-end date ────────────────────────────────────────────────
        pe_el = find_fact(root, "dei:DocumentPeriodEndDate")
        if pe_el is None:
            continue
        pe = pd.to_datetime(pe_el.text).date()

        # ── map to Apple fiscal FY/Qtr ─────────────────────────────────────
        if 10 <= pe.month <= 12:   fy, fq = pe.year + 1, "Q1"
        elif 1 <= pe.month <= 3:   fy, fq = pe.year,     "Q2"
        elif 4 <= pe.month <= 6:   fy, fq = pe.year,     "Q3"
        else:                      fy, fq = pe.year,     "Q4"   # from the 10-K

        # ── build data row ────────────────────────────────────────────────
        row = {"FiscalYear": fy, "FiscalQtr": fq, "PeriodEnd": pe,
               "Revenue": numeric(find_fact(root, REVENUE_TAGS))}
        for lbl, tag in GAAP_TAGS.items():
            row[lbl] = numeric(find_fact(root, tag))
        
        # Debug print for missing quarters
        if fy == 2009 and fq in ["Q1", "Q2"]:
            print(f"Found FY2009 {fq}: Revenue={row['Revenue']}, Period={pe}")
        
        rows.append(row)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Deduplicate (keep 1 row per quarter) & sort
# ─────────────────────────────────────────────────────────────────────────────
order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
df = (pd.DataFrame(rows)
        .sort_values(["FiscalYear", "FiscalQtr"],
                     key=lambda s: s.map(order))
        .drop_duplicates(subset=["FiscalYear", "FiscalQtr"], keep="first")
        .reset_index(drop=True))
df["QtrNum"] = df["FiscalQtr"].map({"Q1":1,"Q2":2,"Q3":3,"Q4":4})
df = (df
  .sort_values(["FiscalYear","QtrNum"])
  .drop(columns="QtrNum")
  .reset_index(drop=True))
pd.set_option("display.float_format", "{:,.0f}".format)
print(df.head(20))   # FY-2009 now shows Q1-Q4
print(df.tail(20))   # FY-2019-2020 show Q1-Q4
df.to_csv("AAPL_quarterly_FY2009_2020.csv", index=False)
print("\nSaved → AAPL_quarterly_FY2009_2020.csv")