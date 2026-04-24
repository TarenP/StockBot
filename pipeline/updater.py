"""
Live data updater.
Fetches latest OHLCV from yfinance for the trained universe only
and appends new rows to the master parquet.
"""

import time
import logging
import os
import sys
import requests
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from math import ceil

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from pipeline.checkpoints import load_checkpoint_asset_list
from pipeline.universe_resolver import (
    constrain_to_configured_universe,
    filter_candidate_tickers,
    get_investable_universe_filters,
    get_universe_mode,
    is_benchmark_constrained_mode,
    load_typed_config,
    normalize_ticker,
    normalize_tickers,
    resolve_configured_universe,
)

logger = logging.getLogger(__name__)

# Silence yfinance's own error output entirely
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


@contextmanager
def _suppress_stderr():
    """Redirect stderr to devnull to swallow yfinance's direct print() errors."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

PARQUET_PATH = Path("MasterDS/stooq_panel.parquet")
WATCHLIST_PATH = Path("broker/state/watchlist.csv")
CHUNK_SIZE   = 50   # tickers per yfinance batch request
_DEFAULT_BENCHMARK_SYMBOLS = ["SPY"]


def _normalize_ticker(ticker: str) -> str:
    return normalize_ticker(ticker)


def _is_bootstrap_symbol(ticker: str) -> bool:
    ticker = _normalize_ticker(ticker)
    if not ticker:
        return False
    return all(ch.isalnum() or ch in {"-", "."} for ch in ticker)


def _clean_tickers(symbols) -> list[str]:
    return [
        ticker for ticker in normalize_tickers(symbols)
        if _is_bootstrap_symbol(ticker)
    ]


def _extend_unique_tickers(candidates: list[str], seen: set[str], symbols) -> None:
    for ticker in _clean_tickers(symbols):
        if ticker in seen:
            continue
        seen.add(ticker)
        candidates.append(ticker)


def _benchmark_symbols(config: dict | None = None) -> list[str]:
    cfg = config or {}
    raw = cfg.get("benchmark_symbols", _DEFAULT_BENCHMARK_SYMBOLS)
    if isinstance(raw, str):
        raw = [part for part in raw.replace(";", ",").split(",") if part.strip()]
    return _clean_tickers(raw)


def _bootstrap_universe(target_size: int = 1500) -> list[str]:
    """
    Build an initial candidate universe for a fresh install.

    Sources (in priority order):
    1. Wikipedia S&P 500 / 400 / 600 lists  (live, ~1500 tickers total)
    2. Finviz screener               (live breadth, sorted by volume)
    3. Yahoo trending                (live supplement)
    4. Static sector map             (fallback)
    5. Hardcoded liquid list         (last-resort fallback)
    """
    from broker.sectors import get_cached_sector_map
    target_size = max(int(target_size or 0), 250)
    candidate_target = max(target_size * 2, target_size + 250)

    # ── Static fallback: ~600 highly liquid US tickers ────────────────────────
    STATIC_UNIVERSE = [
        # Technology
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","INTC","QCOM","TXN",
        "AMAT","LRCX","KLAC","MU","ADI","MCHP","SNPS","CDNS","FTNT","PANW",
        "CRWD","ZS","OKTA","DDOG","NET","SNOW","MDB","TEAM","HUBS","NOW",
        "WDAY","ADSK","ANSS","PTC","EPAM","CTSH","ACN","IBM","HPQ","HPE",
        "DELL","STX","WDC","NTAP","PSTG","ANET","JNPR","CSCO","FFIV","AKAM",
        "INTU","EBAY","ETSY","SHOP","COIN","HOOD","SOFI","AFRM","PLTR","GTLB",
        "CFLT","ESTC","SPLK","APPN","PEGA","NXPI","SWKS","QRVO","CRUS","SLAB",
        "DIOD","AMBA","ALGM","MPWR","SITM","ONTO","FORM","ACLS","ICHR","UCTT",
        "MKSI","COHU","WOLF","OLED","RMBS","LITE","IIVI","COHR","VIAV","CIEN",
        "ADTRAN","CALX","COMM","RBBN","LUMN","FYBR","KEYS","TRMB","ITRI","LDOS",
        "SAIC","CACI","BAH","SMCI","INFN","AKAM","CDNS","SNPS","ANSS","PTC",
        # Communication Services
        "META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR",
        "PARA","WBD","FOXA","FOX","OMC","IPG","TTWO","EA","RBLX","SNAP",
        "PINS","MTCH","ZM","TWLO","DOCU","YELP","IAC","LYV","NWSA","NWS",
        "NYT","AMCX","SIRI","NXST","SBGI","GTN","TEGNA",
        # Consumer Discretionary
        "AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","LOW","BKNG","MAR",
        "HLT","RCL","CCL","NCLH","LVS","MGM","WYNN","F","GM","RIVN",
        "LCID","APTV","LEA","BWA","LKQ","AZO","ORLY","AAP","ULTA","LULU",
        "PVH","RL","TPR","VFC","HBI","GPS","ANF","URBN","ROST","BURL",
        "DG","DLTR","WMT","TGT","COST","KR","SFM","BBY","WSM","RH",
        "POOL","FOXF","LCII","CVCO","SKY","YETI","ONON","DECK","CROX","SKX",
        "WING","SHAK","TXRH","DENN","JACK","CAKE","DINE","EAT","DRI","BLMN",
        "RUTH","CBRL","BJRI","NDLS","PZZA","RRGB","DASH","UBER","LYFT","ABNB",
        "EXPE","TRIP","OPEN","RDFN","COOP","PFSI","UWMC","RKT",
        # Consumer Staples
        "PG","KO","PEP","PM","MO","MDLZ","KHC","GIS","K","CPB",
        "CAG","SJM","HRL","MKC","CLX","CHD","CL","EL","KVUE","KMB",
        "SYY","PFGC","CHEF","USFD","CASY","WEIS","SPTN","ANDE","LANC","JJSF",
        "HAIN","FRPT","BYND","CELH","MNST","KDP","COKE","FIZZ","BROS","DNUT",
        "TSN","PPC","CALM","INGR","GPRE","MGPI",
        # Health Care
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "GILD","VRTX","REGN","BIIB","MRNA","BNTX","PFE","CVS","CI","HUM",
        "ELV","CNC","MOH","HCA","THC","UHS","ENSG","AMED","ALNY","INCY",
        "EXEL","HALO","JAZZ","PRGO","CTLT","IQV","CRL","MEDP","ICLR","CRSP",
        "BEAM","EDIT","NTLA","PACB","ILMN","NTRA","EXAS","VEEV","DXCM","PODD",
        "HOLX","IDXX","MASI","ISRG","ALGN","OMCL","PDCO","HSIC","NVST","IRTC",
        "ACHC","ADUS","LHCG","SGRY","SEM","USPH","CCRN","HCSG","GDRX","HIMS",
        "OSCR","ALHC","PRVA","TDOC","AMWL","RXRX","SDGR","TNDM","NSTG","RGEN",
        # Financials
        "BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP",
        "V","MA","PYPL","FIS","FISV","GPN","WEX","EVTC","FOUR","ICE",
        "CME","CBOE","NDAQ","MKTX","LPLA","RJF","SF","PIPR","EVR","PNC",
        "USB","TFC","MTB","CFG","HBAN","RF","KEY","FITB","ZION","CMA",
        "WAL","FHN","SNV","IBOC","BOKF","UMBF","MET","PRU","AFL","ALL",
        "TRV","CB","AIG","HIG","LNC","UNM","PFG","VOYA","EQH","GL",
        "RGA","AIZ","CINF","ERIE","WRB","ACGL","RNR","MKL","KMPR","SIGI",
        "AON","MMC","WTW","BRO","AJG","SPGI","MCO","MSCI","VRSK","DNB",
        "EFX","TRU","FICO","ENVA","WRLD","OMF","CACC","PRAA","ECPG","ALLY",
        "COF","DFS","SYF","NWLI","CRBG","ARGO","NAVG","JRVR","GSHD","RYAN",
        "QCRH","FFIN","TCBI","IBTX","SBCF","SFNC","HTLF","BANF","CVBF","WAFD",
        "COLB","PACW","BANC","HOPE","HAFC","TRMK","FBMS","HOMB","SFBS","FULT",
        "WSFS","TBBK","INDB","EGBN","WASH","NYCB","DCOM",
        # Industrials
        "CAT","DE","HON","UPS","RTX","LMT","BA","GE","MMM","EMR",
        "ETN","PH","ROK","AME","FTV","GNRC","XYL","RRX","CARR","OTIS",
        "TT","JCI","ALLE","AZEK","TREX","MAS","SWK","SNA","GWW","MSC",
        "FAST","GPC","NDSN","ITW","DOV","IR","IDEX","ROP","VRSK","CSGP",
        "CPRT","EXPD","XPO","SAIA","ODFL","JBHT","CHRW","LSTR","FDX","DAL",
        "UAL","AAL","LUV","ALK","JBLU","HA","SKYW","ATSG","ECHO","HUBG",
        "WERN","HTLD","MRTN","ARCB","TFII","GXO","RXO","SNDR","CVLG","NOC",
        "GD","HII","TDG","HEI","KTOS","BWXT","CW","MOOG","AXON","CACI",
        "SAIC","LDOS","BAH","NDSN","FLOW","IDEX","WATTS","ENPRO","AMETEK","AME",
        "APOG","ARCOSA","AAON","AZEK","TREX","UFPI","BCC","LPX","OSB","WY",
        # Energy
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","DVN",
        "FANG","OXY","HES","APA","MRO","HAL","BKR","NOV","CIVI","SM",
        "MTDR","CRGY","VTLE","CHRD","TALO","SBOW","CTRA","OVV","VNOM","BSM",
        "TRGP","WES","MPLX","EPD","ET","MMP","PAA","PAGP","ENLC","DCP",
        "CEQP","KMI","OKE","WMB","LNG","TELL","CLNE","AMRC","GEVO","REGI",
        "RIG","VAL","DO","NE","BORR","PTEN","HP","NBR","NINE","KLXE",
        "OIS","DNOW","MRC","BOOM","LBRT","PUMP",
        # Materials
        "LIN","APD","SHW","ECL","PPG","NEM","FCX","NUE","STLD","RS",
        "CMC","ATI","AA","CENX","KALU","ARNC","HWM","MTRN","HAYN","MLM",
        "VMC","EXP","SUM","USCR","USLM","ROCK","BCPC","TROX","VNTR","ASIX",
        "CC","OLIN","WLK","LYB","HUN","EMN","CE","AVNT","PKG","IP",
        "WRK","SEE","SLGN","BERY","SON","PTVE","RANPAK","UFPI","BCC","LPX",
        "WY","PCH","RYN","MP","GOLD","AEM","KGC","AGI","EGO","PAAS",
        "SILV","MAG","SSRM","CDE","HL","EXK",
        # Real Estate
        "PLD","AMT","EQIX","CCI","SPG","O","VICI","WPC","NNN","EXR",
        "CUBE","PSA","AVB","EQR","ESS","MAA","UDR","CPT","WELL","VTR",
        "PEAK","DOC","MPW","SBRA","NHI","OHI","CTRE","LTC","COLD","STAG",
        "REXR","EGP","TRNO","ILPT","PLYM","KIM","REG","BRX","ROIC","SITC",
        "SLG","VNO","BXP","HIW","CUZ","PDM","EQC","PGRE","ESRT","APLE",
        "PK","RHP","HST","SHO","SAFE","LADR","BXMT","GPMT","TRTX","KREF",
        # Utilities
        "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","ED","XEL",
        "WEC","ES","ETR","FE","PPL","CMS","NI","AES","LNT","EVRG",
        "PNW","OGE","NWE","AVA","IDA","POR","MGEE","OTTR","SJW","MSEX",
        "AWK","AWR","ATO","BKH","CLECO","UTL",
        # ETFs (market proxies for context features)
        "SPY","QQQ","IWM","DIA","VTI","VOO","IVV","GLD","SLV","USO",
        "XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC",
    ]

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(symbols) -> None:
        _extend_unique_tickers(candidates, seen, symbols)

    # 1. Wikipedia S&P 500 (most reliable live source)
    for wiki_url, label in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600"),
    ]:
        try:
            tables = pd.read_html(wiki_url)
            # Symbol column may be named differently across Wikipedia table versions
            for tbl in tables:
                for col in ("Symbol", "Ticker", "Ticker symbol"):
                    if col in tbl.columns:
                        _add(tbl[col].dropna().tolist())
                        logger.info("Wikipedia %s: now at %d tickers", label, len(candidates))
                        break
        except Exception as exc:
            logger.debug("Wikipedia %s fetch failed: %s", label, exc)

    # 2. Finviz screener (main live breadth source)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
    empty_pages = 0
    max_pages = min(max(ceil(candidate_target / 20), 5), 250)
    for page in range(max_pages):
        start_row = 1 + (page * 20)
        url = f"https://finviz.com/screener.ashx?v=111&o=-volume&r={start_row}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            page_symbols = [
                cell.get_text(strip=True).upper()
                for cell in soup.find_all("a", class_="screener-link-primary")
            ]
            before = len(candidates)
            _add(page_symbols)
            if len(candidates) == before:
                empty_pages += 1
                if empty_pages >= 3:
                    break
            else:
                empty_pages = 0
            if len(candidates) >= candidate_target:
                break
        except Exception as exc:
            logger.debug("Finviz bootstrap scrape failed on page %d: %s", page + 1, exc)
            if page >= 2 and len(candidates) >= target_size:
                break
        time.sleep(0.15)

    # 3. Yahoo trending supplement
    if len(candidates) < target_size:
        try:
            response = requests.get(
                "https://finance.yahoo.com/trending-tickers",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")
                _add(a["data-symbol"] for a in soup.find_all("a", {"data-symbol": True}))
        except Exception as exc:
            logger.debug("Yahoo trending bootstrap supplement failed: %s", exc)

    # 4. Static sector map fallback
    _add(get_cached_sector_map().keys())

    # 5. Hardcoded liquid tickers only if live sources came up short
    if len(candidates) < target_size:
        _add(STATIC_UNIVERSE)

    logger.info("Bootstrapped %d candidate tickers for initial market download.", len(candidates))
    return candidates


def _load_trained_universe(save_dir: str = "models") -> list[str] | None:
    """
    Read the universe (top_n tickers) from the best available checkpoint.
    Falls back to None if no checkpoint found.
    """
    return load_checkpoint_asset_list(save_dir=save_dir)


def _load_parquet_universe(
    max_stale_days: int = 30,
) -> list[str]:
    if not PARQUET_PATH.exists():
        return []
    try:
        df = pd.read_parquet(PARQUET_PATH, columns=["ticker"])
    except Exception as exc:
        logger.debug("Could not load parquet universe: %s", exc)
        return []
    if "ticker" not in df.columns:
        return []

    if "date" not in df.columns:
        try:
            df = pd.read_parquet(PARQUET_PATH)
        except Exception as exc:
            logger.debug("Could not load dated parquet universe: %s", exc)
            return _clean_tickers(df["ticker"].dropna().tolist())

    if "date" not in df.columns:
        return _clean_tickers(df["ticker"].dropna().tolist())

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.dropna(subset=["date", "ticker"])
    if frame.empty:
        return []

    latest_date = frame["date"].max()
    cutoff = latest_date - pd.Timedelta(days=int(max_stale_days))
    recent = frame.loc[frame["date"] >= cutoff, "ticker"].dropna().tolist()
    return _clean_tickers(recent)


def _load_watchlist_universe() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return []
    try:
        df = pd.read_csv(WATCHLIST_PATH)
    except Exception as exc:
        logger.debug("Could not load watchlist universe: %s", exc)
        return []
    if "ticker" not in df.columns:
        return []
    return _clean_tickers(df["ticker"].dropna().tolist())


def get_live_universe(
    preferred: list[str] | None = None,
    save_dir: str = "models",
    target_size: int | None = None,
    config: dict | None = None,
    mode: str | None = None,
    as_of_date=None,
) -> list[str]:
    """
    Resolve the configured live ticker universe.

    Benchmark-constrained modes such as ``sp500`` return the exact membership
    snapshot. Broad mode (``tradable_us``) unions local sources and bootstrap
    candidates, then filters obvious non-equity / malformed instruments.
    """
    cfg = config if config is not None else load_typed_config()
    resolved_mode = (mode or get_universe_mode(cfg)).strip().lower()

    if is_benchmark_constrained_mode(resolved_mode):
        resolved = resolve_configured_universe(
            as_of_date=as_of_date,
            save_dir=save_dir,
            config={**cfg, "universe_mode": resolved_mode},
        )
        preferred_tickers = constrain_to_configured_universe(
            preferred,
            as_of_date=as_of_date,
            save_dir=save_dir,
            config={**cfg, "universe_mode": resolved_mode},
        )
        if not preferred_tickers:
            logger.info(
                "Resolved live universe: %d tickers (mode=%s)",
                len(resolved),
                resolved_mode,
            )
            return resolved

        preferred_set = set(preferred_tickers)
        ordered = preferred_tickers + [ticker for ticker in resolved if ticker not in preferred_set]
        logger.info(
            "Resolved live universe: %d tickers (mode=%s, preferred=%d)",
            len(ordered),
            resolved_mode,
            len(preferred_tickers),
        )
        return ordered

    broad_target = int(
        target_size
        or cfg.get("live_target_size")
        or max(int(cfg.get("top_n", 500)) * 3, 1000)
    )
    filters = get_investable_universe_filters(cfg)
    candidates: list[str] = []
    seen: set[str] = set()

    def _extend(symbols) -> None:
        _extend_unique_tickers(candidates, seen, symbols)

    _extend(preferred)
    _extend(_load_trained_universe(save_dir))
    _extend(_load_parquet_universe(max_stale_days=int(filters["max_stale_days"])))
    _extend(_load_watchlist_universe())
    if len(candidates) < broad_target:
        _extend(_bootstrap_universe(broad_target))

    filtered = filter_candidate_tickers(candidates, config=cfg)
    if broad_target > 0:
        filtered = filtered[:broad_target]

    logger.info(
        "Resolved live universe: %d tickers (mode=%s, target=%d, preferred=%d)",
        len(filtered),
        resolved_mode,
        broad_target,
        len(_clean_tickers(preferred)),
    )
    return filtered


def _prune_to_allowed_universe(
    df: pd.DataFrame,
    allowed_tickers: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        return df, []

    allowed = set(_clean_tickers(allowed_tickers))
    if not allowed:
        return df, []

    df_reset = df.reset_index()
    if "ticker" not in df_reset.columns:
        return df, []

    tickers = df_reset["ticker"].astype(str).str.upper()
    removed = sorted(set(_clean_tickers(tickers.tolist())) - allowed)
    if not removed:
        return df, []

    pruned = df_reset[tickers.isin(allowed)].copy()
    pruned["date"] = pd.to_datetime(pruned["date"]).dt.normalize()
    pruned = pruned.set_index("date").sort_index()
    return pruned, removed


def _update_parquet_configured(
    universe: list[str] | None = None,
    force_full_refresh: bool = False,
    save_dir: str = "models",
    config: dict | None = None,
    universe_mode: str | None = None,
    prune_to_resolved_universe: bool | None = None,
    as_of_date=None,
) -> int:
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        last_date = pd.to_datetime(existing.index).max().normalize()
    else:
        existing = None
        last_date = None

    cfg = config if config is not None else load_typed_config()
    resolved_mode = (universe_mode or get_universe_mode(cfg)).strip().lower()
    if is_benchmark_constrained_mode(resolved_mode):
        resolved_universe = resolve_configured_universe(
            as_of_date=as_of_date,
            save_dir=save_dir,
            config={**cfg, "universe_mode": resolved_mode},
        )
        if prune_to_resolved_universe is None:
            prune_to_resolved_universe = bool(cfg.get("prune_universe_rows", False))

        if universe is None:
            effective_universe = list(resolved_universe)
        else:
            explicit_universe = _clean_tickers(universe)
            effective_universe = constrain_to_configured_universe(
                explicit_universe,
                as_of_date=as_of_date,
                save_dir=save_dir,
                config={**cfg, "universe_mode": resolved_mode},
            )
            dropped = sorted(set(explicit_universe) - set(effective_universe))
            if dropped:
                logger.warning(
                    "Dropped %d ticker(s) outside configured %s universe: %s%s",
                    len(dropped),
                    resolved_mode,
                    ", ".join(dropped[:10]),
                    f" ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else "",
                )
    else:
        resolved_universe = []
        if prune_to_resolved_universe is None:
            prune_to_resolved_universe = bool(cfg.get("prune_universe_rows", False))
        effective_universe = get_live_universe(
            preferred=universe,
            save_dir=save_dir,
            target_size=int(cfg.get("live_target_size", max(int(cfg.get("top_n", 500)) * 3, 1000))),
            config={**cfg, "universe_mode": resolved_mode},
            mode=resolved_mode,
            as_of_date=as_of_date,
        )

    if not effective_universe:
        raise ValueError(
            f"Configured universe for mode={resolved_mode!r} resolved to zero tickers."
        )

    benchmark_symbols = _benchmark_symbols(cfg)
    if benchmark_symbols:
        combined_universe: list[str] = []
        seen_combined: set[str] = set()
        _extend_unique_tickers(combined_universe, seen_combined, effective_universe)
        _extend_unique_tickers(combined_universe, seen_combined, benchmark_symbols)
        effective_universe = combined_universe

    logger.info(
        "Updating %d tickers (mode=%s).",
        len(effective_universe),
        resolved_mode,
    )

    existing_tickers = set(existing["ticker"].unique()) if existing is not None else set()
    new_tickers = [ticker for ticker in effective_universe if ticker not in existing_tickers]
    known_tickers = [ticker for ticker in effective_universe if ticker in existing_tickers]

    if new_tickers:
        logger.info("  %d new tickers â€” fetching full history from 2010...", len(new_tickers))
    if known_tickers:
        logger.info("  %d existing tickers â€” fetching recent data...", len(known_tickers))

    if last_date is not None:
        if force_full_refresh:
            fetch_start = (last_date - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        fetch_start = "2010-01-01"

    fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y-%m-%d")

    all_new_rows: list[pd.DataFrame] = []
    if known_tickers and fetch_start < today_str:
        logger.info("Fetching from %s to %s...", fetch_start, fetch_end)
        df_known = _fetch_yfinance(known_tickers, fetch_start, fetch_end)
        if not df_known.empty:
            all_new_rows.append(df_known)

    if new_tickers:
        logger.info("Fetching full history for %d new tickers...", len(new_tickers))
        df_new_tickers = _fetch_yfinance(new_tickers, "2010-01-01", fetch_end)
        if not df_new_tickers.empty:
            all_new_rows.append(df_new_tickers)
            logger.info(
                "  Got %d rows for %d new tickers.",
                len(df_new_tickers),
                df_new_tickers["ticker"].nunique(),
            )

    if not all_new_rows:
        logger.info("No new price data returned.")
        return 0

    new_df = pd.concat(all_new_rows, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()
    new_df = new_df.set_index("date").sort_index()

    if existing is not None:
        if force_full_refresh:
            cutoff = pd.to_datetime(fetch_start)
            existing = existing[pd.to_datetime(existing.index) < cutoff]
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.set_index("date").sort_index()

    if prune_to_resolved_universe and resolved_universe:
        combined, removed_non_members = _prune_to_allowed_universe(combined, resolved_universe)
        if removed_non_members:
            logger.info(
                "Pruned %d non-member ticker(s) from parquet for mode=%s: %s%s",
                len(removed_non_members),
                resolved_mode,
                ", ".join(removed_non_members[:10]),
                f" ... (+{len(removed_non_members) - 10} more)"
                if len(removed_non_members) > 10 else "",
            )

    if bool(cfg.get("prune_stale_history", False)):
        filters = get_investable_universe_filters(cfg)
        combined, pruned = prune_stale_tickers(
            combined,
            stale_days=int(filters["max_stale_days"]),
            min_history_days=int(filters["min_history_days"]),
        )
        if pruned:
            logger.info(
                "Parquet pruned: %d stale ticker(s) removed, %d remaining.",
                len(pruned),
                combined["ticker"].nunique(),
            )

    combined.to_parquet(PARQUET_PATH, index=True)

    n_new = len(new_df)
    logger.info("Done. %d new rows added. Total: %d", n_new, len(combined))
    return n_new


def prune_stale_tickers(
    df: pd.DataFrame,
    stale_days: int = 30,
    min_history_days: int = 252,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove tickers from the parquet that are likely delisted or inactive.

    A ticker is pruned if:
    - Its most recent date is more than `stale_days` trading days behind
      the most recent date in the whole dataset (i.e. it stopped trading), OR
    - It has fewer than `min_history_days` rows total (too thin to be useful).

    Returns (pruned_df, list_of_removed_tickers).
    """
    if df.empty:
        return df, []

    df_reset = df.reset_index()
    df_reset["date"] = pd.to_datetime(df_reset["date"])
    global_max_date = df_reset["date"].max()
    cutoff_date = global_max_date - pd.Timedelta(days=stale_days)

    ticker_stats = df_reset.groupby("ticker").agg(
        last_date=("date", "max"),
        n_rows=("date", "count"),
    )

    stale = ticker_stats[
        (ticker_stats["last_date"] < cutoff_date) |
        (ticker_stats["n_rows"] < min_history_days)
    ].index.tolist()

    if stale:
        logger.info(
            "Pruning %d stale/thin tickers (last trade > %d days ago or < %d rows): %s%s",
            len(stale), stale_days, min_history_days,
            ", ".join(stale[:10]),
            f" ... (+{len(stale)-10} more)" if len(stale) > 10 else "",
        )
        df_reset = df_reset[~df_reset["ticker"].isin(stale)]
        df = df_reset.set_index("date").sort_index()

    return df, stale


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV for a list of tickers via yfinance.
    Silently skips delisted / missing tickers.
    Returns flat DataFrame with columns [date, open, high, low, close, volume, ticker].
    """
    rows = []
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    pbar = tqdm(chunks, desc="Fetching prices", unit="batch", colour="cyan",
                dynamic_ncols=True)
    for chunk in pbar:
        pbar.set_postfix(first=chunk[0])
        try:
            with _suppress_stderr():
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in chunk:
                    try:
                        t_df = raw.xs(ticker, axis=1, level=1)
                        t_df = t_df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
                        t_df = t_df.dropna(subset=["close"])
                        t_df = t_df[t_df["close"] > 0]
                        if t_df.empty:
                            continue
                        t_df["ticker"] = ticker.upper()
                        t_df.index = pd.to_datetime(t_df.index).normalize()
                        t_df.index.name = "date"
                        rows.append(t_df.reset_index())
                    except (KeyError, Exception):
                        pass   # delisted or missing — skip silently
            else:
                # Single ticker returned
                raw = raw[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
                raw = raw.dropna(subset=["close"])
                raw = raw[raw["close"] > 0]
                if not raw.empty:
                    raw["ticker"] = chunk[0].upper()
                    raw.index = pd.to_datetime(raw.index).normalize()
                    raw.index.name = "date"
                    rows.append(raw.reset_index())

        except Exception as e:
            # Log at debug level — delisted errors are noisy and expected
            logger.debug(f"Skipping chunk {chunk[:2]}: {e}")

        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def update_parquet(
    universe: list[str] | None = None,
    force_full_refresh: bool = False,
    save_dir: str = "models",
    bootstrap_universe_size: int | None = None,
    config: dict | None = None,
    universe_mode: str | None = None,
    prune_to_resolved_universe: bool | None = None,
    as_of_date=None,
) -> int:
    """
    Append new trading days to the master parquet.

    Only updates tickers in `universe`. If universe is None, it resolves the
    configured benchmark universe.

    Args:
        universe:           explicit list of tickers to update
        force_full_refresh: re-download the last 30 days (fixes gaps)
        save_dir:           where to look for checkpoints
        bootstrap_universe_size:
            retained for backward compatibility; ignored by the configured
            benchmark-aware implementation.

    Returns:
        Number of new rows appended.
    """
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _update_parquet_configured(
        universe=universe,
        force_full_refresh=force_full_refresh,
        save_dir=save_dir,
        config=config,
        universe_mode=universe_mode,
        prune_to_resolved_universe=prune_to_resolved_universe,
        as_of_date=as_of_date,
    )

    # ── Load existing parquet ────────────────────────────────────────────────
    if PARQUET_PATH.exists():
        existing  = pd.read_parquet(PARQUET_PATH)
        last_date = pd.to_datetime(existing.index).max().normalize()
    else:
        existing  = None
        last_date = None

    # ── Resolve universe — prefer checkpoint, then explicit arg, then warn ───
    if universe is None:
        universe = _load_trained_universe(save_dir)

    if universe is None and existing is not None:
        logger.warning(
            "No checkpoint found to determine universe. "
            "Falling back to all tickers in parquet — this may be slow. "
            "Run --mode train first to set a universe."
        )
        universe = existing["ticker"].unique().tolist()
    elif universe is None:
        bootstrap_target = int(bootstrap_universe_size or 1500)
        logger.warning(
            "No checkpoint, no explicit universe, and no existing parquet found. "
            "Bootstrapping an initial download universe (~%d symbols).",
            bootstrap_target,
        )
        universe = _bootstrap_universe(bootstrap_target)
        if not universe:
            raise ValueError(
                "Could not bootstrap an initial ticker universe. "
                "Check network access and retry --mode update or --mode train."
            )

    logger.info(f"Updating {len(universe)} tickers.")

    # ── Split into existing vs brand-new tickers ──────────────────────────────
    existing_tickers = set(existing["ticker"].unique()) if existing is not None else set()
    new_tickers      = [t for t in universe if t not in existing_tickers]
    known_tickers    = [t for t in universe if t in existing_tickers]

    if new_tickers:
        logger.info(f"  {len(new_tickers)} new tickers — fetching full history from 2010...")
    if known_tickers:
        logger.info(f"  {len(known_tickers)} existing tickers — fetching recent data...")

    # ── Date range for existing tickers ──────────────────────────────────────
    if last_date is not None:
        if force_full_refresh:
            fetch_start = (last_date - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        fetch_start = "2010-01-01"

    fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y-%m-%d")

    all_new_rows = []

    # ── Fetch existing tickers (recent only) ──────────────────────────────────
    if known_tickers and fetch_start < today_str:
        logger.info(f"Fetching from {fetch_start} to {fetch_end}...")
        df_known = _fetch_yfinance(known_tickers, fetch_start, fetch_end)
        if not df_known.empty:
            all_new_rows.append(df_known)

    # ── Fetch new tickers (full history) ─────────────────────────────────────
    if new_tickers:
        logger.info(f"Fetching full history for {len(new_tickers)} new tickers...")
        df_new_tickers = _fetch_yfinance(new_tickers, "2010-01-01", fetch_end)
        if not df_new_tickers.empty:
            all_new_rows.append(df_new_tickers)
            logger.info(f"  Got {len(df_new_tickers):,} rows for {df_new_tickers['ticker'].nunique()} new tickers.")

    if not all_new_rows:
        logger.info("No new price data returned.")
        return 0

    new_df = pd.concat(all_new_rows, ignore_index=True)

    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()
    new_df = new_df.set_index("date").sort_index()

    # ── Merge with existing ──────────────────────────────────────────────────
    if existing is not None:
        if force_full_refresh:
            cutoff   = pd.to_datetime(fetch_start)
            existing = existing[pd.to_datetime(existing.index) < cutoff]
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.set_index("date").sort_index()

    # ── Prune stale/delisted tickers ─────────────────────────────────────────
    combined, pruned = prune_stale_tickers(combined)
    if pruned:
        logger.info("Parquet pruned: %d tickers removed, %d remaining.", len(pruned), combined["ticker"].nunique())

    combined.to_parquet(PARQUET_PATH, index=True)

    n_new = len(new_df)
    logger.info(f"Done. {n_new} new rows added. Total: {len(combined):,}")
    return n_new
