"""
Shared trading-universe resolution.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_UNIVERSE_MODES = {"tradable_us", "sp500", "sp1500", "custom"}
BENCHMARK_CONSTRAINED_UNIVERSE_MODES = {"sp500", "sp1500", "custom"}
SNAPSHOT_DIRNAME = "universe_snapshots"

_PRICE_PANEL_PATH = Path("MasterDS/stooq_panel.parquet")
_WATCHLIST_PATH = Path("broker/state/watchlist.csv")

_KNOWN_NON_EQUITY_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "GLD", "SLV", "USO",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
}

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
_SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def normalize_tickers(symbols) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols or []:
        ticker = normalize_ticker(symbol)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        cleaned.append(ticker)
    return cleaned


def load_typed_config(path: str = "broker.config") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    cfg: dict[str, object] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split("#")[0].strip()
        lowered = value.lower()
        if lowered == "true":
            parsed: object = True
        elif lowered == "false":
            parsed = False
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        cfg[key.strip()] = parsed
    return cfg


def get_universe_mode(config: dict | None = None, default: str = "tradable_us") -> str:
    cfg = config if config is not None else load_typed_config()
    mode = str(cfg.get("universe_mode", default) or default).strip().lower()
    if mode not in SUPPORTED_UNIVERSE_MODES:
        raise ValueError(
            f"Unsupported universe_mode={mode!r}. "
            f"Expected one of {sorted(SUPPORTED_UNIVERSE_MODES)}."
        )
    return mode


def is_benchmark_constrained_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in BENCHMARK_CONSTRAINED_UNIVERSE_MODES


def get_investable_universe_filters(config: dict | None = None) -> dict[str, float | int]:
    """
    Shared investability filters for live, replay, and training paths.

    These defaults intentionally describe a broad but realistic U.S. common-
    equity universe rather than whatever happens to exist in the local parquet.
    """
    cfg = config if config is not None else load_typed_config()
    return {
        "min_history_days": int(cfg.get("universe_min_history_days", 252)),
        "min_price": float(cfg.get("universe_min_price", 2.0)),
        "min_avg_volume": float(cfg.get("universe_min_avg_volume", 500_000)),
        "max_stale_days": int(cfg.get("universe_max_stale_days", 30)),
    }


def _coerce_snapshot_date(as_of_date=None) -> date:
    if as_of_date is None:
        return date.today()
    if isinstance(as_of_date, pd.Timestamp):
        return as_of_date.to_pydatetime().date()
    if isinstance(as_of_date, datetime):
        return as_of_date.date()
    if isinstance(as_of_date, date):
        return as_of_date
    return pd.Timestamp(as_of_date).to_pydatetime().date()


def _snapshot_dir(save_dir: str = "models") -> Path:
    path = Path(save_dir) / SNAPSHOT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snapshot_path(mode: str, snapshot_date: date, save_dir: str = "models") -> Path:
    return _snapshot_dir(save_dir) / f"{mode}_{snapshot_date.isoformat()}.json"


def _read_snapshot(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read cached universe snapshot %s: %s", path, exc)
        return None

    if isinstance(payload, dict):
        tickers = payload.get("tickers", [])
    else:
        tickers = payload
    cleaned = normalize_tickers(tickers)
    return cleaned or None


def _write_snapshot(
    path: Path,
    mode: str,
    snapshot_date: date,
    tickers: list[str],
    source: str,
) -> None:
    payload = {
        "mode": mode,
        "snapshot_date": snapshot_date.isoformat(),
        "source": source,
        "count": len(tickers),
        "tickers": tickers,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_symbol_column(table: pd.DataFrame) -> list[str]:
    for col in ("Symbol", "Ticker", "Ticker symbol"):
        if col in table.columns:
            return normalize_tickers(table[col].dropna().tolist())
    return []


def _read_local_symbol_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path, columns=["ticker"])
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        logger.debug("Could not read local universe source %s: %s", path, exc)
        return []

    if "ticker" not in df.columns:
        return []
    return normalize_tickers(df["ticker"].dropna().tolist())


def _allow_flag(config: dict | None, key: str, default: bool = False) -> bool:
    cfg = config or {}
    return bool(cfg.get(key, default))


def classify_symbol_structure(ticker: str, config: dict | None = None) -> str | None:
    symbol = normalize_ticker(ticker)
    if not symbol:
        return "empty"
    if any(ch in symbol for ch in {"^", "/", "=", " "}):
        return "malformed"
    if not re.fullmatch(r"[A-Z0-9-]+", symbol):
        return "malformed"
    if len(symbol) > 7:
        return "malformed"

    if "-" in symbol:
        _base, suffix = symbol.rsplit("-", 1)
        suffix = suffix.upper()
        if suffix in {"A", "B", "C", "D"}:
            return None
        if suffix.startswith(("P", "PR")) and not _allow_flag(config, "allow_preferreds"):
            return "preferred"
        if suffix.startswith(("W", "WS", "WT", "U", "R")) and not _allow_flag(config, "allow_warrants"):
            return "warrant"
        if not _allow_flag(config, "allow_structured_instruments"):
            return "structured"

    if len(symbol) == 5 and symbol[-1] in {"F", "Y", "Q"} and not _allow_flag(config, "allow_otc"):
        return "otc"

    if symbol in _KNOWN_NON_EQUITY_SYMBOLS and not _allow_flag(config, "allow_etfs"):
        return "known_non_equity"

    return None


def filter_candidate_tickers(
    symbols,
    config: dict | None = None,
) -> list[str]:
    cfg = config or {}
    filtered: list[str] = []
    removed = Counter()

    for symbol in normalize_tickers(symbols):
        reason = classify_symbol_structure(symbol, config=cfg)
        if reason is not None:
            removed[reason] += 1
            continue
        filtered.append(symbol)

    if removed:
        logger.info(
            "Universe filter removed %d ticker(s): %s",
            sum(removed.values()),
            ", ".join(f"{key}={value}" for key, value in sorted(removed.items())),
        )

    return filtered


def _resolve_wikipedia_list(url: str, label: str) -> list[str]:
    import requests as _requests
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0; +https://github.com)"}
    try:
        resp = _requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
    except Exception as exc:
        raise RuntimeError(f"Could not fetch {label} from Wikipedia: {exc}") from exc
    for table in tables:
        tickers = _extract_symbol_column(table)
        if tickers:
            logger.info("Resolved %s universe from Wikipedia (%d tickers).", label, len(tickers))
            return tickers
    raise RuntimeError(f"Could not locate a symbol column while resolving {label}.")


def _resolve_sp500() -> list[str]:
    return _resolve_wikipedia_list(_SP500_URL, "S&P 500")


def _resolve_sp1500() -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for url, label in (
        (_SP500_URL, "S&P 500"),
        (_SP400_URL, "S&P 400"),
        (_SP600_URL, "S&P 600"),
    ):
        for ticker in _resolve_wikipedia_list(url, label):
            if ticker in seen:
                continue
            seen.add(ticker)
            combined.append(ticker)
    if not combined:
        raise RuntimeError("Could not resolve any S&P 1500 constituents.")
    return combined


def _resolve_tradable_us(
    save_dir: str = "models",
    config: dict | None = None,
) -> list[str]:
    from pipeline.checkpoints import load_checkpoint_asset_list

    cfg = config or {}
    candidates: list[str] = []
    seen: set[str] = set()

    def _extend(symbols) -> None:
        for symbol in normalize_tickers(symbols):
            if symbol in seen:
                continue
            seen.add(symbol)
            candidates.append(symbol)

    _extend(load_checkpoint_asset_list(save_dir=save_dir) or [])
    _extend(_read_local_symbol_column(_PRICE_PANEL_PATH))
    _extend(_read_local_symbol_column(_WATCHLIST_PATH))

    inline = _resolve_custom(cfg)
    if inline:
        _extend(inline)

    return filter_candidate_tickers(candidates, config=cfg)


def _read_custom_universe_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            return normalize_tickers(df["ticker"].dropna().tolist())
        if len(df.columns) >= 1:
            return normalize_tickers(df.iloc[:, 0].dropna().tolist())
        return []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tickers", [])
        return normalize_tickers(payload)
    raw = path.read_text(encoding="utf-8")
    parts = [part for part in re.split(r"[\s,;]+", raw) if part]
    return normalize_tickers(parts)


def _resolve_custom(config: dict | None) -> list[str]:
    cfg = config or {}

    list_like = cfg.get("custom_universe")
    if isinstance(list_like, (list, tuple, set)):
        return normalize_tickers(list_like)

    for key in ("custom_universe_path", "universe_path"):
        raw_path = cfg.get(key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists():
            raise FileNotFoundError(f"Custom universe file not found: {path}")
        tickers = _read_custom_universe_file(path)
        if tickers:
            return tickers

    inline = cfg.get("custom_universe") or cfg.get("custom_tickers") or cfg.get("universe_list")
    if isinstance(inline, str):
        parts = [part for part in re.split(r"[\s,;]+", inline) if part]
        return normalize_tickers(parts)

    return []


def resolve_universe(
    mode: str,
    as_of_date=None,
    save_dir: str = "models",
    config: dict | None = None,
) -> list[str]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in SUPPORTED_UNIVERSE_MODES:
        raise ValueError(
            f"Unsupported universe mode {resolved_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_UNIVERSE_MODES)}."
        )

    snapshot_date = _coerce_snapshot_date(as_of_date)
    snapshot_path = _snapshot_path(resolved_mode, snapshot_date, save_dir=save_dir)
    cached = _read_snapshot(snapshot_path)
    if cached:
        return cached

    if resolved_mode == "tradable_us":
        tickers = _resolve_tradable_us(save_dir=save_dir, config=config)
        source = "local_tradable_us"
    elif resolved_mode == "sp500":
        tickers = _resolve_sp500()
        source = "wikipedia_sp500"
    elif resolved_mode == "sp1500":
        tickers = _resolve_sp1500()
        source = "wikipedia_sp1500"
    else:
        tickers = _resolve_custom(config)
        source = "custom_config"

    if not tickers:
        raise RuntimeError(
            f"Failed to resolve a non-empty universe for mode={resolved_mode!r}."
        )

    _write_snapshot(snapshot_path, resolved_mode, snapshot_date, tickers, source)
    return tickers


def resolve_configured_universe(
    as_of_date=None,
    save_dir: str = "models",
    config: dict | None = None,
) -> list[str]:
    cfg = config if config is not None else load_typed_config()
    mode = get_universe_mode(cfg)
    return resolve_universe(mode=mode, as_of_date=as_of_date, save_dir=save_dir, config=cfg)


def constrain_to_configured_universe(
    tickers: list[str] | None,
    as_of_date=None,
    save_dir: str = "models",
    config: dict | None = None,
) -> list[str]:
    allowed = set(resolve_configured_universe(as_of_date=as_of_date, save_dir=save_dir, config=config))
    ordered: list[str] = []
    seen: set[str] = set()
    for ticker in normalize_tickers(tickers or []):
        if ticker not in allowed or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered
