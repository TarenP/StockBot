"""
Shared run-manifest helpers for replay, live, and maintenance workflows.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

RUN_MANIFEST_DIR = Path("plots/manifests")

# ── Required fields per manifest kind ────────────────────────────────────────
# Tests assert these are present so manifests stay complete over time.

REQUIRED_MANIFEST_FIELDS: dict[str, list[str]] = {
    "live_cycle": [
        "mode",
        "config_hash",
        "code_version",
        "resolved_universe_size",
        "resolved_universe_hash",
        "freshness",
        "freshness_gate",
        "benchmark_status",
        "snapshot_path",
        "watchlist_included",
    ],
    "replay": [
        "mode",
        "config_hash",
        "code_version",
        "checkpoint_path",
        "resolved_universe_size",
        "resolved_universe_hash",
        "replay_window",
        "benchmark",
        "friction",
        "snapshot_path",
        "watchlist_included",
    ],
}

# Fields that every manifest must have regardless of kind
REQUIRED_MANIFEST_FIELDS_COMMON = [
    "kind",
    "generated_at",
]


def _json_default(value: Any):
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_config(config: dict | None) -> str:
    return stable_hash(config or {})


def hash_ticker_list(tickers) -> str:
    cleaned = sorted({str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()})
    return stable_hash(cleaned)


def get_code_version() -> str:
    """Return short git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def validate_manifest(manifest: dict, kind: str) -> list[str]:
    """
    Check that a manifest contains all required fields.

    Returns a list of missing field names. Empty list = valid.
    """
    missing: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS_COMMON:
        if field not in manifest:
            missing.append(field)
    for field in REQUIRED_MANIFEST_FIELDS.get(kind, []):
        if field not in manifest:
            missing.append(field)
    return missing


def write_run_manifest(
    kind: str,
    payload: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    manifest = dict(payload)
    manifest.setdefault("kind", kind)
    manifest.setdefault("generated_at", datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
    manifest.setdefault("code_version", get_code_version())

    # Warn if required fields are missing — helps catch regressions early
    missing = validate_manifest(manifest, kind)
    if missing:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Run manifest '%s' is missing required fields: %s", kind, missing
        )

    if output_path is None:
        RUN_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RUN_MANIFEST_DIR / f"{kind}_{timestamp}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def summarize_price_sentiment_freshness(
    df: pd.DataFrame,
    positions: dict | None = None,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "latest_data_date": None,
            "fresh_price_coverage": 0.0,
            "fresh_sentiment_coverage": 0.0,
            "candidate_count": 0,
            "latest_row_count": 0,
            "stale_holdings": sorted((positions or {}).keys()),
            "stale_holdings_count": len(positions or {}),
        }

    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names or "ticker" not in df.index.names:
        return {
            "latest_data_date": None,
            "fresh_price_coverage": 0.0,
            "fresh_sentiment_coverage": 0.0,
            "candidate_count": 0,
            "latest_row_count": 0,
            "stale_holdings": sorted((positions or {}).keys()),
            "stale_holdings_count": len(positions or {}),
        }

    dates = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
    latest_date = pd.Timestamp(dates.max()).normalize()
    latest_slice = df[dates.normalize() == latest_date]
    latest_tickers = {
        str(ticker).strip().upper()
        for ticker in latest_slice.index.get_level_values("ticker")
    }
    all_tickers = {
        str(ticker).strip().upper()
        for ticker in df.index.get_level_values("ticker")
    }
    candidate_count = len(all_tickers)
    latest_row_count = len(latest_tickers)
    fresh_price_coverage = (latest_row_count / candidate_count) if candidate_count else 0.0

    sentiment_col = "sent_net" if "sent_net" in latest_slice.columns else None
    if sentiment_col is not None and latest_row_count:
        fresh_sentiment_coverage = float(latest_slice[sentiment_col].notna().mean())
    else:
        fresh_sentiment_coverage = 0.0

    holdings = {str(ticker).strip().upper() for ticker in (positions or {}).keys()}
    stale_holdings = sorted(holdings - latest_tickers)
    return {
        "latest_data_date": latest_date.date().isoformat(),
        "fresh_price_coverage": float(fresh_price_coverage),
        "fresh_sentiment_coverage": float(fresh_sentiment_coverage),
        "candidate_count": int(candidate_count),
        "latest_row_count": int(latest_row_count),
        "stale_holdings": stale_holdings,
        "stale_holdings_count": len(stale_holdings),
    }
