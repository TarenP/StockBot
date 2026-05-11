"""SEC EDGAR JSON API helpers.

Uses only public SEC endpoints. Set a descriptive User-Agent in config for
production use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_UA = os.getenv(
    "STOCKBOT_SEC_USER_AGENT",
    "StockBot research sidecar; set STOCKBOT_SEC_USER_AGENT with contact email",
)


def _headers(user_agent: str | None = None) -> dict[str, str]:
    return {
        "User-Agent": user_agent or os.getenv("STOCKBOT_SEC_USER_AGENT", DEFAULT_UA),
        "Accept-Encoding": "gzip, deflate",
    }


def cik_str(cik: str | int) -> str:
    return str(cik).strip().zfill(10)


def get_submissions(cik: str | int, *, user_agent: str | None = None, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(
        f"{SEC_BASE}/submissions/CIK{cik_str(cik)}.json",
        headers=_headers(user_agent),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_company_facts(cik: str | int, *, user_agent: str | None = None, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(
        f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik_str(cik)}.json",
        headers=_headers(user_agent),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def recent_filings(submissions: dict[str, Any], forms: set[str] | None = None) -> list[dict[str, Any]]:
    recent = (submissions or {}).get("filings", {}).get("recent", {})
    rows = []
    forms_filter = {form.upper() for form in forms or set()}
    for idx, form in enumerate(recent.get("form", []) or []):
        if forms_filter and str(form).upper() not in forms_filter:
            continue
        rows.append({
            "form": form,
            "accessionNumber": recent.get("accessionNumber", [None])[idx],
            "filingDate": recent.get("filingDate", [None])[idx],
            "reportDate": recent.get("reportDate", [None])[idx],
            "primaryDocument": recent.get("primaryDocument", [None])[idx],
        })
    return rows


def filing_document_url(cik: str | int, accession_number: str, primary_document: str) -> str:
    accession_clean = str(accession_number).replace("-", "")
    return f"{SEC_ARCHIVES}/{int(cik)}/{accession_clean}/{primary_document}"


def download_filing_document(
    cik: str | int,
    accession_number: str,
    primary_document: str,
    *,
    user_agent: str | None = None,
    timeout: int = 30,
) -> str:
    response = requests.get(
        filing_document_url(cik, accession_number, primary_document),
        headers=_headers(user_agent),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def load_ticker_cik_map(path: str | Path = "broker/state/company_tickers.json") -> dict[str, str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    mapping = {}
    rows = payload.values() if isinstance(payload, dict) else payload
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = cik_str(cik)
    return mapping
