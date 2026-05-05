"""FRED/ALFRED client using free public endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred"


class FredClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = int(timeout)

    def series_observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "series_id": series_id,
            "file_type": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end
        response = requests.get(f"{FRED_BASE}/series/observations", params=params, timeout=self.timeout)
        response.raise_for_status()
        rows = response.json().get("observations", [])
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
        return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

