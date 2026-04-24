import pandas as pd
from pathlib import Path
from uuid import uuid4

import pipeline.data as data_module
from pipeline.data import get_asset_universe


def test_get_asset_universe_respects_as_of_date():
    dates = pd.to_datetime(
        ["2023-01-02", "2023-01-03", "2025-01-02", "2025-01-03"]
    )
    index = pd.MultiIndex.from_tuples(
        [
            (dates[0], "AAA"),
            (dates[1], "AAA"),
            (dates[2], "BBB"),
            (dates[3], "BBB"),
        ],
        names=["date", "ticker"],
    )
    df = pd.DataFrame({"ret_1d": [0.0, 0.0, 0.0, 0.0]}, index=index)

    historical = get_asset_universe(
        df,
        top_n=1,
        lookback_years=1,
        as_of_date=pd.Timestamp("2023-01-03"),
    )
    latest = get_asset_universe(df, top_n=1, lookback_years=1)

    assert historical == ["AAA"]
    assert latest == ["BBB"]


def test_load_master_filters_to_configured_universe(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.5 + i,
                    "volume": 1_000_000.0,
                }
            )
    price_path = Path.cwd() / f".test_data_universe_{uuid4().hex}.parquet"
    pd.DataFrame(rows).set_index("date").to_parquet(price_path)

    try:
        monkeypatch.setattr(
            data_module,
            "resolve_configured_universe",
            lambda **kwargs: ["CCC", "AAA"],
        )
        monkeypatch.setattr(
            data_module,
            "build_features",
            lambda df_prices, **kwargs: df_prices[["close"]].copy(),
        )

        df = data_module.load_master(
            price_path=str(price_path),
            sentiment_path=str(Path.cwd() / ".missing_sentiment.csv"),
            min_history_days=1,
            min_price=0.0,
            min_avg_volume=0.0,
            config={"universe_mode": "sp500"},
        )

        assert sorted(df.index.get_level_values("ticker").unique().tolist()) == ["AAA", "CCC"]
    finally:
        price_path.unlink(missing_ok=True)
