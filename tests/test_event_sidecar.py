from pathlib import Path
from uuid import uuid4

import pandas as pd

from event_sidecar.cache import EventSidecarCache, stable_event_id
from event_sidecar.feature_adapter import load_cached_event_features, precompute_event_features
from event_sidecar.impact import build_ticker_event_features, infer_event_type
from event_sidecar.quality_report import build_event_sidecar_quality_report
from event_sidecar.schemas import MarketEventRecord


def _tmp_cache() -> Path:
    path = Path("tests/_tmp") / f"event_sidecar_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_infer_event_type_detects_conflict_and_oil_events():
    assert infer_event_type("missile attack escalates regional war") == "conflict"
    assert infer_event_type("OPEC crude oil pipeline disruption") == "oil_shock"


def test_event_feature_mapping_scores_sector_exposure():
    event = MarketEventRecord(
        event_id="evt1",
        source="gdelt",
        event_type="oil_shock",
        title="Crude oil supply disruption lifts energy shares",
        as_of_date="2026-05-07",
        sentiment_score=-0.5,
        severity=0.8,
        confidence=0.75,
    )

    records = build_ticker_event_features(
        [event],
        ["XOM", "AAL"],
        sector_map={"XOM": "Energy", "AAL": "Consumer Discretionary"},
        as_of_date="2026-05-07",
    )
    by_ticker = {record.ticker: record for record in records}

    assert by_ticker["XOM"].event_opportunity_score > 0
    assert by_ticker["AAL"].event_risk_score > 0
    assert by_ticker["XOM"].broker_influence is False


def test_event_cache_precomputes_and_loads_features():
    cache_dir = _tmp_cache()
    cache = EventSidecarCache(cache_dir)
    event = MarketEventRecord(
        event_id=stable_event_id("gdelt", "Treasury yields jump after inflation report", "2026-05-07"),
        source="gdelt",
        event_type="rates",
        title="Treasury yields jump after inflation report",
        as_of_date="2026-05-07",
        sentiment_score=-0.4,
        severity=0.7,
        confidence=0.8,
    )
    cache.put_event(event)

    result = precompute_event_features(
        ["AMT", "JPM"],
        cache_dir=cache_dir,
        sector_map={"AMT": "Real Estate", "JPM": "Financials"},
        as_of_date="2026-05-07",
    )
    loaded = load_cached_event_features(["AMT", "JPM"], cache_dir=cache_dir)

    assert result["feature_records"] == 2
    assert loaded["AMT"]["event_risk_score"] > 0
    assert loaded["JPM"]["event_opportunity_score"] > 0


def test_event_quality_report_blocks_influence_without_enough_samples():
    cache_dir = _tmp_cache()
    cache = EventSidecarCache(cache_dir)
    event = MarketEventRecord(
        event_id="evt-quality",
        source="sentiment_csv",
        event_type="earnings",
        title="AAA earnings beat",
        as_of_date="2026-05-01",
        tickers=["AAA"],
        sentiment_score=0.8,
        severity=0.8,
        confidence=0.9,
    )
    cache.put_event(event)
    precompute_event_features(
        ["AAA"],
        cache_dir=cache_dir,
        sector_map={"AAA": "Technology"},
        as_of_date="2026-05-01",
    )
    dates = pd.date_range("2026-05-01", periods=8, freq="B")
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "close": [100, 101, 102, 103, 104, 105, 106, 107],
        }
    ).set_index(["date", "ticker"])
    price_path = cache_dir / "prices.parquet"
    panel.to_parquet(price_path)

    report = build_event_sidecar_quality_report(
        cache_dir=cache_dir,
        price_path=price_path,
        min_samples_for_influence=30,
    )

    assert report["validated_records"] == 1
    assert report["go_no_go"]["influence_allowed"] is False
    assert "samples" in report["go_no_go"]["failed_criteria"]
