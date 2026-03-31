import json
from pathlib import Path

import broker.sectors as sectors


def test_normalize_sector_name_maps_provider_aliases():
    assert sectors.normalize_sector_name("Consumer Cyclical") == "Consumer Discretionary"
    assert sectors.normalize_sector_name("Consumer Defensive") == "Consumer Staples"
    assert sectors.normalize_sector_name("Financial Services") == "Financials"
    assert sectors.normalize_sector_name("Basic Materials") == "Materials"
    assert sectors.normalize_sector_name("Technology") == "Technology"
    assert sectors.normalize_sector_name(None) == "Unknown"


def test_get_cached_sector_map_uses_normalized_disk_cache(monkeypatch):
    cache_path = Path("tests/.sector_cache_test.json")
    try:
        cache_path.write_text(
            json.dumps(
                {
                    "cvna": "Consumer Cyclical",
                    "coin": "Financial Services",
                    "aem": "Basic Materials",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(sectors, "SECTOR_CACHE_PATH", cache_path)

        merged = sectors.get_cached_sector_map(["CVNA", "COIN", "AEM", "AAPL", "MISSING"])

        assert merged["CVNA"] == "Consumer Discretionary"
        assert merged["COIN"] == "Financials"
        assert merged["AEM"] == "Materials"
        assert merged["AAPL"] == "Technology"
        assert merged["MISSING"] == "Unknown"
    finally:
        cache_path.unlink(missing_ok=True)


def test_compute_target_allocations_respects_hard_cap_without_renormalizing():
    target = sectors.compute_target_allocations(
        sector_scores={
            "Technology": 0.95,
            "Healthcare": 0.20,
            "Financials": 0.10,
        },
        current_sector_weights={},
        max_single_sector=0.485,
        min_sectors=3,
    )

    assert target["Technology"] <= 0.485 + 1e-9
    assert sum(target.values()) <= 1.0 + 1e-9
