import pytest

from broker.exposure import (
    effective_bet_count,
    exposure_weights,
    low_price_bucket,
    portfolio_low_price_values,
    portfolio_theme_values,
    theme_bucket,
)


def test_theme_bucket_groups_finance_and_miner_clusters():
    assert theme_bucket("UWMC", "Financials") == "consumer_credit_finance"
    assert theme_bucket("AFRM", "Financials") == "consumer_credit_finance"
    assert theme_bucket("HL", "Materials") == "precious_metals_miners"
    assert theme_bucket("AAPL", "Technology") == "sector_technology"


def test_effective_bet_count_uses_theme_weights():
    weights = exposure_weights({"a": 50.0, "b": 50.0})
    assert effective_bet_count(weights) == pytest.approx(2.0)

    concentrated = exposure_weights({"a": 90.0, "b": 10.0})
    assert effective_bet_count(concentrated) < 2.0


def test_portfolio_low_price_values_break_out_sub_10_exposure():
    positions = {
        "PENNY": {"shares": 10, "last_price": 4.0},
        "LOW": {"shares": 5, "last_price": 8.0},
        "BIG": {"shares": 2, "last_price": 20.0},
    }

    values = portfolio_low_price_values(positions)

    assert low_price_bucket(4.99) == "sub_5"
    assert low_price_bucket(5.00) == "5_to_10"
    assert values["sub_5"] == pytest.approx(40.0)
    assert values["5_to_10"] == pytest.approx(40.0)
    assert values["over_10"] == pytest.approx(40.0)


def test_portfolio_theme_values_uses_static_theme_map():
    positions = {
        "UWMC": {"shares": 10, "last_price": 4.0},
        "AFRM": {"shares": 1, "last_price": 60.0},
        "AAPL": {"shares": 1, "last_price": 100.0},
    }

    values = portfolio_theme_values(positions, {"AAPL": "Technology"})

    assert values["consumer_credit_finance"] == pytest.approx(100.0)
    assert values["sector_technology"] == pytest.approx(100.0)
