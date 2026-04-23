import pandas as pd

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
