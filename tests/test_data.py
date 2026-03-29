import pandas as pd

from pipeline.data import _lag_sentiment_to_next_trading_session


def test_lag_sentiment_to_next_trading_session_moves_signal_forward():
    price_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-05"), "AAA"),
            (pd.Timestamp("2024-01-08"), "AAA"),
            (pd.Timestamp("2024-01-09"), "AAA"),
        ],
        names=["date", "ticker"],
    )
    df_prices = pd.DataFrame(
        {
            "close": [10.0, 10.5, 10.7],
            "volume": [1000, 1100, 1200],
        },
        index=price_index,
    )

    sent_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-05"), "AAA"),
            (pd.Timestamp("2024-01-08"), "AAA"),
        ],
        names=["date", "ticker"],
    )
    df_sent = pd.DataFrame(
        {
            "neg_score": [0.1, 0.2],
            "neutral_score": [0.2, 0.2],
            "pos_score": [0.7, 0.6],
        },
        index=sent_index,
    )

    shifted = _lag_sentiment_to_next_trading_session(df_sent, df_prices, lag_sessions=1)

    assert (pd.Timestamp("2024-01-05"), "AAA") not in shifted.index
    assert shifted.loc[(pd.Timestamp("2024-01-08"), "AAA"), "pos_score"] == 0.7
    assert shifted.loc[(pd.Timestamp("2024-01-09"), "AAA"), "pos_score"] == 0.6
