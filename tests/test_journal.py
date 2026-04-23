from pathlib import Path

import numpy as np
import pandas as pd

import broker.journal as journal_module


class _DummyPortfolio:
    def summary(self) -> str:
        return "Portfolio summary"


def _write_equity_fixture(filename: str, rows: dict) -> Path:
    eq_path = Path("tests") / filename
    pd.DataFrame(rows).to_csv(eq_path, index=False)
    return eq_path


def test_print_report_uses_true_max_drawdown(monkeypatch, capsys):
    eq_path = _write_equity_fixture(
        "_journal_equity_drawdown.csv",
        {
            "time": [
                "2024-01-02T16:00:00",
                "2024-01-03T16:00:00",
                "2024-01-04T16:00:00",
                "2024-01-05T16:00:00",
            ],
            "equity": [100.0, 120.0, 90.0, 110.0],
            "cash": [100.0, 120.0, 90.0, 110.0],
            "spy_price": [100.0, 101.0, 99.0, 102.0],
        },
    )

    try:
        monkeypatch.setattr(journal_module, "EQUITY_PATH", eq_path)

        journal_module.print_report(_DummyPortfolio(), show_benchmark=False)
        output = capsys.readouterr().out

        assert "Max drawdown: -25.00%" in output
    finally:
        eq_path.unlink(missing_ok=True)


def test_print_report_aligns_benchmark_by_timestamp(monkeypatch):
    eq_path = _write_equity_fixture(
        "_journal_equity_align.csv",
        {
            "time": [
                "2024-01-02T16:00:00",
                "2024-01-03T16:00:00",
                "2024-01-04T16:00:00",
                "2024-01-05T16:00:00",
            ],
            "equity": [100.0, 101.0, 103.0, 106.0],
            "cash": [100.0, 101.0, 103.0, 106.0],
            "spy_price": [100.0, np.nan, 102.0, 103.0],
        },
    )

    monkeypatch.setattr(journal_module, "EQUITY_PATH", eq_path)

    import pipeline.benchmark as benchmark_module

    captured = {}

    def fake_compute_metrics(rets, label, **kwargs):
        captured[label] = np.asarray(rets, dtype=float)
        return {
            "total_return": 0.0,
            "ann_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    def fake_benchmark_vs_spy(portfolio_rets, spy_rets, **kwargs):
        captured["relative_portfolio"] = np.asarray(portfolio_rets, dtype=float)
        captured["relative_spy"] = np.asarray(spy_rets, dtype=float)
        return {
            "beta": 0.0,
            "alpha_ann": 0.0,
            "information_ratio": 0.0,
            "upside_capture": 0.0,
            "downside_capture": 0.0,
            "beats_spy_return": False,
        }

    try:
        monkeypatch.setattr(benchmark_module, "compute_metrics", fake_compute_metrics)
        monkeypatch.setattr(benchmark_module, "benchmark_vs_spy", fake_benchmark_vs_spy)

        journal_module.print_report(_DummyPortfolio(), show_benchmark=True)

        assert np.allclose(captured["Broker"], [0.01980198, 0.02912621])
        assert np.allclose(captured["SPY"], [0.02, 0.00980392])
        assert np.allclose(captured["relative_portfolio"], [0.01980198, 0.02912621])
        assert np.allclose(captured["relative_spy"], [0.02, 0.00980392])
    finally:
        eq_path.unlink(missing_ok=True)
