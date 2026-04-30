from pathlib import Path

import numpy as np
import pandas as pd

import broker.journal as journal_module


class _DummyPortfolio:
    cash = 106.0
    positions = {}

    @property
    def equity(self) -> float:
        return 106.0

    @property
    def total_return(self) -> float:
        return 0.06

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


def test_print_report_includes_current_status_snapshot(monkeypatch, capsys):
    base = Path("tests") / "_journal_status"
    base.mkdir(exist_ok=True)
    eq_path = base / "equity.csv"
    history_path = base / "portfolio_history.jsonl"
    cap_path = base / "cap.json"
    attribution_path = base / "attribution.json"
    parity_path = base / "parity.json"

    pd.DataFrame(
        {
            "time": [
                "2024-01-02T16:00:00",
                "2024-01-03T10:00:00",
                "2024-01-03T16:00:00",
            ],
            "equity": [100.0, 102.0, 106.0],
            "cash": [100.0, 50.0, 40.0],
            "spy_price": [100.0, 101.0, 102.0],
        }
    ).to_csv(eq_path, index=False)
    history_path.write_text(
        '{"top_1_concentration": 0.25, "top_3_concentration": 0.60, '
        '"theme_effective_bet_count": 2.5, "low_price_exposure": 0.10}\n'
    )
    cap_path.write_text(
        '{"cycles": 3, "cycles_with_cap_interventions": 2, '
        '"entry_cap_interventions": {"buy_trades_with_caps": 4, '
        '"by_reason": {"theme_cap": 2, "sector_cap": 1, "low_price_or_penny_cap": 1}}}'
    )
    attribution_path.write_text(
        '{"realized_net_pnl": 12.5, "unrealized_pnl": 4.0, '
        '"total_execution_cost": 1.25, '
        '"exit_reason_counts": {"stop_loss": 1}, '
        '"by_theme": [{"theme": "consumer_credit_finance", "total_pnl": -3.0, '
        '"stop_out_rate": 1.0}], '
        '"by_price_bucket": [{"price_bucket": "5_to_10", "total_pnl": -2.0}]}'
    )
    parity_path.write_text('{"compatible": true}')

    try:
        monkeypatch.setattr(journal_module, "EQUITY_PATH", eq_path)
        monkeypatch.setattr(journal_module, "PORTFOLIO_HISTORY_PATH", history_path)
        monkeypatch.setattr(journal_module, "CAP_IMPACT_SUMMARY_PATH", cap_path)
        monkeypatch.setattr(journal_module, "PERFORMANCE_ATTRIBUTION_PATH", attribution_path)
        monkeypatch.setattr(journal_module, "PARITY_REPORT_PATH", parity_path)

        portfolio = _DummyPortfolio()
        portfolio.positions = {"AAA": {"shares": 1.0}}
        portfolio._last_mark_to_market = {
            "updated": {"AAA": 106.0},
            "sources": {"AAA": "live"},
            "missing": [],
            "latest_date": "2024-01-03",
        }

        journal_module.print_report(portfolio, show_benchmark=False)
        output = capsys.readouterr().out

        assert "Current Status" in output
        assert "EOD return:      +6.00%" in output
        assert "Today change:    +3.92%" in output
        assert "Prices updated:  2024-01-03 (1/1 holdings; live=1, cache=0)" in output
        assert "Marked return:   +6.00%" in output
        assert "top1=25.00%" in output
        assert "exec_cost=$1.25" in output
        assert "Exit reasons:    stop_loss=1" in output
        assert "Theme P&L:       consumer_credit_finance total=$-3.00 stop_out=100.0%" in output
        assert "Price buckets:   5_to_10=$-2.00" in output
        assert "3 cycle(s), 2 with cycle caps; 4 entry cap(s)" in output
        assert "Entry cap types: theme_cap=2" in output
        assert "Replay parity:   OK" in output
        assert "python Broker.py --status" in output
    finally:
        for path in [eq_path, history_path, cap_path, attribution_path, parity_path]:
            path.unlink(missing_ok=True)
        base.rmdir()
