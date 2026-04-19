from pathlib import Path

import numpy as np

from pipeline.benchmark import (
    MIN_HISTORY_FOR_STABLE_METRICS,
    benchmark_vs_spy,
    compute_metrics,
    plot_benchmark,
    print_benchmark_report,
)


def test_plot_benchmark_handles_shorter_equal_weight_series():
    save_path = Path("tests") / "_benchmark_plot.png"
    if save_path.exists():
        save_path.unlink()

    plot_benchmark(
        portfolio_rets=np.array([0.01, 0.02, -0.01, 0.00, 0.03], dtype=float),
        spy_rets=np.array([0.00, 0.01, -0.01, 0.01, 0.02], dtype=float),
        ew_rets=np.array([0.02, -0.01, 0.01], dtype=float),
        save_path=str(save_path),
        label="Test Strategy",
    )

    assert save_path.exists()
    save_path.unlink()


def test_compute_metrics_hides_sample_sensitive_metrics_for_short_histories():
    rets = np.array([0.01, -0.02, 0.005, 0.003], dtype=float)

    metrics = compute_metrics(
        rets,
        "Short Sample",
        min_obs_for_annualized=MIN_HISTORY_FOR_STABLE_METRICS,
        min_obs_for_risk=MIN_HISTORY_FOR_STABLE_METRICS,
    )

    assert metrics["n_obs"] == 4
    assert metrics["total_return"] is not None
    assert metrics["max_drawdown"] is not None
    assert metrics["ann_return"] is None
    assert metrics["volatility"] is None
    assert metrics["sharpe"] is None
    assert metrics["sortino"] is None
    assert metrics["calmar"] is None


def test_benchmark_vs_spy_hides_relative_metrics_for_short_histories():
    portfolio_rets = np.array([0.01, -0.02, 0.005, 0.003], dtype=float)
    spy_rets = np.array([0.002, 0.001, -0.001, 0.004], dtype=float)

    rel = benchmark_vs_spy(
        portfolio_rets,
        spy_rets,
        min_obs_for_relative=MIN_HISTORY_FOR_STABLE_METRICS,
    )

    assert rel["n_obs"] == 4
    assert rel["beats_spy_return"] is False
    assert rel["beta"] is None
    assert rel["alpha_ann"] is None
    assert rel["information_ratio"] is None
    assert rel["tracking_error"] is None
    assert rel["beats_spy_sharpe"] is None


def test_print_benchmark_report_marks_hidden_metrics_as_na(capsys):
    portfolio_rets = np.array([0.01, -0.02, 0.005, 0.003], dtype=float)
    spy_rets = np.array([0.002, 0.001, -0.001, 0.004], dtype=float)

    print_benchmark_report(portfolio_rets, spy_rets, label="Short Sample")
    output = capsys.readouterr().out

    assert "n/a" in output
    assert "currently 4 observations" in output
