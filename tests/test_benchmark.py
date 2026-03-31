from pathlib import Path

import numpy as np

from pipeline.benchmark import plot_benchmark


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
