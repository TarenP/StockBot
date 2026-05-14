import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.environment import PortfolioEnv
from pipeline.policy_diagnostics import (
    action_transform_trace,
    is_near_uniform,
    raw_policy_diagnostics,
    weight_concentration_metrics,
)
from pipeline.train import evaluate_diagnostics


def test_weight_concentration_metrics_equal_weight():
    weights = np.ones(100, dtype=np.float32) / 100

    metrics = weight_concentration_metrics(weights, universe_size=100, cash_weight=0.0)

    assert metrics["max_weight"] == pytest.approx(0.01)
    assert metrics["top_10_weight_sum"] == pytest.approx(0.10)
    assert metrics["top10_vs_equal_weight_ratio"] == pytest.approx(1.0)
    assert metrics["effective_number_of_positions"] == pytest.approx(100, rel=1e-5)


def test_weight_concentration_metrics_concentrated_weight():
    weights = np.zeros(100, dtype=np.float32)
    weights[:5] = 0.18
    weights[5:15] = 0.01

    metrics = weight_concentration_metrics(weights, universe_size=100, cash_weight=0.0)

    assert metrics["max_weight"] == pytest.approx(0.18)
    assert metrics["top_10_weight_sum"] == pytest.approx(0.95)
    assert metrics["top10_vs_equal_weight_ratio"] > 5.0
    assert metrics["effective_number_of_positions"] < 15


def test_near_uniform_warning_triggers():
    metrics = weight_concentration_metrics(np.ones(50) / 50, universe_size=50)

    assert is_near_uniform(metrics) is True


def test_effective_number_of_positions_calculation():
    metrics = weight_concentration_metrics(np.array([0.5, 0.5, 0.0]), universe_size=3)

    assert metrics["effective_number_of_positions"] == pytest.approx(2.0)


def test_policy_debug_output_does_not_change_weights():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.0]], dtype=torch.float32)
    projected = torch.nn.functional.softplus(logits) + 1e-6
    before = (projected / projected.sum(dim=-1, keepdim=True)).detach().clone()

    raw_policy_diagnostics(logits)
    action_transform_trace(logits, before.squeeze(0).numpy())

    projected_after = torch.nn.functional.softplus(logits) + 1e-6
    after = projected_after / projected_after.sum(dim=-1, keepdim=True)
    assert torch.allclose(before, after)


class _FixedWeightModel:
    def __init__(self, weights):
        self.weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)

    def eval(self):
        return self

    def train(self):
        return self

    def get_weights(self, obs):
        return self.weights.to(obs.device)


def _validation_frame(n_dates=30):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    close = []
    for i in range(n_dates):
        close.extend([100.0 + i, 50.0 + i * 0.25])
    return pd.DataFrame(
        {
            "ret_1d": np.zeros(len(index), dtype=np.float32),
            "close": close,
        },
        index=index,
    )


def test_validation_metrics_include_concentration_fields():
    env = PortfolioEnv(_validation_frame(), ["AAA", "BBB"], lookback=5, transaction_cost=0.0, feature_cols=["ret_1d"])
    model = _FixedWeightModel([0.8, 0.1, 0.1])

    metrics = evaluate_diagnostics(model, env, torch.device("cpu"))

    for key in (
        "avg_max_weight",
        "avg_top_10_weight_sum",
        "avg_top_20_weight_sum",
        "avg_weight_entropy",
        "avg_effective_number_of_positions",
        "avg_cash_weight",
        "avg_nonzero_positions",
        "avg_reward",
        "avg_excess_return_component",
        "avg_turnover_penalty",
        "avg_drawdown_penalty",
        "avg_entropy_term",
    ):
        assert key in metrics
    assert metrics["avg_max_weight"] == pytest.approx(0.8)
    assert metrics["avg_cash_weight"] == pytest.approx(0.1)


def test_action_transform_trace_preserves_sum_to_one():
    logits = torch.tensor([[0.4, 0.2, -0.1, 0.0]], dtype=torch.float32)
    final_weights = np.array([0.5, 0.2, 0.1, 0.2], dtype=np.float32)

    trace = action_transform_trace(logits, final_weights)

    assert [stage["stage"] for stage in trace] == [
        "raw_policy_shifted",
        "softplus_projection",
        "final_weights",
    ]
    for stage in trace:
        asset_sum = stage["top_50_weight_sum"]
        assert asset_sum + stage["cash_weight"] == pytest.approx(1.0)
