"""
Property-based tests for _check_ablation_gate in broker/replay.py.

**Validates: Requirements 10.2, 10.3, 10.5**
"""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from broker.replay import _check_ablation_gate


def _make_report(rl_sharpe, rl_dd, base_sharpe, base_dd):
    """Build a minimal AblationReport DataFrame for gate testing."""
    return pd.DataFrame([
        {"strategy": "screener_rl",    "sharpe": rl_sharpe,   "max_drawdown": rl_dd},
        {"strategy": "heuristics_only","sharpe": base_sharpe, "max_drawdown": base_dd},
    ])


# ── Property: Gate returns "PASSED" iff both conditions are met simultaneously ──

@given(
    rl_sharpe   = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    base_sharpe = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    rl_dd       = st.floats(min_value=0.0,  max_value=1.0, allow_nan=False, allow_infinity=False),
    base_dd     = st.floats(min_value=0.0,  max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_ablation_gate_iff_both_conditions(rl_sharpe, base_sharpe, rl_dd, base_dd):
    """
    **Property: Gate returns "PASSED" iff both Sharpe and drawdown conditions
    are met simultaneously.**

    PASSED iff:
      rl_sharpe >= base_sharpe + 0.10  AND  rl_dd <= base_dd + 0.05
    FAILED otherwise.

    **Validates: Requirements 10.2, 10.3**
    """
    report = _make_report(rl_sharpe, rl_dd, base_sharpe, base_dd)
    result = _check_ablation_gate(report)

    sharpe_ok   = rl_sharpe >= base_sharpe + 0.10
    drawdown_ok = rl_dd     <= base_dd     + 0.05
    expected    = "PASSED" if (sharpe_ok and drawdown_ok) else "FAILED"

    assert result == expected, (
        f"Expected {expected!r} but got {result!r} for "
        f"rl_sharpe={rl_sharpe}, base_sharpe={base_sharpe}, "
        f"rl_dd={rl_dd}, base_dd={base_dd}"
    )
