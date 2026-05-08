import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from broker.feature_ablation import (
    AGGREGATE_COLUMNS,
    FEATURE_ABLATION_VARIANTS,
    MACRO_ABLATION_SWEEP_VARIANTS,
    MECHANISM_SUMMARY_COLUMNS,
    TOUCH_COLUMNS,
    build_feature_decision,
    build_frozen_feature_config,
    run_feature_ablation_audit,
)


def _panel(n_dates=420):
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    tickers = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    frame = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1_000_000,
            "ret_1d": 0.0,
            "ret_5d": 0.0,
            "ret_20d": 0.0,
        },
        index=index,
    )
    return frame


def test_frozen_feature_config_disables_unpromoted_influence_layers():
    cfg = build_frozen_feature_config(
        {
            "earnings_reaction_enabled": True,
            "macro_regime_enabled": True,
            "insider_adjustment_enabled": True,
            "event_sidecar_broker_influence": True,
        },
        "baseline",
    )

    assert cfg["earnings_reaction_enabled"] is False
    assert cfg["macro_regime_enabled"] is False
    assert cfg["insider_adjustment_enabled"] is False
    assert cfg["event_sidecar_broker_influence"] is False
    assert cfg["event_sidecar_enabled"] is True


def test_feature_decision_uses_audit_gate_vocabulary():
    row = pd.Series(
        {
            "replay_safe": True,
            "feature_touch_count": 0,
            "feature_touch_rate": 0.0,
            "decision_changed_count": 0,
            "wins": 0,
            "winner_rate": 0.0,
            "incumbent_edge": 0.0,
        }
    )

    status, reason = build_feature_decision(row, baseline_policy_score=0.0)

    assert status == "hold_for_more_evidence"
    assert "no decision touches" in reason


def test_feature_ablation_dry_run_writes_required_artifacts():
    out_root = Path("tests/_tmp") / f"feature_ablation_{uuid4().hex}"
    df = _panel()
    try:
        result = run_feature_ablation_audit(
            df,
            df[["close", "volume"]],
            n_windows=3,
            output_root=out_root,
            run_id="dry_run",
            live_config={"rl_enabled": False},
            strategy="heuristics_only",
            dry_run=True,
        )
        run_dir = Path(result["output_dir"])

        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "window_manifest.csv").exists()
        assert (run_dir / "window_manifest.json").exists()
        assert (run_dir / "aggregate_feature_review.csv").exists()
        assert (run_dir / "aggregate_feature_review.json").exists()
        assert (run_dir / "feature_touch_audit.csv").exists()
        assert (run_dir / "feature_touch_audit.json").exists()
        assert (run_dir / "window_feature_mechanism_summary.csv").exists()
        assert (run_dir / "window_feature_mechanism_summary.json").exists()
        assert (run_dir / "promotion_decision.json").exists()

        aggregate = pd.read_csv(run_dir / "aggregate_feature_review.csv")
        touches = pd.read_csv(run_dir / "feature_touch_audit.csv")
        mechanism = pd.read_csv(run_dir / "window_feature_mechanism_summary.csv")
        assert list(aggregate.columns) == AGGREGATE_COLUMNS
        assert list(touches.columns) == TOUCH_COLUMNS
        assert list(mechanism.columns) == MECHANISM_SUMMARY_COLUMNS
        assert set(aggregate["label"]) == {variant.label for variant in FEATURE_ABLATION_VARIANTS}

        for variant in FEATURE_ABLATION_VARIANTS:
            assert (run_dir / variant.label / "window_A" / "metrics.json").exists()
            assert (run_dir / variant.label / "window_A" / "window_feature_mechanism_summary.csv").exists()

        decision = json.loads((run_dir / "promotion_decision.json").read_text())
        assert decision["decision_status"] == "hold_for_more_evidence"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_macro_sweep_variants_are_available_for_runner():
    labels = {variant.label for variant in MACRO_ABLATION_SWEEP_VARIANTS}

    assert {
        "baseline",
        "macro_weight_0.02",
        "macro_weight_0.04",
        "macro_weight_0.06",
        "macro_weight_0.08",
        "macro_risk_off_only",
        "macro_no_bull_boost",
        "macro_drawdown_guard_only",
        "macro_volatility_scaler_only",
    }.issubset(labels)
