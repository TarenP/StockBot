import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

import broker.shadows as shadows_module


def _snapshot_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02"]), ["AAA", "BBB", "CCC"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "ret_5d": [0.8, 0.7, 0.6],
            "sent_net": [0.6, 0.5, 0.4],
            "macd_hist": [0.5, 0.4, 0.3],
            "vol_ratio": [2.0, 1.8, 1.6],
            "rsi": [52.0, 53.0, 51.0],
            "ret_20d": [0.7, 0.6, 0.5],
        },
        index=index,
    )


def _price_lookup() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["AAA"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "close": [10.0, 10.5],
            "volume": [1_000_000.0, 1_100_000.0],
        },
        index=index,
    )


def test_fast_score_population_preserves_validated_sharpe():
    population = [
        {
            "min_score": 0.4,
            "stop_loss": 0.08,
            "take_profit": 0.35,
            "max_sector": 0.25,
            "partial_profit": 0.15,
            "avoid_earnings": 5,
            "rl_exit_threshold": 0.30,
            "rl_conviction_drop": 0.20,
            "no_options": True,
            "rl_enabled": False,
            "rl_phase": 1,
            "fast_score": 0.0,
            "sharpe": 2.5,
            "validated": True,
            "validation_metric_version": shadows_module.VALIDATION_METRIC_VERSION,
            "age": 0,
        }
    ]

    scored = shadows_module.fast_score_population(population, _snapshot_frame())

    assert scored[0]["sharpe"] == 2.5
    assert scored[0]["fast_score"] > 0.0


def test_validate_top_genomes_always_revalidates_baseline(monkeypatch):
    baseline = shadows_module._genome_from_config({})
    baseline["fast_score"] = -1.0
    challenger = shadows_module._genome_from_config({})
    challenger.pop("is_baseline", None)
    challenger["fast_score"] = 10.0

    df_features = pd.DataFrame(
        {"ret_1d": [0.01, 0.02]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["AAA"]],
            names=["date", "ticker"],
        ),
    )

    monkeypatch.setattr(
        "broker.replay.run_replay",
        lambda *args, **kwargs: (np.array([0.01, -0.005, 0.02]), []),
    )

    updated = shadows_module.validate_top_genomes(
        [baseline, challenger],
        df_features,
        _price_lookup(),
        checkpoint_path=None,
        top_n=1,
        replay_years=1,
    )

    assert updated[0]["validated"] is True
    assert updated[0]["validation_metric_version"] == shadows_module.VALIDATION_METRIC_VERSION
    assert updated[1]["validated"] is True


def test_validate_top_genomes_uses_genome_specific_replay_params(monkeypatch):
    baseline = shadows_module._genome_from_config({})
    baseline["fast_score"] = -1.0

    challenger = shadows_module._genome_from_config({})
    challenger.pop("is_baseline", None)
    challenger.update(
        {
            "fast_score": 10.0,
            "min_score": 0.69,
            "partial_profit": 0.31,
            "max_position_pct": 0.19,
            "cash_floor": 0.02,
            "max_gross_exposure": 0.98,
            "target_volatility": 0.23,
            "avoid_earnings": 6,
            "rl_enabled": True,
            "rl_phase": 2,
            "rl_exit_threshold": 0.18,
            "rl_conviction_drop": 0.16,
        }
    )

    df_features = pd.DataFrame(
        {"ret_1d": [0.01, 0.02]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["AAA"]],
            names=["date", "ticker"],
        ),
    )

    captured = {}

    def _fake_run_replay(*args, **kwargs):
        captured.update(kwargs)
        return np.array([0.01, -0.005, 0.02]), []

    monkeypatch.setattr(shadows_module, "_resolve_shadow_checkpoint", lambda path: "models/best_fold9.pt")
    monkeypatch.setattr("broker.replay.run_replay", _fake_run_replay)

    shadows_module.validate_top_genomes(
        [baseline, challenger],
        df_features,
        _price_lookup(),
        checkpoint_path="auto",
        top_n=1,
        replay_years=1,
    )

    assert captured["strategy"] == "screener_rl"
    assert captured["checkpoint_path"] == "models/best_fold9.pt"
    assert captured["min_score"] == challenger["min_score"]
    assert captured["partial_profit_pct"] == challenger["partial_profit"]
    assert captured["max_position_pct"] == challenger["max_position_pct"]
    assert captured["cash_floor"] == challenger["cash_floor"]
    assert captured["max_gross_exposure"] == challenger["max_gross_exposure"]
    assert captured["target_volatility"] == challenger["target_volatility"]
    assert captured["avoid_earnings_days"] == challenger["avoid_earnings"]
    assert captured["rl_phase"] == challenger["rl_phase"]
    assert captured["rl_exit_threshold"] == challenger["rl_exit_threshold"]
    assert captured["rl_conviction_drop"] == challenger["rl_conviction_drop"]


def test_maybe_promote_ignores_stale_validations(monkeypatch):
    baseline = shadows_module._genome_from_config({})
    baseline["validated"] = True
    baseline["sharpe"] = 1.2
    baseline["validation_metric_version"] = shadows_module.VALIDATION_METRIC_VERSION

    stale = shadows_module._genome_from_config({})
    stale.pop("is_baseline", None)
    stale["validated"] = True
    stale["sharpe"] = 99.0
    stale["validation_metric_version"] = 0

    monkeypatch.setattr("pipeline.autotuner._write_config_key", lambda *args, **kwargs: None)

    updated, promoted = shadows_module._maybe_promote(
        [baseline, stale],
        live_config={},
        checkpoint_path=None,
    )

    assert promoted is False
    assert updated[0]["is_baseline"] is True


def test_maybe_promote_persists_exposure_knobs(monkeypatch):
    baseline = shadows_module._genome_from_config({})
    baseline["validated"] = True
    baseline["sharpe"] = 1.0
    baseline["validation_metric_version"] = shadows_module.VALIDATION_METRIC_VERSION

    challenger = shadows_module._genome_from_config({})
    challenger.pop("is_baseline", None)
    challenger.update(
        {
            "validated": True,
            "sharpe": 1.3,
            "validation_metric_version": shadows_module.VALIDATION_METRIC_VERSION,
            "max_position_pct": 0.21,
            "cash_floor": 0.01,
            "max_gross_exposure": 0.99,
            "target_volatility": 0.24,
        }
    )

    writes = {}

    def _capture_write(key, value, path="broker.config"):
        writes[key] = value

    monkeypatch.setattr("pipeline.autotuner._write_config_key", _capture_write)

    _updated, promoted = shadows_module._maybe_promote(
        [baseline, challenger],
        live_config={},
        checkpoint_path=None,
    )

    assert promoted is True
    assert writes["max_position_pct"] == "0.210"
    assert writes["cash_floor"] == "0.010"
    assert writes["max_gross_exposure"] == "0.990"
    assert writes["target_volatility"] == "0.240"


def test_run_shadow_cycle_forces_revalidation_on_metric_upgrade(monkeypatch):
    state_path = Path.cwd() / f".test_shadows_state_{uuid4().hex}.json"
    monkeypatch.setattr(shadows_module, "_STATE_FILE", str(state_path))

    baseline = shadows_module._genome_from_config({})
    baseline["validated"] = True
    baseline["sharpe"] = 1.2
    baseline["fast_score"] = 0.4
    baseline["validation_metric_version"] = 0

    challenger = shadows_module._genome_from_config({})
    challenger.pop("is_baseline", None)
    challenger["validated"] = True
    challenger["sharpe"] = 1.8
    challenger["fast_score"] = 0.9
    challenger["validation_metric_version"] = 0

    calls = {"validated": 0}

    monkeypatch.setattr(shadows_module, "fast_score_population", lambda population, df_features: population)

    def _fake_validate(population, *args, **kwargs):
        calls["validated"] += 1
        for genome in population:
            if genome.get("validated"):
                genome["validation_metric_version"] = shadows_module.VALIDATION_METRIC_VERSION
        return population

    monkeypatch.setattr(shadows_module, "validate_top_genomes", _fake_validate)
    monkeypatch.setattr(shadows_module, "_maybe_promote", lambda population, *args, **kwargs: (population, False))
    monkeypatch.setattr(shadows_module, "evolve_population", lambda population, live_config: population)
    monkeypatch.setattr(shadows_module, "_maybe_enable_options", lambda *args, **kwargs: False)
    monkeypatch.setattr(shadows_module, "_log_summary", lambda *args, **kwargs: None)

    try:
        state_path.write_text(
            json.dumps(
                {
                    "population": [baseline, challenger],
                    "generation": 3,
                    "last_evolved": date.today().isoformat(),
                    "last_validated": date.today().isoformat(),
                    "options_days_beating": 0,
                    "baseline_sharpe": 1.2,
                }
            )
        )

        shadows_module.run_shadow_cycle(
            df_features=pd.DataFrame(),
            price_lookup=pd.DataFrame(),
            live_config={},
        )

        assert calls["validated"] == 1
    finally:
        state_path.unlink(missing_ok=True)
