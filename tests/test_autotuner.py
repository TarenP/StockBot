import pipeline.autotuner as autotuner


def test_fast_param_grid_is_budgeted_and_keeps_live_config():
    cfg = {
        "min_score": "0.58",
        "stop_loss": "0.115",
        "take_profit": "1.000",
        "max_sector": "0.400",
        "max_position_pct": "0.180",
        "cash_floor": "0.030",
        "max_gross_exposure": "0.990",
        "target_volatility": "0.220",
        "autotune_search_mode": "fast",
        "autotune_max_combinations": "12",
    }

    grid = autotuner._build_param_grid(cfg)

    assert len(grid) == 12
    assert grid[0]["take_profit"] == 1.0
    assert len({autotuner._param_signature(row) for row in grid}) == len(grid)


def test_full_param_grid_preserves_exhaustive_mode():
    grid = autotuner._build_param_grid({"autotune_search_mode": "full"})

    assert len(grid) == len(autotuner._PARAM_GRID)
