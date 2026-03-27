# Implementation Plan: RL Broker Integration

## Overview

Integrate the trained PortfolioTransformer into live broker decisions in three phases: Phase 1 (RL ranking controller for entries), Phase 2 (RL conviction-drop exit triggers), and a Phase 3 stub (WeightAdapter, not activated). Implementation follows the dependency order: inference wrapper → config plumbing → BrokerBrain RL logic → replay ablation → Agent.py wiring → WeightAdapter stub.

## Tasks

- [x] 1. Create pipeline/rl_inference.py — inference wrapper foundation
  - Create `pipeline/rl_inference.py` with `ModelNotAvailableError` exception class
  - Implement `get_rl_targets(df_recent, asset_list, checkpoint_path, mode, device, lookback)` function
  - Load checkpoint, extract `model_cfg` / `model_state`, instantiate `PortfolioTransformer`, load state dict
  - Align `df_recent` to `asset_list` using `FEATURE_COLS` column order from `pipeline/features.py`
  - Apply cross-sectional z-score per date across `asset_list` tickers (matching `pipeline/data.py` training normalisation)
  - Clip feature values to `[-5.0, 5.0]` after z-scoring
  - Build obs tensor `(1, lookback, n_assets, n_features)` — pad with zeros for tickers with insufficient history
  - Call `model.get_weights(obs_t)` → softmax weights `(1, n_assets+1)`
  - For `mode="rank"`: exclude cash logit, renormalise asset weights to `[0, 1]` by dividing by their sum, return `pd.Series[ticker → rl_score]`
  - For `mode="weights"`: return full weight vector including `"CASH"` entry, verify sum within 1e-5 of 1.0
  - Raise `ModelNotAvailableError` if checkpoint path does not exist or fails to load
  - Assign `rl_score=0.0` for tickers with fewer than `lookback` dates rather than raising
  - Log a warning for any `FEATURE_COLS` column missing from `df_recent`, fill with 0.0
  - Cache loaded model in module-level dict keyed by `(checkpoint_path, device_str)` to avoid repeated disk I/O
  - Run inference on the device passed as parameter, defaulting to CPU
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 11.1, 11.2, 11.3, 11.4_

  - [x] 1.1 Write property test for get_rl_targets determinism
    - **Property: Identical inputs produce identical outputs (deterministic inference)**
    - **Validates: Requirements 1.4**

  - [x] 1.2 Write property test for mode="weights" sum invariant
    - **Property: RL weight series always sums to 1.0 ± 1e-5 for any valid input**
    - **Validates: Requirements 1.3**

  - [x] 1.3 Write property test for mode="rank" score bounds
    - **Property: All rl_score values are in [0, 1] for any valid asset_list**
    - **Validates: Requirements 1.2**

  - [x] 1.4 Write property test for z-score idempotency
    - **Property: Applying the normalisation pipeline twice produces the same output as once**
    - **Validates: Requirements 11.5**

- [x] 2. Add RL keys to broker.config and wire CLI args in broker/broker.py
  - Add five new keys to `broker.config`: `rl_enabled = false`, `rl_checkpoint_path = models/best_fold9.pt`, `rl_phase = 1`, `rl_exit_threshold = 0.30`, `rl_conviction_drop = 0.20`
  - Add five corresponding `--rl_*` arguments to `parse_args()` in `broker/broker.py`, reading defaults from the config dict
  - Pass all five RL args through to `BrokerBrain(...)` instantiation in `main()`
  - _Requirements: 6.4, 3.4_

- [x] 3. Extend BrokerBrain constructor with RL parameters
  - Add `rl_enabled`, `rl_checkpoint_path`, `rl_phase`, `rl_exit_threshold`, `rl_conviction_drop` to `BrokerBrain.__init__()` with the same defaults as `broker.config`
  - Store all five as instance attributes
  - _Requirements: 6.4, 4.1, 4.2_

- [x] 4. Implement _assert_model_available in broker/brain.py
  - Add `_assert_model_available(self)` method to `BrokerBrain`
  - Check that `rl_checkpoint_path` file exists; log CRITICAL and raise `RuntimeError` if not
  - Attempt to load checkpoint and verify `n_assets` in `model_cfg` matches `len(shortlist)`; log CRITICAL and raise `RuntimeError` on mismatch
  - _Requirements: 6.1, 6.2_

- [x] 5. Implement Phase 1 RL ranking in BrokerBrain.run_cycle
  - At the top of `run_cycle()`, when `rl_enabled=True`, call `_assert_model_available()` and abort (return `[]`) on `RuntimeError`
  - After `_screen_candidates()` produces the shortlist, call `get_rl_targets(df_features, shortlist, rl_checkpoint_path, mode="rank")` when `rl_enabled=True`
  - Sort buy candidates by `rl_score` descending when `rl_enabled=True`; preserve existing `composite_score` sort when `rl_enabled=False`
  - Apply `min_score` threshold to `rl_score` (not `composite_score`) when `rl_enabled=True`
  - Compute `composite_score` for every candidate regardless of mode and log it alongside `rl_score` as a diagnostic
  - Log `rl_score`, `composite_score`, and delta for every candidate evaluated in the cycle
  - Write per-cycle summary log: number of tickers scored by RL, number where RL rank differed from heuristic rank, top-5 RL-ranked tickers with scores
  - Include `rl_score`, `composite_score`, and `rl_mode=true` in the `reason` field of every Decision when RL is enabled
  - Store `rl_score_at_entry` in position metadata when a BUY is executed with RL enabled
  - When `rl_enabled=True`, suppress options decisions (do not call `_evaluate_options`); log a warning if `no_options=False`
  - When `rl_enabled=False`, run_cycle behaviour is unchanged
  - If `Inference_Wrapper` raises any exception during forward pass, abort cycle and log exception (no heuristic fallback)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.3, 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 12.1, 12.2, 12.4_

  - [x] 5.1 Write unit tests for Phase 1 ranking logic
    - Test that candidates are sorted by rl_score when rl_enabled=True
    - Test that composite_score sort is preserved when rl_enabled=False
    - Test that min_score threshold is applied to rl_score not composite_score
    - _Requirements: 3.1, 3.3, 3.4_

- [x] 6. Implement _rl_exit_checks in broker/brain.py (Phase 2)
  - Add `_rl_exit_checks(self, df_features, held_tickers)` method to `BrokerBrain`
  - For each held ticker, call `get_rl_targets(df_features, [ticker], rl_checkpoint_path, mode="rank")`
  - If `current_rl_score < rl_exit_threshold`, generate `Decision(SELL)`
  - If `(entry_rl_score - current_rl_score) > rl_conviction_drop`, generate `Decision(SELL_PARTIAL)` for 50% of position
  - If `Inference_Wrapper` fails for a ticker, retain position and log a warning (no spurious exit)
  - Log entry rl_score, current rl_score, drop magnitude, and threshold crossed for every RL-driven exit
  - Call `_rl_exit_checks()` before heuristic exits in `run_cycle()` when `rl_enabled=True` and `rl_phase >= 2`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 12.3_

  - [x] 6.1 Write unit tests for _rl_exit_checks
    - Test SELL generated when rl_score < rl_exit_threshold
    - Test SELL_PARTIAL generated when conviction drop exceeds threshold
    - Test position retained and warning logged when inference fails
    - _Requirements: 4.1, 4.2, 4.4_

- [x] 7. Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Add strategy parameter to run_replay in broker/replay.py
  - Add `strategy: str = "heuristics_only"` and `checkpoint_path: str | None = None` parameters to `run_replay()`
  - When `strategy="heuristics_only"`: use existing rule-based `_rank` score and `composite_score` (no changes to current logic)
  - When `strategy="screener_heuristics"`: use `TickerScorer` screener for shortlist, rank by `composite_score`
  - When `strategy="screener_rl"`: use `TickerScorer` screener for shortlist, call `get_rl_targets(..., mode="rank")` to rank by `rl_score`
  - When `strategy="rl_weights"`: use `TickerScorer` screener, call `get_rl_targets(..., mode="weights")`, size positions directly from `rl_weight`
  - Apply the same execution cost model across all four variants
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.9_

- [x] 9. Implement _check_ablation_gate in broker/replay.py
  - Add `_check_ablation_gate(report_df: pd.DataFrame) -> str` function
  - Return `"PASSED"` if `screener_rl` Sharpe >= `heuristics_only` Sharpe + 0.10 AND `screener_rl` max_drawdown <= `heuristics_only` max_drawdown + 0.05
  - Return `"FAILED"` otherwise, printing a prominent warning for each failed condition
  - Log the gate result as a single line in the broker log
  - _Requirements: 10.2, 10.3, 10.5_

  - [x] 9.1 Write property test for ablation gate logic
    - **Property: Gate returns "PASSED" iff both Sharpe and drawdown conditions are met simultaneously**
    - **Validates: Requirements 10.2, 10.3**

- [x] 10. Implement run_ablation in broker/replay.py
  - Add `run_ablation(df_features, price_lookup, checkpoint_path, initial_cash, replay_years, save_report, save_plot)` function
  - Run all four strategy variants over the same historical date range and initial cash using `run_replay()`
  - Collect `AblationReport` DataFrame with columns: `strategy`, `total_return`, `ann_return`, `sharpe`, `max_drawdown`, `win_rate`, `spy_alpha`, `n_trades`
  - Compute `spy_alpha` as `ann_return - spy_ann_return` for each variant
  - Call `_check_ablation_gate(report_df)` and log the result
  - Save report to `save_report` (CSV) and summary chart to `save_plot` (PNG)
  - _Requirements: 9.6, 9.7, 9.8, 10.1, 10.4, 10.5_

- [x] 11. Wire --mode ablation in Agent.py
  - Add `"ablation"` to the `--mode` choices list in `parse_args()`
  - Add `--rl_checkpoint` argument to `parse_args()` (path to `.pt` file, default `None`)
  - Implement `run_ablation_mode(args)` function that loads `df_features`, calls `run_ablation()` from `broker/replay.py`, and prints the result
  - Add `"ablation": run_ablation_mode` to the dispatch dict in `__main__`
  - _Requirements: 9.1_

- [x] 12. Add WeightAdapter stub in pipeline/rl_inference.py (Phase 3, not activated)
  - Add `WeightAdapter` class to `pipeline/rl_inference.py` with `__init__(min_weight_threshold, cash_floor)` and `adapt(rl_weights, portfolio, sector_map, equity) -> list` method
  - `adapt()` diffs current portfolio weights vs target rl_weights; generates SELL for weight=0, BUY for weight > min_weight_threshold
  - Apply sector cap, penny cap, and drawdown circuit breaker checks inside `adapt()`
  - Ensure total BUY value does not exceed `equity * (1 - cash_floor)`
  - Class is defined but NOT called from any live code path (Phase 3 only)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 13. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- `rl_enabled` defaults to `false` — the existing heuristic pipeline is unchanged until explicitly opted in
- Phase 3 (`WeightAdapter`) is implemented as a stub only; it is not wired into any live code path
- Checkpoints at tasks 7 and 13 ensure incremental validation before and after the replay/ablation layer
- Property tests validate universal correctness properties; unit tests validate specific examples and edge cases
