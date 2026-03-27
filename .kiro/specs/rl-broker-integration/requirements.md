# Requirements Document

## Introduction

This feature bridges the trained RL Transformer model (PortfolioTransformer, trained via PPO walk-forward) into live broker decisions. Currently the broker (Broker.py / broker/brain.py) runs a fully heuristic pipeline — screener → composite score → sector allocation → sizing → exits — and the RL model is only used in Agent.py `--mode predict` and `--mode backtest` with zero influence on live trading.

The integration proceeds in three phases:

- **Phase 1 (Ranking Controller):** RL scores/ranks screener candidates; heuristics become diagnostics, not final authority for entries.
- **Phase 2 (Hold/Exit Influence):** RL conviction drop triggers reduce/exit decisions on held positions.
- **Phase 3 (Weight Controller, future):** RL drives target portfolio weights; broker becomes a pure execution and risk engine.

A reusable inference wrapper, universe alignment contract, adapter layer, hard model-required mode, and extended replay testing framework are all required before Phase 1 is declared complete.

---

## Glossary

- **RL_Model**: The trained PortfolioTransformer checkpoint loaded from `models/best_fold*.pt`.
- **Inference_Wrapper**: The `get_rl_targets(df_recent, asset_list, checkpoint_path, mode)` function extracted from Agent.py predict logic.
- **Broker_Brain**: The `BrokerBrain` class in `broker/brain.py` that produces `Decision` objects each cycle.
- **Screener**: The `TickerScorer` model in `pipeline/screener.py` that narrows the full universe to a shortlist.
- **Shortlist**: The ordered list of tickers output by the Screener, passed to the RL_Model for ranking.
- **RL_Score**: The per-ticker conviction value output by the Inference_Wrapper for a given ticker in the Shortlist.
- **RL_Weight**: The continuous portfolio weight in [0, 1] output by the RL_Model for a given asset (used in Phase 3).
- **Composite_Score**: The heuristic weighted signal score in [0, 1] computed by `broker/analyst.py _composite_score()`.
- **Adapter**: The component that converts RL continuous weights into discrete BUY/SELL/HOLD orders with share counts.
- **Replay_Engine**: The extended `broker/replay.py` that supports multiple strategy variants for ablation testing.
- **Ablation_Report**: The structured comparison of all strategy variants (heuristics-only, screener+heuristics, screener+RL, RL-weights) vs SPY after costs.
- **Universe_Alignment**: The guarantee that the Broker_Brain and RL_Model operate on the same tickers, same FEATURE_COLS, and the same cross-sectional z-scoring at decision time.
- **Hard_Model_Required_Mode**: The operating mode in which the broker refuses to proceed with any trade decisions if the RL_Model checkpoint is unavailable or fails to load.
- **FEATURE_COLS**: The canonical ordered list of 19 features defined in `pipeline/features.py`.
- **Z_Score**: Cross-sectional standardisation of each feature across all tickers on a given date, as applied during RL training.

---

## Requirements

### Requirement 1: Inference Wrapper

**User Story:** As a developer, I want a reusable inference function that encapsulates all RL model loading and forward-pass logic, so that any component (broker, replay, backtest) can obtain RL scores without duplicating Agent.py predict code.

#### Acceptance Criteria

1. THE Inference_Wrapper SHALL accept `df_recent` (a MultiIndex [date, ticker] DataFrame of FEATURE_COLS), `asset_list` (ordered list of ticker strings), `checkpoint_path` (path to a `.pt` file), and `mode` (one of `"rank"` or `"weights"`) as inputs.
2. WHEN `mode` is `"rank"`, THE Inference_Wrapper SHALL return a `pd.Series` indexed by ticker with RL_Score values in [0, 1], one per ticker in `asset_list`.
3. WHEN `mode` is `"weights"`, THE Inference_Wrapper SHALL return a `pd.Series` indexed by ticker (plus a `"CASH"` entry) with RL_Weight values that sum to 1.0 within a tolerance of 1e-5.
4. WHEN called twice with identical inputs, THE Inference_Wrapper SHALL return identical outputs (deterministic inference).
5. IF `checkpoint_path` does not exist or fails to load, THEN THE Inference_Wrapper SHALL raise a `ModelNotAvailableError` with a message identifying the missing path.
6. IF `df_recent` contains fewer than `lookback` dates for any ticker in `asset_list`, THEN THE Inference_Wrapper SHALL assign that ticker an RL_Score of 0.0 rather than raising an exception.
7. THE Inference_Wrapper SHALL apply the same cross-sectional Z_Score normalisation to `df_recent` that was applied during RL training, using the FEATURE_COLS column order from `pipeline/features.py`.
8. THE Inference_Wrapper SHALL run inference on the same device (CPU or CUDA) that is passed as a parameter, defaulting to CPU.

---

### Requirement 2: Universe Alignment

**User Story:** As a developer, I want the broker and the RL model to operate on the same universe at every decision cycle, so that RL scores are computed on the same tickers and features the broker will act on.

#### Acceptance Criteria

1. WHEN the Broker_Brain runs a cycle with RL enabled, THE Broker_Brain SHALL pass the Shortlist tickers to the Inference_Wrapper as `asset_list`, ensuring the RL_Model scores exactly the tickers the broker is considering.
2. THE Broker_Brain SHALL build the observation tensor for the Inference_Wrapper using the same `df_features` slice (same dates, same FEATURE_COLS, same Z_Score) that was used to produce the Shortlist.
3. IF a ticker appears in the Shortlist but has no rows in `df_features` for the required lookback window, THEN THE Broker_Brain SHALL exclude that ticker from both the Shortlist and the RL scoring pass.
4. THE Broker_Brain SHALL NOT pass tickers to the Inference_Wrapper that were not first output by the Screener, preserving the two-stage architecture.
5. WHEN the Screener is unavailable and the rule-based fallback is used, THE Broker_Brain SHALL still pass the fallback candidate list to the Inference_Wrapper using the same Universe_Alignment contract.

---

### Requirement 3: Phase 1 — RL as Ranking Controller for Entries

**User Story:** As a portfolio manager, I want the RL model to rank buy candidates instead of the heuristic composite score, so that entry decisions are driven by the model's learned signal rather than a hand-tuned formula.

#### Acceptance Criteria

1. WHEN RL mode is enabled and the Inference_Wrapper returns RL_Scores for the Shortlist, THE Broker_Brain SHALL sort buy candidates in descending order of RL_Score rather than Composite_Score.
2. THE Broker_Brain SHALL still compute Composite_Score for every candidate and log it alongside the RL_Score as a diagnostic signal, but SHALL NOT use Composite_Score to determine entry order.
3. WHEN RL mode is enabled, THE Broker_Brain SHALL apply the existing `min_score` threshold to RL_Score (not Composite_Score) to filter candidates before sizing.
4. WHEN RL mode is disabled, THE Broker_Brain SHALL use Composite_Score as the ranking signal, preserving the existing heuristic behaviour exactly.
5. THE Broker_Brain SHALL log the RL_Score, Composite_Score, and the delta between them for every candidate evaluated in a cycle, to support post-hoc signal attribution.
6. WHILE RL mode is enabled, THE Broker_Brain SHALL continue to apply all existing risk controls (sector budget, penny cap, ATR stop, earnings avoidance, cash floor) after RL ranking, not before.

---

### Requirement 4: Phase 2 — RL Conviction Drop Triggers Exit

**User Story:** As a portfolio manager, I want the RL model's falling conviction on a held position to trigger a reduce or exit decision, so that the model's view of deteriorating opportunity is acted on before heuristic stops are hit.

#### Acceptance Criteria

1. WHEN RL mode is enabled and the Inference_Wrapper returns an RL_Score for a held ticker that is below a configurable `rl_exit_threshold` (default 0.30), THE Broker_Brain SHALL generate a SELL decision for that position.
2. WHEN RL mode is enabled and the RL_Score for a held ticker drops by more than a configurable `rl_conviction_drop` (default 0.20) relative to the RL_Score recorded at entry, THE Broker_Brain SHALL generate a SELL_PARTIAL decision for 50% of the position.
3. THE Broker_Brain SHALL record the RL_Score at the time of entry for every position opened while RL mode is enabled, storing it in the position metadata.
4. IF the Inference_Wrapper fails to return a score for a held ticker during an exit check, THEN THE Broker_Brain SHALL retain the position and log a warning rather than generating a spurious exit.
5. THE Broker_Brain SHALL log the RL_Score, entry RL_Score, and the reason for any RL-driven exit decision in the trade journal.
6. WHILE RL mode is enabled, THE Broker_Brain SHALL evaluate RL-driven exits before heuristic exits (stop-loss, take-profit, signal deterioration) in each cycle, so that RL exits take precedence.

---

### Requirement 5: Phase 3 — RL as Weight Controller (Future Constraint)

**User Story:** As a portfolio manager, I want the RL model to output target portfolio weights that the broker translates into orders, so that the broker becomes a pure execution and risk engine rather than a decision maker.

#### Acceptance Criteria

1. WHEN Phase 3 mode is enabled, THE Adapter SHALL accept the RL_Weight vector (n_assets + 1, including cash) from the Inference_Wrapper and produce a list of BUY, SELL, and HOLD decisions that move the current portfolio toward the target weights.
2. THE Adapter SHALL ensure that the total allocated value implied by all BUY decisions does not exceed `portfolio.equity * (1 - cash_floor)`.
3. WHEN the RL_Weight for a held ticker is 0.0, THE Adapter SHALL generate a SELL decision for the full position.
4. WHEN the RL_Weight for a ticker not currently held is greater than a configurable `min_weight_threshold` (default 0.01), THE Adapter SHALL generate a BUY decision sized to reach the target weight.
5. THE Adapter SHALL apply all existing risk controls (sector cap, penny cap, drawdown circuit breaker) to the orders it generates, even when operating in Phase 3 mode.
6. THE Adapter SHALL NOT be activated in Phase 1 or Phase 2 mode; it is reserved for Phase 3.

---

### Requirement 6: Hard Model-Required Mode

**User Story:** As a developer, I want the broker to refuse to trade when RL mode is enabled but the model is unavailable, so that there is no silent fallback to heuristics that would produce unaudited decisions.

#### Acceptance Criteria

1. WHEN `rl_enabled` is true in the broker configuration and the checkpoint file specified by `rl_checkpoint_path` does not exist, THE Broker_Brain SHALL abort the cycle before generating any decisions and log a `CRITICAL` error identifying the missing checkpoint.
2. WHEN `rl_enabled` is true and the checkpoint loads but the model architecture does not match the current `asset_list` size, THE Broker_Brain SHALL abort the cycle and log a `CRITICAL` error describing the mismatch.
3. WHEN `rl_enabled` is true and the Inference_Wrapper raises any exception during the forward pass, THE Broker_Brain SHALL abort the cycle and log the exception rather than falling back to heuristic ranking.
4. THE Broker_Brain SHALL expose a `rl_enabled` flag (default false) that can be set via `broker.config` or CLI argument, so that the hard model-required mode is opt-in and the existing heuristic pipeline remains the default.
5. WHEN `rl_enabled` is false, THE Broker_Brain SHALL operate exactly as it does today with no code paths touching the Inference_Wrapper.

---

### Requirement 7: Two-Stage ML Architecture

**User Story:** As a developer, I want the screener to narrow the full universe before the RL model ranks candidates, so that the RL model operates on a tractable shortlist rather than all 11,500 tickers.

#### Acceptance Criteria

1. THE Broker_Brain SHALL invoke the Screener first to reduce the full universe to a Shortlist of at most `screener_top_n` tickers before invoking the Inference_Wrapper.
2. THE Inference_Wrapper SHALL only score tickers that appear in the Shortlist; it SHALL NOT score the full universe directly.
3. WHEN the Screener produces a Shortlist of fewer than 5 tickers, THE Broker_Brain SHALL log a warning and proceed with the available tickers rather than aborting.
4. THE Broker_Brain SHALL record the Shortlist size and the number of tickers that received RL_Scores in the cycle log for observability.

---

### Requirement 8: Options Disabled During RL Integration

**User Story:** As a risk manager, I want options trading to be disabled whenever RL mode is active, so that the integration is validated on equity decisions only before adding options complexity.

#### Acceptance Criteria

1. WHEN `rl_enabled` is true, THE Broker_Brain SHALL NOT generate OPEN_OPTION or CLOSE_OPTION decisions regardless of the `no_options` configuration value.
2. WHEN `rl_enabled` is true and the `no_options` flag is false in `broker.config`, THE Broker_Brain SHALL log a warning that options have been suppressed due to RL mode and SHALL NOT treat this as an error.
3. THE Broker_Brain SHALL enforce the options suppression check before the options evaluation step in `_evaluate_options`, so that no options analysis is performed when RL mode is active.

---

### Requirement 9: Replay Testing — Four Strategy Variants

**User Story:** As a developer, I want the broker replay to support four distinct strategy variants run over the same historical period, so that I can measure the incremental contribution of each component.

#### Acceptance Criteria

1. THE Replay_Engine SHALL support a `strategy` parameter accepting one of: `"heuristics_only"`, `"screener_heuristics"`, `"screener_rl"`, or `"rl_weights"`.
2. WHEN `strategy` is `"heuristics_only"`, THE Replay_Engine SHALL use the existing rule-based candidate scoring (no Screener, no RL_Model) as the baseline.
3. WHEN `strategy` is `"screener_heuristics"`, THE Replay_Engine SHALL use the Screener to produce the Shortlist and Composite_Score to rank within it.
4. WHEN `strategy` is `"screener_rl"`, THE Replay_Engine SHALL use the Screener to produce the Shortlist and RL_Score to rank within it (Phase 1 logic).
5. WHEN `strategy` is `"rl_weights"`, THE Replay_Engine SHALL use RL_Weight vectors directly to size positions (Phase 3 logic).
6. THE Replay_Engine SHALL run all four variants over the same historical date range and initial cash amount to ensure comparability.
7. THE Replay_Engine SHALL compute and report Sharpe ratio, total return, annualised return, maximum drawdown, and SPY alpha after execution costs for each variant.
8. THE Replay_Engine SHALL produce a side-by-side Ablation_Report comparing all four variants and SPY on the same metrics.
9. WHEN running `"screener_rl"` or `"rl_weights"` variants, THE Replay_Engine SHALL apply the same execution cost model as the other variants to ensure fair comparison.

---

### Requirement 10: Ablation Gate Before Integration Declared Complete

**User Story:** As a developer, I want a formal ablation test that must pass before the RL integration is declared production-ready, so that the RL model is only promoted if it demonstrably improves on the heuristic baseline.

#### Acceptance Criteria

1. THE Ablation_Report SHALL include Sharpe ratio, total return, maximum drawdown, and SPY alpha for all four strategy variants over the same replay period.
2. WHEN the `"screener_rl"` variant does not achieve a Sharpe ratio at least 0.10 higher than `"heuristics_only"` over the replay period, THE Replay_Engine SHALL print a prominent warning that the RL integration has not cleared the ablation gate.
3. WHEN the `"screener_rl"` variant achieves a maximum drawdown more than 5 percentage points worse than `"heuristics_only"`, THE Replay_Engine SHALL print a prominent warning regardless of Sharpe improvement.
4. THE Ablation_Report SHALL be saved to `plots/ablation_report.csv` and a summary chart to `plots/ablation.png` at the end of the replay run.
5. THE Replay_Engine SHALL log the ablation gate result (`PASSED` or `FAILED`) as a single line in the broker log for easy grep.

---

### Requirement 11: Feature and Z-Score Consistency

**User Story:** As a developer, I want the feature computation and normalisation at inference time to be identical to what was used during training, so that the RL model receives inputs from the same distribution it was trained on.

#### Acceptance Criteria

1. THE Inference_Wrapper SHALL use the FEATURE_COLS list imported from `pipeline/features.py` to select and order columns from `df_recent`, ensuring column order matches the training observation space.
2. THE Inference_Wrapper SHALL apply cross-sectional Z_Score normalisation per date across all tickers in `asset_list` before constructing the observation tensor, matching the normalisation applied in `pipeline/data.py` during training.
3. IF a feature column in FEATURE_COLS is missing from `df_recent`, THEN THE Inference_Wrapper SHALL fill that column with 0.0 (the z-scored mean) rather than raising an exception, and SHALL log a warning identifying the missing column.
4. THE Inference_Wrapper SHALL clip all feature values to the range [-5.0, 5.0] after z-scoring, matching the observation space bounds defined in `pipeline/environment.py`.
5. FOR ALL valid `df_recent` inputs, applying the Inference_Wrapper normalisation pipeline twice SHALL produce the same output as applying it once (idempotent normalisation).

---

### Requirement 12: Observability and Diagnostics

**User Story:** As a developer, I want every RL-influenced decision to be logged with both the RL signal and the heuristic signal, so that I can audit and compare the two systems during the integration period.

#### Acceptance Criteria

1. WHEN RL mode is enabled, THE Broker_Brain SHALL include `rl_score`, `composite_score`, and `rl_mode=true` in the `reason` field of every Decision it generates.
2. THE Broker_Brain SHALL write a per-cycle summary to the broker log that includes: number of tickers scored by RL, number of tickers where RL rank differed from heuristic rank, and the top-5 RL-ranked tickers with their scores.
3. WHEN an RL-driven exit is triggered, THE Broker_Brain SHALL log the entry RL_Score, the current RL_Score, the drop magnitude, and the threshold that was crossed.
4. THE Broker_Brain SHALL track and log the number of cycles in which the RL model was successfully invoked versus the number of cycles in which it was skipped or failed, to support reliability monitoring.
