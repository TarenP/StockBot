# Requirements Document

## Introduction

This document captures requirements for fixing all ML weaknesses identified in the trading system audit. The changes span four critical bug fixes and seven architectural improvements, ordered by priority. Critical fixes address correctness bugs that silently corrupt live trading behaviour. Architectural improvements reduce train/inference distribution mismatch, improve model robustness, and add richer market context.

The system under improvement is a two-stage ML pipeline: a `TickerScorer` screener (GRU-based binary classifier) feeds a shortlist to a `PortfolioTransformer` RL agent (PPO-trained Transformer) that ranks candidates for the broker.

---

## Glossary

- **BrokerBrain**: The decision engine in `broker/brain.py` that orchestrates screening, RL scoring, and trade execution.
- **PortfolioTransformer**: The Transformer-based RL policy network in `pipeline/model.py`.
- **PortfolioEnv**: The Gymnasium environment in `pipeline/environment.py` used for RL training.
- **TickerScorer**: The GRU-based screener model in `pipeline/screener.py`.
- **rl_score_at_entry**: The RL conviction score recorded when a position is opened, used later for conviction-drop exit checks.
- **Checkpoint**: A saved `.pt` file containing model weights and metadata produced by `pipeline/train.py`.
- **Fold**: One walk-forward training split; the system trains one checkpoint per fold.
- **Shortlist**: The top-N tickers returned by the screener, passed to the RL agent for ranking.
- **Composite Score**: The heuristic signal score computed by `_composite_score()` in `broker/analyst.py` on raw (un-normalised) features.
- **Feature Snapshot Score**: The fallback signal score computed by `_feature_snapshot_score()` in `broker/analyst.py` on cross-sectionally normalised features.
- **SPY**: The S&P 500 ETF used as the market benchmark.
- **VIX**: The CBOE Volatility Index, a measure of implied market volatility.
- **Market Breadth**: The percentage of S&P 500 constituents trading above their 200-day moving average.
- **Regime**: A discrete market state (e.g., low-vol bull, high-vol bear) inferred from rolling statistics.
- **Meta-Model**: A small logistic regression model that learns to blend neural and heuristic screener scores.
- **Curriculum**: A training schedule that starts with a small asset universe and gradually expands it.
- **Zero-Padding**: Filling missing observations with zeros when a ticker has fewer than `lookback` days of history.

---

## Requirements

### Requirement 1: RL Score Persistence Fix

**User Story:** As a broker operator, I want the RL conviction score at entry to be stored reliably in the position record, so that conviction-drop exit checks have accurate baseline scores to compare against.

#### Acceptance Criteria

1. WHEN a BUY decision is executed and `rl_enabled` is true, THE BrokerBrain SHALL store the `rl_score_at_entry` value directly into `portfolio.positions[ticker]` immediately after `portfolio.buy()` returns successfully.
2. THE BrokerBrain SHALL NOT rely on a `_rl_score_at_entry` attribute attached to the `Decision` object as the sole persistence mechanism for `rl_score_at_entry`.
3. WHEN `_rl_exit_checks()` reads `pos.get("rl_score_at_entry")` for a held position, THE value SHALL match the RL score that was active at the time of entry.
4. IF `portfolio.buy()` returns False for a ticker, THEN THE BrokerBrain SHALL NOT write `rl_score_at_entry` to `portfolio.positions` for that ticker.

---

### Requirement 2: Cash Weight Floor Constraint

**User Story:** As a system designer, I want the PortfolioTransformer to be prevented from learning negative cash weights, so that the RL agent cannot implicitly model leverage during training or inference.

#### Acceptance Criteria

1. WHEN `PortfolioTransformer.get_weights()` computes portfolio weights, THE PortfolioTransformer SHALL apply a floor of zero to the cash weight before normalisation.
2. THE PortfolioTransformer SHALL ensure the cash weight in the returned weight vector is greater than or equal to zero for all valid inputs.
3. WHEN the raw cash logit produces a negative concentration value after `softplus`, THE PortfolioTransformer SHALL clamp the cash concentration to a minimum of `1e-6` before normalisation.
4. THE PortfolioTransformer SHALL ensure the sum of all returned weights equals 1.0 within a tolerance of `1e-5`.

---

### Requirement 3: Sharpe Reward Cold-Start Fix

**User Story:** As an ML engineer, I want the RL training environment to use Sharpe-based rewards from the very first step, so that the agent receives consistent reward signals throughout training without a cold-start period that uses a different reward function.

#### Acceptance Criteria

1. WHEN `PortfolioEnv.step()` is called and fewer than `sharpe_window` returns have been accumulated, THE PortfolioEnv SHALL compute the Sharpe reward using all available returns with `min_periods=1`.
2. THE PortfolioEnv SHALL NOT fall back to using the raw single-step portfolio return as the reward during the first `sharpe_window` steps.
3. WHEN only one return has been accumulated, THE PortfolioEnv SHALL return a Sharpe reward of 0.0 (mean divided by a near-zero std is clamped, not the raw return).
4. THE PortfolioEnv SHALL use the same reward formula for all steps regardless of how many returns have been accumulated.

---

### Requirement 4: Research Score Consistency Fix

**User Story:** As a broker operator, I want the live research path and the local-fallback research path to produce comparable composite scores, so that the broker's entry threshold behaves consistently regardless of which research path is used.

#### Acceptance Criteria

1. WHEN `research()` computes a composite score on raw (un-normalised) features, THE Analyst SHALL use `_composite_score()` with raw indicator scales.
2. WHEN `research_from_features()` computes a composite score on cross-sectionally normalised features, THE Analyst SHALL use `_feature_snapshot_score()` with normalised-feature-aware scaling.
3. THE difference between the composite score produced by `research()` and the composite score produced by `research_from_features()` for the same ticker on the same date SHALL be less than 0.05 on average across a representative test set.
4. WHEN both research paths are available for the same ticker, THE BrokerBrain SHALL prefer the live `research()` path and use `research_from_features()` only as a fallback.

---

### Requirement 5: Training/Inference Universe Alignment

**User Story:** As an ML engineer, I want the RL model to be trained on the same distribution of assets it will score at inference time, so that the model's learned representations are relevant to the actual shortlist it evaluates.

#### Acceptance Criteria

1. WHEN `train_fold()` is called, THE Trainer SHALL use the screener shortlist (top 100 candidates per training date) as the RL training universe rather than the full top-N universe.
2. THE Trainer SHALL produce a screener shortlist for each training date by running the trained `TickerScorer` on the available feature data for that date.
3. WHEN the screener shortlist size varies across training dates, THE PortfolioEnv SHALL accept a dynamic asset list of up to 100 assets.
4. THE Trainer SHALL save the shortlist-derived asset list in the checkpoint so that inference can verify universe compatibility.
5. WHEN `get_rl_targets()` is called at inference time with a shortlist of up to 100 assets, THE RL Inference module SHALL produce valid scores without requiring retraining.

---

### Requirement 6: Checkpoint Ensemble

**User Story:** As an ML engineer, I want the RL inference to average weights across all available fold checkpoints, so that the ensemble produces more stable and better out-of-sample scores than any single fold checkpoint.

#### Acceptance Criteria

1. WHEN `get_rl_targets()` is called and multiple fold checkpoints exist in `models/`, THE RL Inference module SHALL load all available `best_fold*.pt` checkpoints.
2. THE RL Inference module SHALL compute softmax weights from each checkpoint's actor logits and average them before producing the final score.
3. WHEN only one checkpoint is available, THE RL Inference module SHALL behave identically to the current single-checkpoint inference.
4. THE RL Inference module SHALL log the number of checkpoints used in the ensemble.
5. WHEN a checkpoint file is corrupt or fails to load, THE RL Inference module SHALL skip that checkpoint, log a warning, and continue with the remaining checkpoints.

---

### Requirement 7: Market-Wide Context Features

**User Story:** As an ML engineer, I want the screener and RL agent to observe market-wide context signals, so that both models can condition their predictions on the current market environment.

#### Acceptance Criteria

1. THE Feature Builder SHALL compute and include SPY 20-day return as a feature column available to both the screener and the RL observation.
2. THE Feature Builder SHALL compute and include VIX level (fetched from yfinance ticker `^VIX`) as a feature column.
3. THE Feature Builder SHALL compute and include market breadth (percentage of S&P 500 constituents trading above their 200-day moving average) as a feature column.
4. WHEN VIX data is unavailable for a given date, THE Feature Builder SHALL forward-fill the last known VIX value for up to 5 trading days before filling with 0.0.
5. WHEN market breadth data is unavailable, THE Feature Builder SHALL forward-fill the last known value for up to 5 trading days before filling with 0.5.
6. THE updated `FEATURE_COLS` list SHALL include `spy_ret_20d`, `vix_level`, and `market_breadth` so that both the screener and RL observation tensors include these columns.

---

### Requirement 8: Screener Regression Label

**User Story:** As an ML engineer, I want the screener to be trained on a continuous regression target (forward return rank as a 0–1 percentile) rather than a binary top-10% label, so that the model receives richer gradient signal and learns finer-grained ranking.

#### Acceptance Criteria

1. WHEN `_build_samples()` constructs training labels, THE Screener Trainer SHALL compute the forward return rank percentile (0.0 to 1.0) for each ticker within each cross-section as the regression target.
2. THE Screener Trainer SHALL use mean squared error loss (or a ranking-aware loss) instead of binary cross-entropy when training with regression targets.
3. WHEN evaluating shortlist quality, THE Screener Trainer SHALL continue to use the existing `_evaluate_ranked_groups()` metric (precision@K, recall@K, lift@K) for checkpoint selection.
4. THE `TickerScorer` model output SHALL remain a single scalar per ticker, interpretable as a ranking score.
5. WHEN `run_screener()` produces scores, THE scores SHALL remain in the range [0, 1] so that downstream blending and thresholding are unaffected.

---

### Requirement 9: Screener Blend Meta-Model

**User Story:** As an ML engineer, I want a small logistic regression meta-model to replace the static blend weight between neural and heuristic screener scores, so that the blend adapts to market conditions and is retrained monthly.

#### Acceptance Criteria

1. THE Screener Trainer SHALL train a logistic regression meta-model that takes as input the neural score and heuristic score for each sample and predicts the binary top-10% label.
2. THE meta-model SHALL be trained on the validation set after the neural model has converged, using the neural and heuristic scores as features.
3. WHEN `run_screener()` blends scores, THE Screener SHALL use the meta-model's predicted probability as the final blended score instead of the static linear blend.
4. THE meta-model weights SHALL be saved alongside the screener checkpoint in `models/screener.pt`.
5. WHEN the meta-model is unavailable (e.g., legacy checkpoint), THE Screener SHALL fall back to the static blend weight stored in the checkpoint.
6. THE meta-model SHALL be retrained whenever the screener is retrained (i.e., monthly or on-demand via `python Agent.py --mode train_screener`).

---

### Requirement 10: Fundamental Features

**User Story:** As an ML engineer, I want P/E ratio, revenue growth YoY, and short interest percentage to be added to the feature set, so that both the screener and RL agent can incorporate fundamental valuation signals.

#### Acceptance Criteria

1. THE Feature Builder SHALL fetch P/E ratio (`trailingPE`) from yfinance for each ticker and include it as feature column `pe_ratio`.
2. THE Feature Builder SHALL fetch revenue growth YoY (`revenueGrowth`) from yfinance and include it as feature column `revenue_growth`.
3. THE Feature Builder SHALL fetch short interest as a percentage of float (`shortPercentOfFloat`) from yfinance and include it as feature column `short_interest_pct`.
4. WHEN fundamental data is unavailable for a ticker on a given date, THE Feature Builder SHALL forward-fill the last known value for up to 20 trading days before filling with 0.0.
5. THE updated `FEATURE_COLS` list SHALL include `pe_ratio`, `revenue_growth`, and `short_interest_pct`.
6. THE fundamental data fetch SHALL be cached per ticker with a staleness threshold of 24 hours to avoid redundant API calls during a single training run.

---

### Requirement 11: Regime Detection

**User Story:** As an ML engineer, I want a simple regime classifier to condition the RL policy on the current market regime, so that the agent can adapt its behaviour to different volatility and correlation environments.

#### Acceptance Criteria

1. THE Regime Classifier SHALL assign one of four discrete regime labels (0–3) to each trading date based on rolling 20-day realised volatility of SPY and rolling 20-day average pairwise correlation of S&P 500 constituents.
2. THE Regime Classifier SHALL be implemented as a rule-based classifier with fixed thresholds (e.g., low/high volatility × low/high correlation), not a learned model.
3. THE Feature Builder SHALL include the current regime label as a one-hot encoded feature (4 binary columns: `regime_0`, `regime_1`, `regime_2`, `regime_3`) in `FEATURE_COLS`.
4. WHEN regime data is unavailable for a given date, THE Feature Builder SHALL use the most recent available regime label.
5. THE regime features SHALL be broadcast to all tickers on the same date so that the RL agent observes the same market-wide regime signal across all assets in its observation tensor.

---

### Requirement 12: Zero-Padding Mask

**User Story:** As an ML engineer, I want zero-padded observations in the RL inference to be masked so that the model does not attend to padding tokens, reducing spurious patterns learned from padding.

#### Acceptance Criteria

1. WHEN `_build_obs_tensor()` constructs the observation tensor for tickers with fewer than `lookback` days of history, THE RL Inference module SHALL produce a boolean padding mask indicating which time steps are zero-padded.
2. THE PortfolioTransformer's `AssetEncoder` SHALL accept an optional `src_key_padding_mask` and pass it to the `TransformerEncoder` to suppress attention over padded positions.
3. WHEN no tickers have insufficient history, THE RL Inference module SHALL pass `None` as the mask, preserving existing behaviour.
4. THE padding mask SHALL be shaped `(batch * n_assets, lookback)` with `True` indicating positions that should be ignored by the attention mechanism.

---

### Requirement 13: Extended RL Training with Curriculum

**User Story:** As an ML engineer, I want the RL training to use more steps and a curriculum that starts with a small universe and expands it, so that the agent learns stable policies before being exposed to the full asset complexity.

#### Acceptance Criteria

1. THE Trainer SHALL increase the default `total_steps` in `PPO_CFG` from 100,000 to 500,000.
2. WHEN curriculum training is enabled, THE Trainer SHALL begin each fold with a universe of 20 assets (the top-20 by screener score) and expand to the full shortlist size at 50% of `total_steps`.
3. WHEN curriculum training is disabled (default), THE Trainer SHALL use the full asset list from the start, preserving backward compatibility.
4. THE curriculum expansion SHALL be logged at the step where the universe size changes.
5. THE `train_fold()` function SHALL accept a `curriculum` boolean parameter (default `False`) to enable or disable curriculum training.

