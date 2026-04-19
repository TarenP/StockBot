# Design Document: ML Improvements

## Overview

This document describes the technical design for all 13 ML improvements identified in the trading system audit. The changes are grouped into four critical bug fixes (Requirements 1–4) and nine architectural improvements (Requirements 5–13). Each section identifies the affected files, the precise change required, and the rationale.

The system uses Python with PyTorch for ML components, pandas/numpy for data processing, and yfinance for market data. Property-based tests use the `hypothesis` library (already present in `.hypothesis/` directory).

---

## Architecture

The existing two-stage pipeline is preserved. Changes extend it without restructuring:

```
Market Data (yfinance / stooq parquet)
    ↓
Feature Builder (pipeline/features.py)          ← NEW: market context + fundamental + regime features
    ↓
TickerScorer screener (pipeline/screener.py)    ← NEW: regression labels, meta-model blend
    ↓  top-100 shortlist
PortfolioTransformer RL agent (pipeline/model.py, pipeline/train.py)
    ↓  trained on shortlist universe (NEW)
RL Inference (pipeline/rl_inference.py)         ← NEW: checkpoint ensemble, padding mask
    ↓
BrokerBrain (broker/brain.py)                   ← FIX: rl_score persistence
    ↓
Portfolio execution (broker/broker.py)
```

---

## Components and Interfaces

### 1. BrokerBrain — RL Score Persistence Fix (`broker/brain.py`, `broker/broker.py`)

**Problem:** `rl_score_at_entry` is stored via a `_rl_score_at_entry` attribute on the `Decision` object. The actual write to `portfolio.positions` happens in `broker/broker.py` after `portfolio.buy()`. If `portfolio.buy()` returns `False` (e.g., insufficient cash), the attribute is never written, but the code path in `broker.py` checks `hasattr(d, "_rl_score_at_entry")` without re-checking the buy result. This is fragile.

**Fix:** Move the persistence logic entirely into `broker/broker.py`'s execution loop. After `portfolio.buy()` returns `True`, write `rl_score_at_entry` directly from the `Decision.score` field (which already holds the RL score when `rl_enabled=True`). Remove the `_rl_score_at_entry` attribute pattern from `brain.py`.

```python
# broker/broker.py — in the BUY execution block
ok = portfolio.buy(d.ticker, adj_shares, d.price, d.reason)
if ok and brain.rl_enabled and d.ticker in portfolio.positions:
    portfolio.positions[d.ticker]["rl_score_at_entry"] = d.score
```

The `Decision.score` field is already set to `rl_score_val` when `rl_enabled=True` (see `brain.py` line: `score = rl_score_val if (self.rl_enabled and rl_score_val is not None) else composite`). This eliminates the fragile attribute and makes the persistence unconditional on a successful buy.

---

### 2. Cash Weight Floor — `pipeline/model.py`

**Problem:** `get_weights()` computes `concentration = F.softplus(logits / temperature) + 1e-6` and normalises. The cash logit is a learnable scalar `self.cash_logit`. If the model learns a very negative `cash_logit`, `softplus` approaches 0, and after adding `1e-6` the cash weight is near-zero but positive. However, the asset logits can dominate, effectively pushing cash weight to near-zero. The real risk is that during training the Dirichlet distribution can assign near-zero concentration to cash, allowing the policy to learn to ignore the cash floor entirely.

**Fix:** After computing `concentration`, apply an explicit floor to the cash component before normalisation:

```python
@torch.no_grad()
def get_weights(self, obs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits, _ = self.forward(obs)
    concentration = F.softplus(logits / temperature) + 1e-6
    # Floor cash concentration to prevent negative/zero cash weights
    concentration[:, -1] = torch.clamp(concentration[:, -1], min=1e-6)
    return concentration / concentration.sum(dim=-1, keepdim=True)
```

This ensures the cash weight is always strictly positive and the weights sum to 1.

---

### 3. Sharpe Reward Cold-Start Fix — `pipeline/environment.py`

**Problem:** The `step()` method uses `reward = float(port_ret)` for the first `sharpe_window` steps. This means the agent receives a fundamentally different reward signal during warm-up, creating a non-stationary training objective.

**Fix:** Replace the conditional with a rolling Sharpe that uses `min_periods=1`:

```python
# In step():
window = np.array(self.return_history[-self.sharpe_window:])
mean_r = window.mean()
std_r = window.std() + 1e-9
reward = float(mean_r / std_r)
```

Remove the `if len(self.return_history) >= self.sharpe_window` branch entirely. With a single return, `std_r = 1e-9` (near-zero), so `reward = mean_r / 1e-9` would be huge — we need to clamp:

```python
window = np.array(self.return_history[-self.sharpe_window:])
mean_r = window.mean()
std_r = max(window.std(), 1e-4)   # floor std to prevent explosion on 1 sample
reward = float(np.clip(mean_r / std_r, -10.0, 10.0))
```

This gives reward=0 when mean_r≈0 (first step with near-zero return), and a bounded Sharpe signal thereafter.

---

### 4. Research Score Consistency Fix — `broker/analyst.py`

**Problem:** `research()` calls `_composite_score(report, sent)` which expects raw indicator scales (e.g., `ret_5d` as a raw decimal like 0.03 for 3%). `research_from_features()` calls `_feature_snapshot_score(report)` which uses `np.tanh(value / scale)` — designed for normalised z-scores. The two functions produce scores in [0,1] but with different calibrations, causing 0.1–0.2 divergence.

**Fix:** The two functions are already correctly separated. The fix is to ensure `_feature_snapshot_score` is calibrated to produce scores in the same range as `_composite_score` for typical normalised inputs. Specifically, tune the `_feature_snapshot_score` weights so that the expected score for a neutral ticker (all features = 0) is 0.5, and the range matches `_composite_score`'s range.

The current `_feature_snapshot_score` already starts at 0.5 and adds bounded contributions. The issue is that `_composite_score` uses raw scales (e.g., `0.5 + mom5 * 5`) which can produce very different values for the same underlying signal. The fix is to add a calibration test and adjust the `_feature_snapshot_score` weights to match the empirical distribution of `_composite_score` outputs on the same data.

No code restructuring is needed — the two functions remain separate. The fix is purely in the weight coefficients of `_feature_snapshot_score` to reduce the mean absolute difference below 0.05.

---

### 5. Training/Inference Universe Alignment — `pipeline/train.py`, `Agent.py`

**Design:** Add a `build_shortlist_universe()` function that, for each training date, runs the trained screener to get the top-100 candidates. The RL training then uses this shortlist as its asset universe.

```python
def build_shortlist_universe(
    df_features: pd.DataFrame,
    screener_model: TickerScorer,
    top_n: int = 100,
    device: torch.device = None,
) -> list[str]:
    """Return the union of top-N screener candidates across all training dates."""
    ...
```

The `train_fold()` function gains a `shortlist_universe` parameter. When provided, it overrides the `asset_list` parameter. The checkpoint saves `asset_list` (the shortlist union) for inference compatibility.

**Sequencing:** The screener must be trained before the RL agent. `Agent.py` already trains the screener first (`--mode train_screener`), then the RL agent. The new flow:
1. Train screener on full universe
2. Run screener on training data to get per-date shortlists
3. Take union of top-100 per date → RL training universe (≤ 100 unique tickers)
4. Train RL on this shortlist universe

---

### 6. Checkpoint Ensemble — `pipeline/rl_inference.py`

**Design:** Add an `_load_ensemble()` function that loads all `best_fold*.pt` checkpoints and returns a list of `(model, ckpt)` pairs. The `get_rl_targets()` function averages the softmax weights from all models:

```python
def _load_ensemble(
    models_dir: str,
    device: torch.device,
) -> list[tuple[PortfolioTransformer, dict]]:
    """Load all best_fold*.pt checkpoints from models_dir."""
    import glob
    paths = sorted(glob.glob(os.path.join(models_dir, "best_fold*.pt")))
    ensemble = []
    for path in paths:
        try:
            model, ckpt = _load_model(path, device)
            ensemble.append((model, ckpt))
        except ModelNotAvailableError as e:
            logger.warning("Skipping checkpoint %s: %s", path, e)
    return ensemble
```

The ensemble averaging happens at the weight level (after softmax), not at the logit level, to ensure valid probability distributions:

```python
# Average weights across ensemble members
all_weights = np.stack([
    model.get_weights(obs_t).squeeze(0).cpu().numpy()
    for model, _ in ensemble
], axis=0)
weights_np = all_weights.mean(axis=0)
```

The `checkpoint_path` parameter in `get_rl_targets()` is repurposed as `models_dir` (defaulting to `"models"`), with backward compatibility for explicit `.pt` paths.

---

### 7. Market-Wide Context Features — `pipeline/features.py`

**Design:** Add a `build_market_context()` function that fetches SPY prices and VIX from yfinance, and computes market breadth from the existing panel data:

```python
def build_market_context(
    df: pd.DataFrame,
    spy_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    breadth_window: int = 200,
    spy_ret_window: int = 20,
) -> pd.DataFrame:
    """
    Returns a date-indexed DataFrame with columns:
      spy_ret_20d, vix_level, market_breadth
    """
    ...
```

These market-wide features are then broadcast to all tickers on each date before being appended to `FEATURE_COLS`. The broadcast is done by merging on the date level of the MultiIndex.

**VIX normalisation:** VIX is divided by 100 to bring it into a similar scale as other features (typical range 0.10–0.80).

**Market breadth:** Computed from the existing panel — for each date, count the fraction of tickers where `close > rolling_200d_mean(close)`. This avoids an additional API call.

---

### 8. Screener Regression Label — `pipeline/screener.py`

**Design:** Replace the binary label `(fwd_rets >= threshold).astype(np.float32)` with a percentile rank:

```python
from scipy.stats import rankdata
rank_pct = rankdata(fwd_rets_vec[valid_mask], method='average') / valid_mask.sum()
labels_vec = np.zeros(len(fwd_rets_vec), dtype=np.float32)
labels_vec[valid_mask] = rank_pct.astype(np.float32)
```

The loss function changes from `BCEWithLogitsLoss` to `MSELoss`. The model output is passed through `sigmoid` to keep scores in [0,1]:

```python
criterion = nn.MSELoss()
# In training loop:
pred_prob = torch.sigmoid(model(xb))
loss = criterion(pred_prob, yb)  # yb is now rank percentile in [0,1]
```

The `_evaluate_ranked_groups()` function is unchanged — it still uses the binary top-10% label for evaluation (the `yo` array), while training uses the continuous rank target. This preserves the existing checkpoint selection metric.

---

### 9. Screener Blend Meta-Model — `pipeline/screener.py`

**Design:** After the main training loop, train a 2-feature logistic regression:

```python
from sklearn.linear_model import LogisticRegression

def _train_meta_model(
    neural_probs: np.ndarray,   # shape (N,)
    heuristic_probs: np.ndarray, # shape (N,)
    labels: np.ndarray,          # shape (N,) binary top-10%
) -> LogisticRegression:
    X = np.stack([neural_probs, heuristic_probs], axis=1)
    meta = LogisticRegression(C=1.0, max_iter=200)
    meta.fit(X, labels)
    return meta
```

The meta-model is saved as `{"coef": meta.coef_, "intercept": meta.intercept_}` in the screener checkpoint. At inference, `run_screener()` reconstructs the meta-model from these arrays and uses it to blend scores.

**Fallback:** If the checkpoint lacks `meta_model_coef`, fall back to the static `blend_weight`.

---

### 10. Fundamental Features — `pipeline/features.py`

**Design:** Add a `fetch_fundamentals()` function with a file-based cache:

```python
FUNDAMENTALS_CACHE = "models/fundamentals_cache.json"
FUNDAMENTALS_STALENESS_HOURS = 24

def fetch_fundamentals(
    tickers: list[str],
    cache_path: str = FUNDAMENTALS_CACHE,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by ticker with columns:
      pe_ratio, revenue_growth, short_interest_pct
    Fetches from yfinance with 24-hour cache.
    """
    ...
```

The fundamentals are fetched once per training run and broadcast to all dates for each ticker (since they change slowly). The `build_features()` function merges fundamentals by ticker before computing cross-sectional z-scores.

**Missing data handling:** `pe_ratio` defaults to 0.0 (neutral), `revenue_growth` defaults to 0.0, `short_interest_pct` defaults to 0.0. Forward-fill within each ticker's time series for up to 20 trading days.

---

### 11. Regime Detection — `pipeline/features.py`

**Design:** Add a `compute_regimes()` function:

```python
def compute_regimes(
    spy_returns: pd.Series,
    vol_window: int = 20,
    corr_window: int = 20,
    vol_threshold: float = 0.015,   # ~24% annualised
) -> pd.Series:
    """
    Returns date-indexed Series of regime labels (0-3):
      0: low vol, low corr  (calm bull)
      1: low vol, high corr (trending bull)
      2: high vol, low corr (choppy/rotation)
      3: high vol, high corr (risk-off / bear)
    """
    rolling_vol = spy_returns.rolling(vol_window).std()
    # corr proxy: use cross-sectional return dispersion from panel
    high_vol = rolling_vol > vol_threshold
    # For correlation proxy, use average pairwise correlation of top-50 tickers
    # (computed from the panel data already available)
    ...
```

The regime label is one-hot encoded into 4 binary columns and broadcast to all tickers on each date.

---

### 12. Zero-Padding Mask — `pipeline/rl_inference.py`, `pipeline/model.py`

**Design:** `_build_obs_tensor()` returns a mask alongside the tensor:

```python
def _build_obs_tensor(...) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    ...
    # Build padding mask: True = padded (should be ignored)
    # Shape: (1, n_assets, lookback) → reshaped to (n_assets, lookback) for TransformerEncoder
    padding_mask = None
    if insufficient:
        mask_np = np.zeros((1, n_assets, lookback), dtype=bool)
        for ticker in insufficient:
            ai = asset_map.get(ticker)
            if ai is not None:
                # All time steps for this asset are padded
                mask_np[0, ai, :] = True
        padding_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    return obs_t, insufficient, padding_mask
```

`AssetEncoder.forward()` gains an optional `src_key_padding_mask` parameter:

```python
def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
    x = self.input_proj(x)
    x = self.pos_enc(x)
    x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
    return self.norm(x[:, -1, :])
```

The mask is reshaped from `(batch, n_assets, lookback)` to `(batch * n_assets, lookback)` before being passed to `AssetEncoder`.

---

### 13. Extended RL Training with Curriculum — `pipeline/train.py`

**Design:** `PPO_CFG["total_steps"]` increases to 500,000. The `train_fold()` function gains a `curriculum: bool = False` parameter. When enabled:

```python
if curriculum:
    # Phase 1: small universe (top-20 by screener score)
    small_asset_list = asset_list[:20]
    train_env = PortfolioEnv(df_train, small_asset_list, ...)
    curriculum_switch_step = cfg["total_steps"] // 2

    # In training loop:
    if curriculum and steps_done >= curriculum_switch_step and using_small_universe:
        train_env = PortfolioEnv(df_train, asset_list, ...)  # full universe
        using_small_universe = False
        tqdm.write(f"  Curriculum: expanded universe to {len(asset_list)} assets at step {steps_done:,}")
```

The model architecture is fixed at `n_assets = len(asset_list)` (full size) from the start, so the curriculum only affects the environment's asset list, not the model dimensions. The model simply receives zero-padded observations for the unused asset slots during Phase 1.

---

## Data Models

### Updated `FEATURE_COLS` (pipeline/features.py)

```python
FEATURE_COLS = [
    # Existing technical features
    "ret_1d", "ret_5d", "ret_20d",
    "rsi", "macd_hist", "bb_pct", "atr",
    "vol_ratio", "vol_zscore", "price_pos_52w",
    # Existing sentiment features
    "sent_net", "sent_ma3", "sent_ma7", "sent_ma14",
    "sent_surprise", "sent_accel", "sent_trend",
    "sent_pos_raw", "sent_neg_spike",
    # NEW: market context features (Req 7)
    "spy_ret_20d", "vix_level", "market_breadth",
    # NEW: fundamental features (Req 10)
    "pe_ratio", "revenue_growth", "short_interest_pct",
    # NEW: regime features (Req 11)
    "regime_0", "regime_1", "regime_2", "regime_3",
]
```

Total: 19 existing + 3 market context + 3 fundamental + 4 regime = **29 features**.

### Screener Checkpoint Schema (models/screener.pt)

```python
{
    "model_state": ...,
    "n_features": int,
    "feature_cols": list[str],
    "lookback": int,
    "forward_days": int,
    "top_pct": float,
    "eval_top_n": int,
    "blend_weight": float,          # legacy fallback
    "meta_model_coef": np.ndarray,  # NEW: shape (1, 2)
    "meta_model_intercept": np.ndarray,  # NEW: shape (1,)
    "val_metrics": dict,
    "test_metrics": dict,
    "label_type": str,              # NEW: "regression" or "binary"
}
```

### RL Checkpoint Schema (models/best_fold*.pt)

```python
{
    "model_state": dict,
    "model_cfg": dict,
    "fold": int,
    "steps": int,
    "val_sharpe": float,
    "val_return": float,
    "top_n": int,
    "asset_list": list[str],        # NEW: shortlist-derived universe
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: RL Score Round-Trip Persistence

*For any* ticker and any RL score value in [0, 1], when a BUY is executed successfully with `rl_enabled=True`, the value stored in `portfolio.positions[ticker]["rl_score_at_entry"]` SHALL equal the RL score that was passed to the buy path.

**Validates: Requirements 1.1, 1.3**

---

### Property 2: Cash Weight Non-Negativity

*For any* observation tensor of valid shape `(batch, lookback, n_assets, n_features)`, the cash weight returned by `PortfolioTransformer.get_weights()` SHALL be greater than or equal to zero.

**Validates: Requirements 2.1, 2.2, 2.3**

---

### Property 3: Portfolio Weights Sum to One

*For any* observation tensor of valid shape, the sum of all weights returned by `PortfolioTransformer.get_weights()` SHALL equal 1.0 within a tolerance of 1e-5.

**Validates: Requirements 2.4**

---

### Property 4: Consistent Sharpe Reward Formula

*For any* sequence of portfolio returns of length 1 to `sharpe_window`, the reward returned by `PortfolioEnv.step()` SHALL be computed using the rolling Sharpe formula (mean / std) on all available returns, never the raw single-step return.

**Validates: Requirements 3.1, 3.2, 3.4**

---

### Property 5: Research Score Calibration

*For any* feature vector with values in the normalised range [-3, 3], the absolute difference between `_composite_score()` (applied to de-normalised equivalents) and `_feature_snapshot_score()` (applied to normalised values) SHALL be less than 0.05.

**Validates: Requirements 4.3**

---

### Property 6: Screener Output Range

*For any* input observation array of valid shape, all scores returned by `run_screener()` SHALL be in the range [0.0, 1.0].

**Validates: Requirements 8.5, 9.3**

---

### Property 7: Regression Label Validity

*For any* cross-section of forward returns with at least 2 valid tickers, the percentile rank labels produced by `_build_samples()` SHALL all be in [0.0, 1.0] and SHALL be monotonically ordered with the underlying forward returns (higher return → higher rank).

**Validates: Requirements 8.1**

---

### Property 8: Ensemble Averaging Correctness

*For any* set of N PortfolioTransformer models with identical architecture, the ensemble weight vector produced by `get_rl_targets()` SHALL equal the element-wise mean of the N individual weight vectors within a tolerance of 1e-6.

**Validates: Requirements 6.2**

---

### Property 9: VIX Fill Behaviour

*For any* VIX time series with gaps of length 1 to 5, the filled series SHALL use the last known value for positions within 5 days of the gap start, and 0.0 for positions beyond 5 days.

**Validates: Requirements 7.4**

---

### Property 10: Regime Label Validity

*For any* (rolling_volatility, rolling_correlation) pair, the regime label produced by `compute_regimes()` SHALL be an integer in {0, 1, 2, 3}.

**Validates: Requirements 11.1**

---

### Property 11: Regime Broadcast Consistency

*For any* date in the feature DataFrame, all tickers on that date SHALL have identical values for `regime_0`, `regime_1`, `regime_2`, `regime_3`.

**Validates: Requirements 11.5**

---

### Property 12: Padding Mask Shape Invariant

*For any* observation tensor of shape `(batch, lookback, n_assets, n_features)` with at least one asset having insufficient history, the padding mask produced by `_build_obs_tensor()` SHALL have shape `(batch * n_assets, lookback)`.

**Validates: Requirements 12.1, 12.4**

---

### Property 13: Curriculum Universe Expansion

*For any* training run with `curriculum=True` and `total_steps > 0`, the training environment SHALL use an asset list of size 20 for all steps before `total_steps // 2`, and the full asset list for all steps from `total_steps // 2` onward.

**Validates: Requirements 13.2**

---

### Property 14: Fundamental Fill Behaviour

*For any* fundamental data series with gaps of length 1 to 20, the filled series SHALL use the last known value for positions within 20 days of the gap start, and 0.0 for positions beyond 20 days.

**Validates: Requirements 10.4**

---

## Error Handling

| Scenario | Component | Handling |
|---|---|---|
| `portfolio.buy()` returns False | `broker/broker.py` | Do not write `rl_score_at_entry`; log at DEBUG level |
| Checkpoint file corrupt | `pipeline/rl_inference.py` | Skip checkpoint, log WARNING, continue with remaining |
| All checkpoints corrupt | `pipeline/rl_inference.py` | Raise `ModelNotAvailableError` |
| VIX fetch fails | `pipeline/features.py` | Forward-fill ≤5 days, then fill 0.0; log WARNING |
| Market breadth unavailable | `pipeline/features.py` | Forward-fill ≤5 days, then fill 0.5; log WARNING |
| Fundamental fetch fails for ticker | `pipeline/features.py` | Fill 0.0 for all fundamental columns; log DEBUG |
| Screener not trained before RL | `pipeline/train.py` | Raise `FileNotFoundError` with clear message |
| Meta-model absent in checkpoint | `pipeline/screener.py` | Fall back to static `blend_weight`; log INFO |
| Curriculum with asset_list < 20 | `pipeline/train.py` | Use full asset_list from start; log WARNING |

---

## Testing Strategy

### Unit Tests (example-based)

- `tests/test_rl_score_persistence.py`: Verify `rl_score_at_entry` is written on successful buy and not written on failed buy.
- `tests/test_cash_weight_floor.py`: Verify cash weight ≥ 0 for adversarial logit inputs.
- `tests/test_sharpe_reward.py`: Verify reward formula is consistent across all step counts.
- `tests/test_research_score_consistency.py`: Verify `_composite_score` and `_feature_snapshot_score` are called by the correct research paths.
- `tests/test_screener_regression.py`: Verify regression labels are in [0,1] and correctly ordered.
- `tests/test_checkpoint_ensemble.py`: Verify ensemble of 1 matches single-model inference.
- `tests/test_feature_cols.py`: Verify `FEATURE_COLS` contains all new columns.
- `tests/test_regime_classifier.py`: Verify regime labels are in {0,1,2,3} and one-hot encoding is correct.
- `tests/test_padding_mask.py`: Verify mask shape and values for tickers with insufficient history.
- `tests/test_curriculum.py`: Verify `train_fold()` accepts `curriculum` parameter and defaults to False.

### Property-Based Tests (hypothesis)

Each property in the Correctness Properties section maps to one property-based test using `hypothesis`. Tests are configured with `@settings(max_examples=100)`.

- **Property 1** → `test_rl_score_round_trip_persistence` in `tests/test_rl_score_persistence.py`
- **Property 2** → `test_cash_weight_non_negative` in `tests/test_cash_weight_floor.py`
- **Property 3** → `test_weights_sum_to_one` in `tests/test_cash_weight_floor.py`
- **Property 4** → `test_sharpe_reward_consistent` in `tests/test_sharpe_reward.py`
- **Property 5** → `test_research_score_calibration` in `tests/test_research_score_consistency.py`
- **Property 6** → `test_screener_output_range` in `tests/test_screener_regression.py`
- **Property 7** → `test_regression_label_validity` in `tests/test_screener_regression.py`
- **Property 8** → `test_ensemble_averaging` in `tests/test_checkpoint_ensemble.py`
- **Property 9** → `test_vix_fill_behaviour` in `tests/test_market_context_features.py`
- **Property 10** → `test_regime_label_validity` in `tests/test_regime_classifier.py`
- **Property 11** → `test_regime_broadcast_consistency` in `tests/test_regime_classifier.py`
- **Property 12** → `test_padding_mask_shape` in `tests/test_padding_mask.py`
- **Property 13** → `test_curriculum_universe_expansion` in `tests/test_curriculum.py`
- **Property 14** → `test_fundamental_fill_behaviour` in `tests/test_fundamental_features.py`

### Integration Tests

- Verify screener trains before RL in `Agent.py --mode train` flow.
- Verify ensemble loads all available checkpoints from `models/`.
- Verify fundamental cache is written and read correctly.
- Verify market context features are present in the feature DataFrame after `build_features()`.

### Tag Format

Each property test is tagged with:
```python
# Feature: ml-improvements, Property N: <property_text>
```
