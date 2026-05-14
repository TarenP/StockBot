"""
RL Inference Wrapper
====================

Provides `get_rl_targets()` — a reusable function that loads a
PortfolioTransformer checkpoint and runs a forward pass to produce
per-ticker RL scores or portfolio weights.

Extracted and generalised from Agent.py run_predict() so that any
component (broker, replay, backtest) can obtain RL signals without
duplicating inference logic.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from pipeline.action_projection import normalize_projection_settings, projection_kwargs
from pipeline.features import FEATURE_COLS
from pipeline.model import PortfolioTransformer

logger = logging.getLogger(__name__)

# Module-level model cache: (checkpoint_path, device_str) → model
_MODEL_CACHE: dict[tuple[str, str], PortfolioTransformer] = {}


# ── Exception ─────────────────────────────────────────────────────────────────

class ModelNotAvailableError(Exception):
    """Raised when the RL checkpoint cannot be loaded or does not exist."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cross-sectional z-score per date across all tickers.
    Matches the normalisation in pipeline/data.py build_features().
    Returns a new DataFrame with the same index/columns.
    """
    grp = df.groupby(level="date")
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, 1e-9).fillna(1e-9)
    return (df - mean) / std


def _weights_to_rank_scores(asset_weights: np.ndarray) -> np.ndarray:
    """
    Convert per-asset model weights into shortlist rank percentiles.

    We keep zero for assets with zero/no conviction so callers can distinguish
    "unscored" names from genuinely ranked candidates. Positive-weight assets
    are mapped to percentiles in (0, 1], where 1.0 is the top-ranked ticker.
    """
    weights = np.asarray(asset_weights, dtype=float)
    scores = np.zeros_like(weights, dtype=float)
    positive_mask = np.isfinite(weights) & (weights > 0.0)
    if not positive_mask.any():
        return scores

    ranked = (
        pd.Series(weights[positive_mask], dtype=float)
        .rank(method="average", ascending=True)
        .to_numpy(dtype=float)
    )
    scores[positive_mask] = ranked / float(len(ranked))
    return scores


def _normalize_asset_weights(asset_weights: np.ndarray) -> np.ndarray:
    """
    Normalize positive asset weights onto [0, 1] by total positive exposure.

    Zero or non-finite weights remain at 0.0 so callers can distinguish names
    that received no usable RL conviction.
    """
    weights = np.asarray(asset_weights, dtype=float)
    normalized = np.zeros_like(weights, dtype=float)
    positive_mask = np.isfinite(weights) & (weights > 0.0)
    total_positive = float(weights[positive_mask].sum())
    if total_positive <= 0.0:
        return normalized
    normalized[positive_mask] = weights[positive_mask] / total_positive
    return normalized


def _load_projection_config(path: str = "broker.config") -> dict:
    cfg = {}
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key not in {"rl_action_projection", "rl_action_temperature", "rl_action_top_k"}:
                    continue
                cfg[key] = value.split("#")[0].strip()
    except Exception:
        return {}
    return cfg


def _resolve_projection_settings(
    checkpoint_meta: dict | None = None,
    rl_action_projection: str | None = None,
    rl_action_temperature: float | None = None,
    rl_action_top_k: int | None = None,
) -> dict:
    config = _load_projection_config()

    def choose(explicit, key):
        if explicit is not None:
            return explicit
        if config.get(key) is not None:
            return config[key]
        if checkpoint_meta is not None and checkpoint_meta.get(key) is not None:
            return checkpoint_meta[key]
        return None

    return normalize_projection_settings(
        projection=choose(rl_action_projection, "rl_action_projection"),
        temperature=choose(rl_action_temperature, "rl_action_temperature"),
        top_k=choose(rl_action_top_k, "rl_action_top_k"),
    )


def _load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PortfolioTransformer, dict]:
    """
    Load (or retrieve from cache) a PortfolioTransformer from a checkpoint.
    Returns (model, checkpoint_meta).
    Raises ModelNotAvailableError on any failure.
    """
    device_str = str(device)
    cache_key = (checkpoint_path, device_str)

    if cache_key in _MODEL_CACHE:
        # Return cached model; we still need meta for top_n etc.
        # Re-load meta cheaply (weights_only=True for meta dict is fine)
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelNotAvailableError(
                f"Failed to load checkpoint '{checkpoint_path}': {exc}"
            ) from exc
        return _MODEL_CACHE[cache_key], ckpt

    if not os.path.exists(checkpoint_path):
        raise ModelNotAvailableError(
            f"Checkpoint not found: '{checkpoint_path}'"
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ModelNotAvailableError(
            f"Failed to load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    model_cfg = ckpt.get("model_cfg", {})
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ModelNotAvailableError(
            f"Checkpoint '{checkpoint_path}' is missing 'model_state'."
        )

    try:
        model = PortfolioTransformer(**model_cfg)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelNotAvailableError(
            f"Failed to instantiate model from checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    _MODEL_CACHE[cache_key] = model
    return model, ckpt


def _build_obs_tensor(
    df_recent: pd.DataFrame,
    asset_list: list[str],
    feature_cols: list[str],
    lookback: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], "torch.Tensor | None"]:
    """
    Build observation tensor (1, lookback, n_assets, n_features).

    - Aligns df_recent to asset_list using feature_cols column order.
    - Applies cross-sectional z-score per date.
    - Clips to [-5.0, 5.0].
    - Pads with zeros for tickers with insufficient history.

    Returns (obs_tensor, tickers_with_insufficient_history, padding_mask).
    padding_mask is a bool tensor of shape (1, n_assets, lookback) with True
    for padded positions, or None if no tickers are padded.
    """
    n_assets = len(asset_list)
    n_features = len(feature_cols)

    # ── Warn about missing feature columns ───────────────────────────────────
    for col in feature_cols:
        if col not in df_recent.columns:
            logger.warning(
                "Feature column '%s' missing from df_recent — filling with 0.0", col
            )

    # ── Ensure all feature columns exist (fill missing with 0.0) ─────────────
    df = df_recent.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]

    # ── Apply cross-sectional z-score per date ────────────────────────────────
    df = _zscore_df(df)
    df = df.clip(-5.0, 5.0)
    df = df.fillna(0.0)

    # ── Collect sorted dates ──────────────────────────────────────────────────
    all_dates = sorted(df.index.get_level_values("date").unique())
    recent_dates = all_dates[-lookback:] if len(all_dates) >= lookback else all_dates

    obs = np.zeros((lookback, n_assets, n_features), dtype=np.float32)
    asset_map = {a: i for i, a in enumerate(asset_list)}
    insufficient = []

    # Determine which tickers have enough history
    for ticker in asset_list:
        try:
            ticker_dates = df.xs(ticker, level="ticker").index
        except KeyError:
            ticker_dates = pd.Index([])
        if len(ticker_dates) < lookback:
            insufficient.append(ticker)

    # Fill obs array — tickers with insufficient history stay as zeros
    for t_idx, date in enumerate(recent_dates):
        obs_t_idx = lookback - len(recent_dates) + t_idx  # right-align in window
        try:
            slice_df = df.loc[date]
        except KeyError:
            continue
        for ticker, row in slice_df.iterrows():
            if ticker in asset_map and ticker not in insufficient:
                obs[obs_t_idx, asset_map[ticker], :] = row.values.astype(np.float32)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # ── Build padding mask ────────────────────────────────────────────────────
    # Shape: (1, n_assets, lookback) — True means "ignore this position"
    padding_mask = None
    if insufficient:
        mask_np = np.zeros((1, n_assets, lookback), dtype=bool)
        for ticker in insufficient:
            ai = asset_map.get(ticker)
            if ai is not None:
                mask_np[0, ai, :] = True
        padding_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)

    return obs_t, insufficient, padding_mask


# ── Public API ────────────────────────────────────────────────────────────────

def _load_ensemble(
    models_dir: str,
    device: torch.device,
) -> list[tuple]:
    """
    Load all best_fold*.pt checkpoints from models_dir.

    Returns a list of (model, ckpt) pairs. Corrupt checkpoints are skipped
    with a WARNING log. Raises ModelNotAvailableError if no valid checkpoints
    are found.
    """
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(models_dir, "best_fold*.pt")))
    ensemble = []
    for path in paths:
        try:
            model, ckpt = _load_model(path, device)
            ensemble.append((model, ckpt))
        except ModelNotAvailableError as exc:
            logger.warning("Skipping checkpoint %s: %s", path, exc)
    if not ensemble:
        raise ModelNotAvailableError(
            f"No valid checkpoints found in '{models_dir}'."
        )
    return ensemble


def get_rl_targets(
    df_recent: pd.DataFrame,
    asset_list: list[str],
    checkpoint_path: str,
    mode: str = "rank",
    device: Optional[torch.device] = None,
    lookback: int = 20,
    rl_action_projection: str | None = None,
    rl_action_temperature: float | None = None,
    rl_action_top_k: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Load the PortfolioTransformer checkpoint and run a forward pass.

    Parameters
    ----------
    df_recent : pd.DataFrame
        MultiIndex [date, ticker] DataFrame with FEATURE_COLS columns.
    asset_list : list[str]
        Ordered list of tickers to score.
    checkpoint_path : str
        Path to a .pt checkpoint file, a directory containing best_fold*.pt
        files, or "auto" to load all checkpoints from the "models/" directory.
        When a directory or "auto" is given, all valid checkpoints are loaded
        and their weights are averaged (ensemble inference).
    mode : str
        "rank"    → pd.Series[ticker → rl_score ∈ [0, 1]] as shortlist rank percentiles
        "weights" → pd.Series[ticker | "CASH" → weight], sum = 1.0
    device : torch.device | None
        Inference device. Defaults to CPU.
    lookback : int
        Number of historical dates to use for the observation window.

    Returns
    -------
    pd.Series
        RL scores (mode="rank") or portfolio weights (mode="weights").

    Raises
    ------
    ModelNotAvailableError
        If checkpoint_path does not exist or fails to load.
    """
    if device is None:
        device = torch.device("cpu")

    # ── Determine whether to use ensemble or single checkpoint ───────────────
    use_ensemble = (
        checkpoint_path == "auto"
        or os.path.isdir(checkpoint_path)
    )

    # ── Load model(s) first to determine expected n_features ─────────────────
    # The checkpoint was trained with a specific n_features. If FEATURE_COLS
    # has grown (e.g. new market context / fundamental / regime features were
    # added), we must slice the observation tensor to match the checkpoint.
    if use_ensemble:
        models_dir = "models" if checkpoint_path == "auto" else checkpoint_path
        ensemble = _load_ensemble(models_dir, device)
        # Use the first checkpoint's n_features as the canonical value
        ckpt_n_features = ensemble[0][1].get("model_cfg", {}).get("n_features", len(FEATURE_COLS))
        projection_settings = _resolve_projection_settings(
            ensemble[0][1],
            rl_action_projection=rl_action_projection,
            rl_action_temperature=rl_action_temperature,
            rl_action_top_k=rl_action_top_k,
        )
    else:
        model, ckpt = _load_model(checkpoint_path, device)
        ckpt_n_features = ckpt.get("model_cfg", {}).get("n_features", len(FEATURE_COLS))
        projection_settings = _resolve_projection_settings(
            ckpt,
            rl_action_projection=rl_action_projection,
            rl_action_temperature=rl_action_temperature,
            rl_action_top_k=rl_action_top_k,
        )
    projection_args = projection_kwargs(projection_settings)
    logger.info(
        "RL inference projection: mode=%s temperature=%.4f top_k=%d",
        projection_settings["rl_action_projection"],
        projection_settings["rl_action_temperature"],
        projection_settings["rl_action_top_k"],
    )

    # Select only the features the model was trained on.
    # New features added after training are silently dropped for inference.
    feature_cols_for_inference = FEATURE_COLS[:ckpt_n_features]
    if ckpt_n_features != len(FEATURE_COLS):
        logger.info(
            "Checkpoint trained with %d features; current FEATURE_COLS has %d. "
            "Using first %d features for inference. Retrain to use all features.",
            ckpt_n_features, len(FEATURE_COLS), ckpt_n_features,
        )

    # ── Build observation tensor ──────────────────────────────────────────────
    obs_t, insufficient, padding_mask = _build_obs_tensor(
        df_recent, asset_list, feature_cols_for_inference, lookback, device
    )

    if insufficient:
        logger.warning(
            "%d ticker(s) have fewer than %d dates of history — assigning rl_score=0.0: %s",
            len(insufficient),
            lookback,
            insufficient[:10],
        )

    if use_ensemble:
        # ── Ensemble inference ────────────────────────────────────────────────
        logger.info("Ensemble inference using %d checkpoint(s)", len(ensemble))

        all_weights = []
        for model, _ckpt in ensemble:
            w = model.get_weights(obs_t, padding_mask=padding_mask, **projection_args)
            all_weights.append(w.squeeze(0).cpu().numpy())

        weights_np = np.stack(all_weights, axis=0).mean(axis=0)  # (n_assets + 1,)
    else:
        # ── Single checkpoint inference ───────────────────────────────────────
        weights = model.get_weights(obs_t, padding_mask=padding_mask, **projection_args)
        weights_np = weights.squeeze(0).cpu().numpy()

    asset_weights = weights_np[:-1]   # (n_assets,)
    cash_weight = float(weights_np[-1])

    # Zero out tickers with insufficient history
    asset_map = {a: i for i, a in enumerate(asset_list)}
    for ticker in insufficient:
        if ticker in asset_map:
            asset_weights[asset_map[ticker]] = 0.0

    normalized_asset_weights = _normalize_asset_weights(asset_weights)
    rank_scores = _weights_to_rank_scores(asset_weights)
    insufficient_set = set(insufficient)

    if mode == "rank":
        # Use rank percentiles for entry/exit semantics so score meaning does
        # not shrink as the shortlist grows. Zero-weight assets stay at 0.0.
        return pd.Series(rank_scores, index=asset_list, name="rl_score", dtype=float)

    elif mode == "weights":
        # Full weight vector including CASH
        # Re-normalise to account for zeroed-out insufficient tickers
        all_weights = np.append(asset_weights, cash_weight)
        total = all_weights.sum()
        if total > 1e-9:
            all_weights = all_weights / total

        result = pd.Series(
            all_weights,
            index=list(asset_list) + ["CASH"],
            name="rl_weight",
            dtype=float,
        )

        weight_sum = result.sum()
        if abs(weight_sum - 1.0) > 1e-5:
            logger.warning(
                "Weight sum %.8f deviates from 1.0 by more than 1e-5", weight_sum
            )

        return result

    elif mode == "audit":
        return pd.DataFrame(
            {
                "rl_raw_weight": np.asarray(asset_weights, dtype=float),
                "rl_weight": np.asarray(normalized_asset_weights, dtype=float),
                "rl_rank_pct": np.asarray(rank_scores, dtype=float),
                "insufficient_history": [
                    ticker in insufficient_set for ticker in asset_list
                ],
            },
            index=pd.Index(asset_list, name="ticker"),
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'rank', 'weights', or 'audit'.")


# ── Phase 3 stub ──────────────────────────────────────────────────────────────

class WeightAdapter:
    """
    Converts RL continuous weight vector into discrete BUY/SELL/HOLD orders.

    Only activated when rl_phase=3. Not used in Phase 1 or Phase 2.
    This class is defined but NOT called from any live code path.
    """

    def __init__(
        self,
        min_weight_threshold: float = 0.01,
        cash_floor: float = 0.05,
    ):
        self.min_weight_threshold = min_weight_threshold
        self.cash_floor = cash_floor

    def adapt(
        self,
        rl_weights: pd.Series,
        portfolio,
        sector_map: dict,
        equity: float,
        sector_cap: float = 0.40,
        penny_threshold: float = 5.0,
        penny_budget_pct: float = 0.20,
        drawdown_circuit_breaker: float = 0.20,
    ) -> list:
        """
        Diff current portfolio weights vs target rl_weights.
        Generates SELL for weight=0, BUY for weight > min_weight_threshold.
        Applies sector cap, penny cap, and drawdown circuit breaker checks.
        Ensures total BUY value does not exceed equity * (1 - cash_floor).

        Parameters
        ----------
        rl_weights : pd.Series
            Target weights indexed by ticker (plus "CASH"), summing to 1.0.
        portfolio : Portfolio
            Current portfolio instance.
        sector_map : dict[str, str]
            Ticker → sector mapping.
        equity : float
            Current portfolio equity value.
        sector_cap : float
            Maximum fraction of equity allowed in any single sector.
        penny_threshold : float
            Price below which a stock is considered a penny stock.
        penny_budget_pct : float
            Maximum fraction of equity that may be allocated to penny stocks.
        drawdown_circuit_breaker : float
            If portfolio drawdown exceeds this fraction, skip all BUYs.

        Returns
        -------
        list[Decision]
            List of BUY/SELL decisions to move toward target weights.
        """
        from broker.brain import Decision  # local import — Phase 3 only

        decisions: list = []

        # ── Drawdown circuit breaker ──────────────────────────────────────────
        # Skip all BUYs if portfolio is in drawdown > drawdown_circuit_breaker
        initial_cash = getattr(portfolio, "initial_cash", equity)
        drawdown = 1.0 - (equity / initial_cash) if initial_cash > 0 else 0.0
        in_drawdown = drawdown > drawdown_circuit_breaker
        if in_drawdown:
            logger.warning(
                "WeightAdapter: drawdown circuit breaker triggered (%.1f%% > %.1f%%). "
                "All BUYs suppressed.",
                drawdown * 100,
                drawdown_circuit_breaker * 100,
            )

        # ── Compute current portfolio weights ─────────────────────────────────
        position_values = getattr(portfolio, "position_values", {})
        current_weights: dict[str, float] = {}
        if equity > 0:
            for ticker, val in position_values.items():
                current_weights[ticker] = val / equity

        # ── Compute current sector allocations ────────────────────────────────
        sector_alloc: dict[str, float] = {}
        for ticker, weight in current_weights.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_alloc[sector] = sector_alloc.get(sector, 0.0) + weight

        # ── Compute current penny stock allocation ────────────────────────────
        penny_alloc = 0.0
        for ticker, pos in getattr(portfolio, "positions", {}).items():
            price = pos.get("last_price", 0.0)
            if price < penny_threshold and price > 0:
                penny_alloc += position_values.get(ticker, 0.0) / equity if equity > 0 else 0.0

        # ── Budget for BUYs ───────────────────────────────────────────────────
        max_buy_value = equity * (1.0 - self.cash_floor)
        committed_buy_value = 0.0

        # ── Generate SELL decisions ───────────────────────────────────────────
        for ticker, current_weight in current_weights.items():
            if ticker == "CASH":
                continue
            target_weight = float(rl_weights.get(ticker, 0.0))
            if target_weight == 0.0 and current_weight > 0.0:
                pos = portfolio.positions.get(ticker, {})
                shares = pos.get("shares", 0.0)
                price = pos.get("last_price", 0.0)
                if shares > 0 and price > 0:
                    logger.info(
                        "WeightAdapter: SELL %s (target weight=0, current=%.3f)",
                        ticker, current_weight,
                    )
                    decisions.append(Decision(
                        action="SELL",
                        ticker=ticker,
                        shares=shares,
                        price=price,
                        score=0.0,
                        reason="rl_phase3: target_weight=0",
                    ))
                    # Update sector alloc after sell
                    sector = sector_map.get(ticker, "Unknown")
                    sector_alloc[sector] = max(0.0, sector_alloc.get(sector, 0.0) - current_weight)

        # ── Generate BUY decisions ────────────────────────────────────────────
        if not in_drawdown:
            # Sort by target weight descending for priority allocation
            buy_candidates = [
                (ticker, float(weight))
                for ticker, weight in rl_weights.items()
                if ticker != "CASH"
                and float(weight) > self.min_weight_threshold
                and ticker not in current_weights
            ]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, target_weight in buy_candidates:
                # Check BUY budget
                target_value = equity * target_weight
                if committed_buy_value + target_value > max_buy_value:
                    logger.debug(
                        "WeightAdapter: skipping BUY %s — would exceed cash floor budget",
                        ticker,
                    )
                    continue

                # Sector cap check
                sector = sector_map.get(ticker, "Unknown")
                current_sector_weight = sector_alloc.get(sector, 0.0)
                if current_sector_weight + target_weight > sector_cap:
                    logger.debug(
                        "WeightAdapter: skipping BUY %s — sector '%s' at cap (%.1f%%)",
                        ticker, sector, current_sector_weight * 100,
                    )
                    continue

                # Penny cap check — need price to evaluate
                # Use a placeholder price of 0.0 if not available; caller should
                # ensure portfolio prices are up to date before calling adapt()
                price = 0.0
                pos = getattr(portfolio, "positions", {}).get(ticker)
                if pos is not None:
                    price = pos.get("last_price", 0.0)

                if price > 0 and price < penny_threshold:
                    if penny_alloc + target_weight > penny_budget_pct:
                        logger.debug(
                            "WeightAdapter: skipping BUY %s (penny stock, budget exhausted)",
                            ticker,
                        )
                        continue
                    penny_alloc += target_weight

                if price <= 0:
                    logger.warning(
                        "WeightAdapter: no price available for %s — skipping BUY", ticker
                    )
                    continue

                shares = target_value / price
                if shares < 0.001:
                    continue

                logger.info(
                    "WeightAdapter: BUY %s %.2f shares @ %.4f (target_weight=%.3f)",
                    ticker, shares, price, target_weight,
                )
                decisions.append(Decision(
                    action="BUY",
                    ticker=ticker,
                    shares=shares,
                    price=price,
                    score=target_weight,
                    reason=f"rl_phase3: target_weight={target_weight:.4f}",
                ))
                committed_buy_value += target_value
                sector_alloc[sector] = sector_alloc.get(sector, 0.0) + target_weight

        return decisions
