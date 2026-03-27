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
import torch.nn.functional as F

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
) -> tuple[torch.Tensor, list[str]]:
    """
    Build observation tensor (1, lookback, n_assets, n_features).

    - Aligns df_recent to asset_list using feature_cols column order.
    - Applies cross-sectional z-score per date.
    - Clips to [-5.0, 5.0].
    - Pads with zeros for tickers with insufficient history.

    Returns (obs_tensor, tickers_with_insufficient_history).
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
    return obs_t, insufficient


# ── Public API ────────────────────────────────────────────────────────────────

def get_rl_targets(
    df_recent: pd.DataFrame,
    asset_list: list[str],
    checkpoint_path: str,
    mode: str = "rank",
    device: Optional[torch.device] = None,
    lookback: int = 20,
) -> pd.Series:
    """
    Load the PortfolioTransformer checkpoint and run a forward pass.

    Parameters
    ----------
    df_recent : pd.DataFrame
        MultiIndex [date, ticker] DataFrame with FEATURE_COLS columns.
    asset_list : list[str]
        Ordered list of tickers to score.
    checkpoint_path : str
        Path to a .pt checkpoint file.
    mode : str
        "rank"    → pd.Series[ticker → rl_score ∈ [0, 1]]
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

    # ── Load model ────────────────────────────────────────────────────────────
    model, ckpt = _load_model(checkpoint_path, device)

    # ── Build observation tensor ──────────────────────────────────────────────
    obs_t, insufficient = _build_obs_tensor(
        df_recent, asset_list, FEATURE_COLS, lookback, device
    )

    if insufficient:
        logger.warning(
            "%d ticker(s) have fewer than %d dates of history — assigning rl_score=0.0: %s",
            len(insufficient),
            lookback,
            insufficient[:10],
        )

    # ── Forward pass ─────────────────────────────────────────────────────────
    weights = model.get_weights(obs_t)  # (1, n_assets + 1)
    weights_np = weights.squeeze(0).cpu().numpy()  # (n_assets + 1,)

    asset_weights = weights_np[:-1]   # (n_assets,)
    cash_weight = float(weights_np[-1])

    # Zero out tickers with insufficient history
    asset_map = {a: i for i, a in enumerate(asset_list)}
    for ticker in insufficient:
        if ticker in asset_map:
            asset_weights[asset_map[ticker]] = 0.0

    if mode == "rank":
        # Renormalise asset weights to [0, 1] by dividing by their sum
        total = asset_weights.sum()
        if total > 1e-9:
            scores = asset_weights / total
        else:
            scores = asset_weights.copy()

        return pd.Series(scores, index=asset_list, name="rl_score", dtype=float)

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

    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'rank' or 'weights'.")


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
