"""
Broker decision engine.

Decision process each cycle:
  1. Validate + update prices (cross-checks suspicious moves)
  2. Volatility-adjusted stop-loss and partial take-profit
  3. Re-research held positions for signal deterioration
  4. Score sectors dynamically — broker decides its own allocations
  5. Screen full universe for candidates
  6. Skip earnings-window stocks (configurable)
  7. Deep-research top candidates
  8. Sector-aware position sizing with diversification penalty
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from broker.analyst   import research, fetch_ticker_data, research_from_features
from broker.portfolio import Portfolio, fetch_latest_market_price
from broker.sectors   import (
    get_sectors_bulk, score_sectors, compute_target_allocations,
    get_portfolio_sector_weights,
)
from broker.exposure import (
    effective_bet_count,
    exposure_weights,
    low_price_bucket,
    portfolio_low_price_values,
    portfolio_theme_values,
    theme_bucket,
)
from broker.validator import validate_portfolio_prices
from pipeline.rl_inference import get_rl_targets

logger = logging.getLogger(__name__)


def _build_recent_return_frame(
    df_features: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    if "ret_1d" not in df_features.columns:
        return pd.DataFrame()
    dates = sorted(df_features.index.get_level_values("date").unique())
    if not dates:
        return pd.DataFrame()
    recent_dates = dates[-lookback_days:]
    recent = df_features[df_features.index.get_level_values("date").isin(recent_dates)]
    try:
        return recent["ret_1d"].unstack("ticker")
    except Exception:
        return pd.DataFrame()


def _candidate_correlation_stats(
    returns_frame: pd.DataFrame,
    candidate: str,
    held_tickers: list[str],
    min_obs: int = 20,
) -> dict | None:
    if returns_frame.empty or candidate not in returns_frame.columns or not held_tickers:
        return None

    corrs = []
    for held in held_tickers:
        if held not in returns_frame.columns:
            continue
        pair = returns_frame[[candidate, held]].dropna()
        if len(pair) < min_obs:
            continue
        corr = float(pair[candidate].corr(pair[held]))
        if np.isfinite(corr):
            corrs.append(abs(corr))

    if not corrs:
        return None

    return {
        "max_abs_corr": float(np.max(corrs)),
        "mean_abs_corr": float(np.mean(corrs)),
    }


@dataclass
class Decision:
    action:  str          # "BUY", "SELL", "SELL_PARTIAL", "HOLD"
    ticker:  str
    shares:  float
    price:   float
    score:   float
    reason:  str


class BrokerBrain:
    def __init__(
        self,
        portfolio:            Portfolio,
        max_positions:        int   = 20,
        max_position_pct:     float = 0.10,   # max 10% of equity per position
        stop_loss_atr_mult:   float = 2.5,    # stop = entry - 2.5 × ATR
        stop_loss_pct_floor:  float = 0.07,   # minimum stop regardless of ATR
        stop_loss_pct_ceil:   float = 0.25,   # maximum stop regardless of ATR
        partial_profit_pct:   float = 0.20,   # sell half at +20%
        full_profit_pct:      float = 0.45,   # sell rest at +45%
        trailing_stop_pct:    float = 0.12,   # trail winners instead of capping them early
        trailing_activation_pct: float = 0.18,  # activate trailing once a cushion exists
        signal_exit_score:    float = 0.18,   # only very weak signals can trigger heuristic exits
        signal_exit_grace_cycles: int = 2,    # repeated weakness required before bailing
        signal_exit_min_hold_days: int = 5,   # avoid churning fresh entries
        signal_exit_winner_buffer_pct: float = 0.10,  # let winners run in healthy regimes
        min_score:            float = 0.60,
        penny_max_pct:        float = 0.20,
        penny_threshold:      float = 5.0,
        max_sector_pct:       float = 0.40,   # hard cap per sector
        max_pair_correlation: float = 0.80,
        correlation_lookback_days: int = 60,
        theme_max_pct:       float = 1.00,
        low_price_max_pct:   float = 1.00,
        sentiment_policy:    str   = "informational",
        sentiment_negative_weight_mult: float = 0.80,
        sentiment_veto_composite_floor: float = 0.50,
        weak_theme_min_positions: int = 2,
        weak_theme_return_threshold: float = -0.03,
        weak_theme_penalty_mult: float = 0.50,
        weak_theme_cooldown_cycles: int = 0,
        weak_theme_cooldown_min_hits: int = 2,
        low_price_rank_policy: str = "late_cap",
        low_price_rank_penalty_mult: float = 0.70,
        low_price_high_rank_floor: float = 0.80,
        earnings_reaction_enabled: bool = False,
        earnings_reaction_rank_strength: float = 0.10,
        earnings_reaction_weight_strength: float = 0.10,
        macro_regime_enabled: bool = False,
        macro_regime_weight_strength: float = 0.08,
        macro_regime_mode: str = "standard",
        insider_adjustment_enabled: bool = False,
        insider_adjustment_rank_strength: float = 0.08,
        insider_adjustment_weight_strength: float = 0.08,
        allow_unpromoted_feature_influence: bool = False,
        llm_sidecar_features: dict | None = None,
        llm_sidecar_broker_influence: bool = False,
        llm_sidecar_min_confidence: float = 0.65,
        event_sidecar_features: dict | None = None,
        event_sidecar_broker_influence: bool = False,
        pattern_features: dict | None = None,
        pattern_sidecar_broker_influence: bool = False,
        avoid_earnings_days:  int   = 3,       # skip stocks within N days of earnings
        device=None,
        rl_enabled:           bool  = False,
        rl_checkpoint_path:   str | None = "models/best_fold9.pt",
        rl_phase:             int   = 1,
        rl_exit_threshold:    float = 0.30,
        rl_conviction_drop:   float = 0.20,
        rl_min_score:         float = 0.0,   # separate threshold for RL rank percentiles (0 = top-k only)
        dead_money_days:      int   = 0,     # exit positions with no progress after N days (0 = disabled)
        dead_money_min_return: float = 0.02, # "no progress" = return below this threshold
    ):
        self.portfolio            = portfolio
        self.max_positions        = max_positions
        self.max_position_pct     = max_position_pct
        self.stop_loss_atr_mult   = stop_loss_atr_mult
        self.stop_loss_pct_floor  = stop_loss_pct_floor
        self.stop_loss_pct_ceil   = stop_loss_pct_ceil
        self.partial_profit_pct   = partial_profit_pct
        self.full_profit_pct      = full_profit_pct
        self.trailing_stop_pct    = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.signal_exit_score    = signal_exit_score
        self.signal_exit_grace_cycles = signal_exit_grace_cycles
        self.signal_exit_min_hold_days = signal_exit_min_hold_days
        self.signal_exit_winner_buffer_pct = signal_exit_winner_buffer_pct
        self.min_score            = min_score
        self.penny_max_pct        = penny_max_pct
        self.penny_threshold      = penny_threshold
        self.max_sector_pct       = max_sector_pct
        self.max_pair_correlation = max_pair_correlation
        self.correlation_lookback_days = correlation_lookback_days
        self.theme_max_pct       = theme_max_pct
        self.low_price_max_pct   = low_price_max_pct
        self.sentiment_policy    = str(sentiment_policy or "informational").strip().lower()
        self.sentiment_negative_weight_mult = sentiment_negative_weight_mult
        self.sentiment_veto_composite_floor = sentiment_veto_composite_floor
        self.weak_theme_min_positions = weak_theme_min_positions
        self.weak_theme_return_threshold = weak_theme_return_threshold
        self.weak_theme_penalty_mult = weak_theme_penalty_mult
        self.weak_theme_cooldown_cycles = weak_theme_cooldown_cycles
        self.weak_theme_cooldown_min_hits = weak_theme_cooldown_min_hits
        self.low_price_rank_policy = str(low_price_rank_policy or "late_cap").strip().lower()
        self.low_price_rank_penalty_mult = low_price_rank_penalty_mult
        self.low_price_high_rank_floor = low_price_high_rank_floor
        self.earnings_reaction_enabled = bool(earnings_reaction_enabled)
        self.earnings_reaction_rank_strength = earnings_reaction_rank_strength
        self.earnings_reaction_weight_strength = earnings_reaction_weight_strength
        self.macro_regime_enabled = bool(macro_regime_enabled)
        self.macro_regime_weight_strength = macro_regime_weight_strength
        self.macro_regime_mode = str(macro_regime_mode or "standard").strip().lower()
        self.insider_adjustment_enabled = bool(insider_adjustment_enabled)
        self.insider_adjustment_rank_strength = insider_adjustment_rank_strength
        self.insider_adjustment_weight_strength = insider_adjustment_weight_strength
        self.allow_unpromoted_feature_influence = bool(allow_unpromoted_feature_influence)
        self.llm_sidecar_features = {
            str(k).upper(): v for k, v in (llm_sidecar_features or {}).items()
        }
        self.llm_sidecar_broker_influence = bool(llm_sidecar_broker_influence)
        self.llm_sidecar_min_confidence = float(llm_sidecar_min_confidence)
        self.event_sidecar_features = {
            str(k).upper(): v for k, v in (event_sidecar_features or {}).items()
        }
        self.event_sidecar_broker_influence = bool(event_sidecar_broker_influence)
        self.pattern_features = {
            str(k).upper(): v for k, v in (pattern_features or {}).items()
        }
        self.pattern_sidecar_broker_influence = bool(pattern_sidecar_broker_influence)
        self.avoid_earnings_days  = avoid_earnings_days
        self.device               = device
        self.rl_enabled           = rl_enabled
        self.rl_checkpoint_path   = rl_checkpoint_path
        self.rl_phase             = rl_phase
        self.rl_exit_threshold    = rl_exit_threshold
        self.rl_conviction_drop   = rl_conviction_drop
        self.rl_min_score         = rl_min_score
        self.dead_money_days      = dead_money_days
        self.dead_money_min_return = dead_money_min_return

        # Cache sector map across cycles (refreshed weekly)
        self._sector_map:   dict[str, str]   = {}
        self._sector_cache_date: datetime | None = None
        # Ensure _base_min_score is always set (broker.py also sets this after init)
        self._base_min_score: float = min_score
        self._last_rl_scores: pd.Series = pd.Series(dtype=float)
        self._last_rl_audit: pd.DataFrame = pd.DataFrame()
        self._last_cycle_audit: pd.DataFrame = pd.DataFrame()
        self._last_allocation_summary: dict = {}
        self._weak_theme_states: dict[str, dict] = {}
        self._weak_theme_cycle_id: int = 0

    def _current_market_regime(self, df_features: pd.DataFrame) -> int | None:
        regime_cols = [f"regime_{i}" for i in range(4) if f"regime_{i}" in df_features.columns]
        if not regime_cols or df_features.empty:
            return None

        try:
            latest_date = df_features.index.get_level_values("date").max()
            snap = df_features.xs(latest_date, level="date")
            regime_means = snap[regime_cols].mean(axis=0)
        except Exception:
            return None

        if regime_means.empty:
            return None

        best_col = str(regime_means.astype(float).idxmax())
        try:
            return int(best_col.rsplit("_", 1)[-1])
        except Exception:
            return None

    def _effective_trailing_stop_pct(self, market_regime: int | None) -> float:
        trail = float(self.trailing_stop_pct)
        if market_regime == 2:
            trail *= 0.85
        elif market_regime == 3:
            trail *= 0.70
        return float(np.clip(trail, 0.05, 0.25))

    def _signal_exit_profile(self, market_regime: int | None) -> tuple[float, int]:
        threshold = float(self.signal_exit_score)
        streak = int(max(self.signal_exit_grace_cycles, 1))

        if market_regime == 0:
            threshold = max(0.0, threshold - 0.08)
            streak += 2
        elif market_regime == 1:
            threshold = max(0.0, threshold - 0.05)
            streak += 1
        elif market_regime == 3:
            threshold = min(1.0, threshold + 0.08)
            streak = max(1, streak - 1)

        return float(threshold), int(streak)

    def _effective_rl_entry_floor(self, market_regime: int | None) -> float:
        _ = market_regime
        return float(np.clip(float(self.rl_min_score), 0.0, 0.95))

    @staticmethod
    def _conviction_from_score(
        score: float,
        score_floor: float,
        market_regime: int | None,
    ) -> float:
        conviction = np.clip(
            (float(score) - float(score_floor)) / max(1.0 - float(score_floor), 1e-6),
            0.0,
            1.0,
        )
        if market_regime == 0:
            conviction = conviction ** 0.80
        elif market_regime == 1:
            conviction = conviction ** 0.90
        elif market_regime == 3:
            conviction = conviction ** 1.20
        return float(np.clip(conviction, 0.0, 1.0))

    def _effective_max_position_pct(
        self,
        market_regime: int | None,
        conviction: float,
    ) -> float:
        cap = float(self.max_position_pct)
        if market_regime == 0:
            cap *= 1.15
        elif market_regime == 1:
            cap *= 1.08
        elif market_regime == 3:
            cap *= 0.85
        cap *= 1.0 + 0.12 * max(float(conviction) - 0.5, 0.0)
        return float(np.clip(cap, 0.05, 0.25))

    def _effective_sector_cap(
        self,
        market_regime: int | None,
        conviction: float,
    ) -> float:
        cap = float(self.max_sector_pct)
        if market_regime == 0:
            cap *= 1.25
        elif market_regime == 1:
            cap *= 1.12
        elif market_regime == 3:
            cap *= 0.80
        cap *= 1.0 + 0.10 * max(float(conviction) - 0.4, 0.0)
        return float(np.clip(cap, 0.15, 0.55))

    def _sector_overflow_scale(
        self,
        market_regime: int | None,
        conviction: float,
    ) -> float:
        if market_regime == 0:
            base = 0.75
        elif market_regime == 1:
            base = 0.50
        elif market_regime == 2:
            base = 0.20
        else:
            base = 0.0
        return float(np.clip(base * float(conviction), 0.0, 1.0))

    def _effective_max_pair_correlation(
        self,
        market_regime: int | None,
        conviction: float,
    ) -> float:
        limit = float(self.max_pair_correlation)
        if market_regime == 0:
            limit += 0.08 * float(conviction)
        elif market_regime == 1:
            limit += 0.05 * float(conviction)
        elif market_regime == 3:
            limit -= 0.10
        return float(np.clip(limit, 0.55, 1.0))

    def _should_exit_on_signal(
        self,
        pos: dict,
        composite_score: float,
        pnl_pct: float,
        market_regime: int | None,
    ) -> bool:
        threshold, required_streak = self._signal_exit_profile(market_regime)
        weak_signal_streak = int(pos.get("weak_signal_streak", 0))
        days_held = int(pos.get("days_held", 0))

        if composite_score >= threshold:
            return False
        if days_held < self.signal_exit_min_hold_days:
            return False
        if weak_signal_streak < required_streak:
            return False

        winner_buffer = float(self.signal_exit_winner_buffer_pct)
        if pnl_pct >= winner_buffer and market_regime in (None, 0, 1):
            return False
        if pnl_pct >= (winner_buffer * 1.5) and market_regime == 2:
            return False

        return True

    # ── Main decision cycle ───────────────────────────────────────────────────

    def run_cycle(
        self,
        df_features: pd.DataFrame,
        screener_top_n: int = 50,
        risk_engine=None,   # PortfolioRiskEngine instance
    ) -> list[Decision]:
        decisions = []
        self._weak_theme_cycle_id += 1
        market_regime = self._current_market_regime(df_features)
        if risk_engine is not None and hasattr(risk_engine, "set_market_regime"):
            risk_engine.set_market_regime(market_regime)

        # ── RL pre-flight check ───────────────────────────────────────────────
        if self.rl_enabled:
            try:
                self._assert_model_available()
            except RuntimeError as exc:
                logger.error(f"RL model unavailable — aborting cycle: {exc}")
                return []

            # Warn if options are not already suppressed
            if not getattr(self, "no_options", True):
                logger.warning(
                    "RL mode is active: options decisions are suppressed "
                    "regardless of the no_options setting."
                )

        # ── 1. Refresh sector map (weekly) ────────────────────────────────────
        self._maybe_refresh_sector_map(df_features)

        # ── 2. Validate + update prices ───────────────────────────────────────
        held_tickers = list(self.portfolio.positions.keys())
        if held_tickers:
            raw_prices = self._get_current_prices(held_tickers)
            clean_prices = validate_portfolio_prices(
                self.portfolio.positions, raw_prices
            )
            self.portfolio.update_prices(clean_prices)

        # ── 3. RL exit checks (Phase 2) — deferred until after RL scoring ───────
        # rl_scores must be computed first so exits use the cross-sectional pass.
        # The actual call is made after step 6 below.
        rl_exited_tickers: set[str] = set()
        local_research_fallbacks = 0

        # ── 4. Heuristic exit decisions ───────────────────────────────────────
        # (RL-exited tickers are excluded once rl_exited_tickers is populated)
        for ticker in list(self.portfolio.positions.keys()):
            # Skip tickers that already have an RL-driven exit decision
            if ticker in rl_exited_tickers:
                continue

            pos   = self.portfolio.positions[ticker]
            price = pos["last_price"]
            cost  = pos["avg_cost"]
            if cost <= 0:
                continue

            days_held = int(pos.get("days_held", 0))
            pos["days_held"] = days_held + 1
            pos["peak_price"] = max(
                float(pos.get("peak_price", cost)),
                float(cost),
                float(price),
            )

            pnl_pct = (price - cost) / cost

            # Compute volatility-adjusted stop for this position
            stop_pct = self._get_stop_loss_pct(ticker, pos)

            # Full stop-loss
            if pnl_pct <= -stop_pct:
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=0.0,
                    reason=f"Stop-loss ({pnl_pct:.1%} vs -{stop_pct:.1%} ATR-adjusted)",
                ))
                continue

            peak_price = float(pos.get("peak_price", price))
            peak_pnl_pct = (peak_price - cost) / cost if cost > 0 else 0.0
            drawdown_from_peak = (price - peak_price) / peak_price if peak_price > 0 else 0.0
            trail_pct = self._effective_trailing_stop_pct(market_regime)
            if (
                self.trailing_stop_pct > 0
                and peak_pnl_pct >= self.trailing_activation_pct
                and drawdown_from_peak <= -trail_pct
            ):
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=0.9,
                    reason=(
                        f"Trailing stop ({drawdown_from_peak:.1%} from peak after "
                        f"{peak_pnl_pct:.1%} max gain)"
                    ),
                ))
                continue

            # Partial take-profit at +20% — sell half, let rest run
            if pnl_pct >= self.partial_profit_pct and not pos.get("partial_taken"):
                half_shares = pos["shares"] * 0.5
                decisions.append(Decision(
                    action="SELL_PARTIAL", ticker=ticker,
                    shares=half_shares, price=price, score=0.8,
                    reason=f"Partial take-profit ({pnl_pct:.1%}), selling 50%",
                ))
                continue

            # Full take-profit at +45%
            if pnl_pct >= self.full_profit_pct:
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=1.0,
                    reason=f"Full take-profit ({pnl_pct:.1%})",
                ))
                continue

            # Signal deterioration check
            report, used_local_research = self._research_with_fallback(ticker, df_features)
            if used_local_research:
                local_research_fallbacks += 1
            if report:
                composite_score = float(report.get("composite_score", 0.0))
                threshold, required_streak = self._signal_exit_profile(market_regime)
                if composite_score < threshold:
                    pos["weak_signal_streak"] = int(pos.get("weak_signal_streak", 0)) + 1
                else:
                    pos["weak_signal_streak"] = 0

                if self._should_exit_on_signal(pos, composite_score, pnl_pct, market_regime):
                    decisions.append(Decision(
                        action="SELL", ticker=ticker,
                        shares=pos["shares"], price=price,
                        score=composite_score,
                        reason=(
                            f"Signal deteriorated (score={composite_score:.2f}, "
                            f"streak={pos['weak_signal_streak']}/{required_streak})"
                        ),
                    ))
                    continue
            else:
                pos["weak_signal_streak"] = 0

            # Dead-money exit — rotate out of positions making no progress
            if (
                self.dead_money_days > 0
                and days_held >= self.dead_money_days
                and pnl_pct < self.dead_money_min_return
            ):
                logger.info(
                    "Dead-money exit %s: held %d days, return %.1f%% < %.1f%% threshold",
                    ticker, days_held, pnl_pct * 100, self.dead_money_min_return * 100,
                )
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=0.1,
                    reason=(
                        f"Dead-money exit: held {days_held}d, "
                        f"return={pnl_pct:.1%} < {self.dead_money_min_return:.1%} threshold"
                    ),
                ))
                continue

            # Delisted / halted ticker check — only for positions held 5+ days
            # with no price movement. Avoids false positives on newly opened positions
            # where last_price == avg_cost by definition.
            if price > 0 and price == pos.get("avg_cost", 0) and days_held >= 5:
                # If last_price equals avg_cost and we've never had a price update,
                # the ticker may be halted. Check if yfinance returns data.
                try:
                    from broker.analyst import fetch_ticker_data
                    recent = fetch_ticker_data(ticker, days=10)
                    if recent is None or recent.empty:
                        logger.warning(
                            "Ticker %s appears delisted or halted — force-closing at last price",
                            ticker,
                        )
                        decisions.append(Decision(
                            action="SELL", ticker=ticker,
                            shares=pos["shares"], price=price, score=0.0,
                            reason="Delisted/halted — no price data available",
                        ))
                except Exception:
                    pass

        # ── 5. Sector scoring — broker decides allocations ────────────────────
        sector_scores = score_sectors(df_features, self._sector_map)
        current_sector_weights = get_portfolio_sector_weights(
            self.portfolio.positions, self._sector_map
        )
        target_sector_allocs = compute_target_allocations(
            sector_scores,
            current_sector_weights,
            max_single_sector=self.max_sector_pct,
        )

        # ── 6. Screen for candidates ──────────────────────────────────────────
        candidates = self._screen_candidates(df_features, top_n=screener_top_n)

        # ── Phase 1: RL scoring of shortlist ──────────────────────────────────
        rl_scores: pd.Series | None = None
        rl_score_table: pd.DataFrame | None = None
        self._last_rl_scores = pd.Series(dtype=float)
        self._last_rl_audit = pd.DataFrame()
        self._last_cycle_audit = pd.DataFrame()
        self._last_allocation_summary = {}
        if self.rl_enabled and candidates:
            try:
                rl_score_table = get_rl_targets(
                    df_features,
                    candidates,
                    self.rl_checkpoint_path,
                    mode="audit",
                )
                if isinstance(rl_score_table, pd.Series):
                    rl_scores = rl_score_table.astype(float)
                    rl_score_table = pd.DataFrame(
                        {
                            "rl_rank_pct": rl_scores,
                            "rl_weight": np.nan,
                            "rl_raw_weight": np.nan,
                        }
                    )
                else:
                    rl_scores = rl_score_table["rl_rank_pct"].astype(float)
                logger.info(
                    "RL scored %d tickers in shortlist.", len(candidates)
                )
                # Cache for broker.py to use when storing entry rank percentile
                self._last_rl_scores = rl_scores
                self._last_rl_audit = rl_score_table.copy()
            except Exception as exc:
                logger.error(
                    "RL inference failed — aborting cycle: %s", exc, exc_info=True
                )
                return decisions

        # ── Phase 2: RL exit checks — now that rl_scores is available ─────────
        if self.rl_enabled and self.rl_phase >= 2 and rl_scores is not None:
            rl_exit_decisions = self._rl_exit_checks(
                list(self.portfolio.positions.keys()), rl_scores
            )
            decisions.extend(rl_exit_decisions)
            rl_exited_tickers = {d.ticker for d in rl_exit_decisions}

        # ── Deduplicate exits: one action per ticker, highest priority wins ───
        # Priority: SELL (stop/tp/signal) > SELL_PARTIAL
        # If both heuristic and RL generated exits for the same ticker, keep
        # the SELL (full exit) over SELL_PARTIAL, and the first SELL wins.
        exit_by_ticker: dict[str, Decision] = {}
        buy_decisions: list[Decision] = []
        for d in decisions:
            if d.action in ("SELL", "SELL_PARTIAL"):
                existing = exit_by_ticker.get(d.ticker)
                if existing is None:
                    exit_by_ticker[d.ticker] = d
                elif existing.action == "SELL_PARTIAL" and d.action == "SELL":
                    # Full exit beats partial
                    exit_by_ticker[d.ticker] = d
            else:
                buy_decisions.append(d)
        decisions = list(exit_by_ticker.values()) + buy_decisions

        # ── 7. Buy decisions ──────────────────────────────────────────────────
        sells_pending = sum(1 for d in decisions if d.action in ("SELL",))
        n_slots = self.max_positions - (
            len(self.portfolio.positions) - sells_pending
        )
        n_slots = max(0, n_slots)

        researched: list[dict] = []
        shortlist_considered = 0
        already_held_skips = 0
        earnings_skips = 0
        research_none_skips = 0
        threshold_skips = 0
        penny_budget_skips = 0
        sector_budget_skips = 0
        correlation_blocked_skips = 0
        risk_blocked_skips = 0
        sentiment_blocked_skips = 0
        tiny_alloc_skips = 0
        buyable_count = 0
        cycle_audit_rows: list[dict] = []
        if n_slots > 0 and candidates:
            held_tickers_now = list(self.portfolio.positions.keys())
            recent_return_frame = (
                _build_recent_return_frame(
                    df_features,
                    lookback_days=self.correlation_lookback_days,
                )
                if held_tickers_now else pd.DataFrame()
            )
            for ticker in candidates[:min(n_slots * 3, 40)]:
                shortlist_considered += 1
                if ticker in self.portfolio.positions:
                    already_held_skips += 1
                    continue

                # Skip if earnings are imminent
                near_earnings = self._near_earnings(ticker)
                if near_earnings:
                    earnings_skips += 1
                    logger.debug(f"Skipping {ticker} — near earnings window")
                    continue

                report, used_local_research = self._research_with_fallback(ticker, df_features)
                if used_local_research:
                    local_research_fallbacks += 1
                if report is None:
                    research_none_skips += 1
                    continue

                composite = report["composite_score"]
                sentiment_label = self._sentiment_label(report)
                rl_score_val = float(rl_scores.get(ticker, 0.0)) if rl_scores is not None else None
                rl_weight_val = None
                rl_raw_weight_val = None
                if rl_score_table is not None and ticker in rl_score_table.index:
                    rl_weight_val = float(rl_score_table.at[ticker, "rl_weight"])
                    rl_raw_weight_val = float(rl_score_table.at[ticker, "rl_raw_weight"])

                # Diagnostic logging — always log both scores when RL is active
                if self.rl_enabled and rl_score_val is not None:
                    delta = rl_score_val - composite
                    logger.debug(
                        "  %s  rl_score=%.4f  composite=%.4f  delta=%+.4f",
                        ticker, rl_score_val, composite, delta,
                    )

                # Apply RL percentile threshold. A zero RL score means the model
                # gave no usable conviction for this candidate, so skip it even
                # when rl_min_score is zero.
                if self.rl_enabled:
                    rl_entry_floor = self._effective_rl_entry_floor(market_regime)
                    if (
                        rl_score_val is None
                        or rl_score_val <= 0.0
                        or rl_score_val < rl_entry_floor
                    ):
                        threshold_skips += 1
                        continue
                else:
                    if composite < self.min_score:
                        threshold_skips += 1
                        continue

                raw_signal_for_rank = (
                    float(rl_score_val)
                    if self.rl_enabled and rl_score_val is not None
                    else float(composite)
                )
                candidate_price = float(report.get("price", np.nan))
                is_low_price_candidate = (
                    np.isfinite(candidate_price)
                    and candidate_price < 10.0
                )
                low_price_rank_scale = 1.0
                if is_low_price_candidate:
                    if self.low_price_rank_policy == "pre_penalty":
                        low_price_rank_scale = float(np.clip(
                            self.low_price_rank_penalty_mult,
                            0.05,
                            1.0,
                        ))
                    elif (
                        self.low_price_rank_policy in {"exclude_high_rank", "exclude_top"}
                        and raw_signal_for_rank >= float(self.low_price_high_rank_floor)
                    ):
                        cycle_audit_rows.append(
                            {
                                "ticker": ticker,
                                "candidate_status": "low_price_high_rank_excluded",
                                "logged_score": raw_signal_for_rank,
                                "composite_score": float(composite),
                                "rl_rank_pct": float(rl_score_val) if rl_score_val is not None else np.nan,
                                "rl_weight": float(rl_weight_val) if rl_weight_val is not None else np.nan,
                                "rl_raw_weight": (
                                    float(rl_raw_weight_val) if rl_raw_weight_val is not None else np.nan
                                ),
                                "sector": self._sector_map.get(ticker.upper(), "Unknown"),
                                "theme_bucket": theme_bucket(
                                    ticker,
                                    self._sector_map.get(ticker.upper(), "Unknown"),
                                ),
                                "low_price_bucket": low_price_bucket(
                                    candidate_price,
                                    self.penny_threshold,
                                ),
                                "low_price_rank_policy": self.low_price_rank_policy,
                                "low_price_rank_scale": 0.0,
                                "low_price_high_rank_floor": float(self.low_price_high_rank_floor),
                            }
                        )
                        threshold_skips += 1
                        continue

                if self._sentiment_vetoes_entry(sentiment_label, composite):
                    sentiment_blocked_skips += 1
                    cycle_audit_rows.append(
                        {
                            "ticker": ticker,
                            "candidate_status": "sentiment_blocked",
                            "logged_score": (
                                float(rl_score_val)
                                if rl_score_val is not None else float(composite)
                            ),
                            "composite_score": float(composite),
                            "rl_rank_pct": float(rl_score_val) if rl_score_val is not None else np.nan,
                            "rl_weight": float(rl_weight_val) if rl_weight_val is not None else np.nan,
                            "rl_raw_weight": (
                                float(rl_raw_weight_val) if rl_raw_weight_val is not None else np.nan
                            ),
                            "sector": self._sector_map.get(ticker.upper(), "Unknown"),
                            "theme_bucket": theme_bucket(
                                ticker,
                                self._sector_map.get(ticker.upper(), "Unknown"),
                            ),
                            "low_price_bucket": low_price_bucket(
                                float(report.get("price", np.nan)),
                                self.penny_threshold,
                            ),
                            "sentiment_label": sentiment_label,
                            "sentiment_policy": self.sentiment_policy,
                            "sentiment_veto_floor": float(self.sentiment_veto_composite_floor),
                        }
                    )
                    continue

                soft_rank_scale, soft_weight_scale, soft_signal_notes = self._soft_signal_adjustments(
                    report,
                    market_regime,
                )
                report["sector"] = self._sector_map.get(ticker.upper(), "Unknown")
                report["theme_bucket"] = theme_bucket(ticker, report["sector"])
                report["sentiment_label"] = sentiment_label
                report["rank_score"] = raw_signal_for_rank * low_price_rank_scale * soft_rank_scale
                report["low_price_rank_policy"] = self.low_price_rank_policy
                report["low_price_rank_scale"] = low_price_rank_scale
                report["soft_signal_rank_scale"] = soft_rank_scale
                report["soft_signal_weight_scale"] = soft_weight_scale
                report["soft_signal_notes"] = soft_signal_notes
                if rl_score_val is not None:
                    report["rl_score"] = rl_score_val
                    report["rl_rank_pct"] = rl_score_val
                if rl_weight_val is not None:
                    report["rl_weight"] = rl_weight_val
                if rl_raw_weight_val is not None:
                    report["rl_raw_weight"] = rl_raw_weight_val
                researched.append(report)
                cycle_audit_rows.append(
                    {
                        "ticker": ticker,
                        "candidate_status": "researched",
                        "logged_score": (
                            float(rl_score_val)
                            if rl_score_val is not None else float(composite)
                        ),
                        "composite_score": float(composite),
                        "rl_rank_pct": float(rl_score_val) if rl_score_val is not None else np.nan,
                        "rl_weight": float(rl_weight_val) if rl_weight_val is not None else np.nan,
                        "rl_raw_weight": (
                            float(rl_raw_weight_val) if rl_raw_weight_val is not None else np.nan
                        ),
                        "sector": report["sector"],
                        "theme_bucket": report["theme_bucket"],
                        "low_price_bucket": low_price_bucket(
                            float(report.get("price", np.nan)),
                            self.penny_threshold,
                        ),
                        "sentiment_label": sentiment_label,
                        "sentiment_policy": self.sentiment_policy,
                        "low_price_rank_policy": self.low_price_rank_policy,
                        "low_price_rank_scale": float(low_price_rank_scale),
                        "soft_signal_rank_scale": float(soft_rank_scale),
                        "soft_signal_weight_scale": float(soft_weight_scale),
                        "soft_signal_summary": self._format_soft_signal_notes(soft_signal_notes),
                        "llm_event_confidence": float(report.get("llm_event_confidence", 0.0) or 0.0),
                        "llm_event_trusted": bool(report.get("llm_event_trusted", False)),
                        "llm_thesis_impact": report.get("llm_thesis_impact", "unknown"),
                        "event_score": float(report.get("event_score", 0.0) or 0.0),
                        "event_risk_score": float(report.get("event_risk_score", 0.0) or 0.0),
                        "event_opportunity_score": float(
                            report.get("event_opportunity_score", 0.0) or 0.0
                        ),
                        "event_mention_count": int(report.get("event_mention_count", 0) or 0),
                        "event_top_types": ",".join(report.get("event_top_types", []) or []),
                        "crowd_mention_velocity": float(
                            report.get("crowd_mention_velocity", 0.0) or 0.0
                        ),
                        "pattern_score": float(report.get("pattern_score", 0.0) or 0.0),
                        "pattern_confidence": float(report.get("pattern_confidence", 0.0) or 0.0),
                        "primary_pattern": report.get("primary_pattern", "none"),
                        "rank_score": float(report["rank_score"]),
                    }
                )

            # Sort by rl_score (RL mode) or composite_score (heuristic mode)
            if self.rl_enabled:
                researched.sort(
                    key=lambda r: (-r.get("rank_score", r.get("rl_score", 0.0)), -r["composite_score"], r["ticker"]),
                )
            else:
                researched.sort(key=lambda r: (-r.get("rank_score", r["composite_score"]), r["ticker"]))

            # ── Per-cycle RL summary log ──────────────────────────────────────
            if self.rl_enabled and rl_scores is not None:
                n_scored = len(candidates)

                # Compute heuristic rank order vs RL rank order
                heuristic_order = sorted(
                    researched,
                    key=lambda r: (-r["composite_score"], r["ticker"]),
                )
                heuristic_rank = {r["ticker"]: i for i, r in enumerate(heuristic_order)}
                rl_rank = {r["ticker"]: i for i, r in enumerate(researched)}
                n_rank_diff = sum(
                    1 for t in rl_rank if heuristic_rank.get(t, -1) != rl_rank[t]
                )

                top5 = [
                    f"{r['ticker']}({r.get('rl_score', 0.0):.3f})"
                    for r in researched[:5]
                ]
                logger.info(
                    "RL cycle summary: scored=%d  rank_diffs=%d  top5=%s",
                    n_scored, n_rank_diff, " ".join(top5),
                )
                if not researched:
                    logger.info(
                        "RL cycle produced no buyable candidates after research/filtering "
                        "(rl_min_score=%.3f).",
                        self.rl_min_score,
                    )
                logger.info(
                    "RL filter counts: shortlisted=%d considered=%d researched=%d "
                    "held=%d earnings=%d no_research=%d threshold=%d sentiment_blocked=%d",
                    n_scored,
                    shortlist_considered,
                    len(researched),
                    already_held_skips,
                    earnings_skips,
                    research_none_skips,
                    threshold_skips,
                    sentiment_blocked_skips,
                )
            else:
                logger.debug(
                    "Heuristic filter counts: considered=%d researched=%d "
                    "held=%d earnings=%d no_research=%d threshold=%d sentiment_blocked=%d",
                    shortlist_considered,
                    len(researched),
                    already_held_skips,
                    earnings_skips,
                    research_none_skips,
                    threshold_skips,
                    sentiment_blocked_skips,
                )

            if local_research_fallbacks:
                logger.info(
                    "Research fallback used local feature snapshots for %d ticker(s).",
                    local_research_fallbacks,
                )

            equity      = self.portfolio.equity
            penny_value = sum(
                v for t, v in self.portfolio.position_values.items()
                if self.portfolio.positions[t]["last_price"] < self.penny_threshold
            )

            # Track sector spend this cycle to respect targets
            sector_spent: dict[str, float] = {}
            theme_spent: dict[str, float] = {}
            low_price_spent: dict[str, float] = {}
            theme_values = portfolio_theme_values(self.portfolio.positions, self._sector_map)
            low_price_values = portfolio_low_price_values(
                self.portfolio.positions,
                self.penny_threshold,
            )
            theme_weights = exposure_weights(theme_values)
            theme_effective_bets = effective_bet_count(theme_weights)

            for report in researched[:n_slots]:
                ticker        = report["ticker"]
                price         = report["price"]
                composite     = report["composite_score"]
                rl_score_val  = report.get("rl_score")
                rl_rank_pct_val = report.get("rl_rank_pct", rl_score_val)
                rl_weight_val = report.get("rl_weight")
                rl_raw_weight_val = report.get("rl_raw_weight")
                sector        = report.get("sector", "Unknown")
                theme         = report.get("theme_bucket") or theme_bucket(ticker, sector)
                is_penny      = price < self.penny_threshold
                price_bucket  = low_price_bucket(price, self.penny_threshold)

                # Conviction score: use rl_score when RL enabled, else composite
                score = rl_score_val if (self.rl_enabled and rl_score_val is not None) else composite
                score_floor = (
                    self._effective_rl_entry_floor(market_regime)
                    if (self.rl_enabled and rl_score_val is not None)
                    else self.min_score
                )
                conviction = self._conviction_from_score(score, score_floor, market_regime)

                # ── Penny cap ─────────────────────────────────────────────────
                if is_penny:
                    penny_budget = equity * self.penny_max_pct - penny_value
                    if penny_budget <= 0:
                        logger.debug(f"Penny cap reached, skipping {ticker}")
                        penny_budget_skips += 1
                        continue

                # ── Sector budget check ───────────────────────────────────────
                target_alloc = target_sector_allocs.get(sector, 0.05)
                current_sector_val = sum(
                    v for t, v in self.portfolio.position_values.items()
                    if self._sector_map.get(t.upper(), "Unknown") == sector
                ) + sector_spent.get(sector, 0.0)
                soft_sector_budget = equity * target_alloc - current_sector_val
                effective_sector_cap = self._effective_sector_cap(market_regime, conviction)
                hard_sector_budget = equity * effective_sector_cap - current_sector_val
                sector_budget = max(soft_sector_budget, 0.0)
                overflow_scale = self._sector_overflow_scale(market_regime, conviction)
                if hard_sector_budget > sector_budget and overflow_scale > 0:
                    sector_budget += (hard_sector_budget - sector_budget) * overflow_scale

                if sector_budget <= equity * 0.01:
                    logger.debug(
                        f"Sector budget exhausted for {sector} "
                        f"(target={target_alloc:.1%}), skipping {ticker}"
                    )
                    sector_budget_skips += 1
                    continue

                corr_stats = _candidate_correlation_stats(
                    recent_return_frame,
                    ticker,
                    held_tickers_now,
                )
                corr_limit = self._effective_max_pair_correlation(market_regime, conviction)
                if (
                    corr_stats is not None
                    and corr_stats["max_abs_corr"] > corr_limit
                ):
                    logger.debug(
                        "Correlation cap blocked %s (max_abs_corr=%.2f > %.2f)",
                        ticker,
                        corr_stats["max_abs_corr"],
                        corr_limit,
                    )
                    correlation_blocked_skips += 1
                    continue

                # ── Position sizing ───────────────────────────────────────────
                # Base size from conviction
                effective_max_position_pct = self._effective_max_position_pct(
                    market_regime,
                    conviction,
                )
                alloc_pct   = effective_max_position_pct * conviction
                alloc_pct   = np.clip(alloc_pct, 0.01, effective_max_position_pct)
                alloc_value = equity * alloc_pct
                target_weight_pre_caps = float(alloc_value / equity) if equity > 0 else np.nan
                model_suggested_weight = (
                    float(rl_weight_val)
                    if rl_weight_val is not None and np.isfinite(float(rl_weight_val))
                    else float(score)
                )
                allocation_steps = {
                    "model_suggested_weight": model_suggested_weight,
                    "target_weight_pre_caps": target_weight_pre_caps,
                    "post_sentiment_weight": np.nan,
                    "post_soft_signal_weight": np.nan,
                    "post_vol_weight": np.nan,
                    "post_sector_weight": np.nan,
                    "post_theme_weight": np.nan,
                    "post_weak_sleeve_weight": np.nan,
                    "post_correlation_weight": np.nan,
                    "post_low_price_weight": np.nan,
                    "target_weight_post_caps": np.nan,
                    "final_weight": np.nan,
                    "major_downweight_reason": "none",
                }

                sentiment_label = report.get("sentiment_label", self._sentiment_label(report))
                sentiment_scale = self._sentiment_weight_scale(sentiment_label)
                alloc_value *= sentiment_scale
                allocation_steps["post_sentiment_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )
                soft_weight_scale = float(report.get("soft_signal_weight_scale", 1.0) or 1.0)
                alloc_value *= soft_weight_scale
                allocation_steps["post_soft_signal_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                # Volatility scaling
                if risk_engine is not None:
                    alloc_value = risk_engine.vol_scale_allocation(alloc_value)
                allocation_steps["post_vol_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                # Constrain by sector budget
                alloc_value = min(alloc_value, sector_budget)
                allocation_steps["post_sector_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                current_theme_val = (
                    theme_values.get(theme, 0.0) + theme_spent.get(theme, 0.0)
                )
                theme_budget = equity * self.theme_max_pct - current_theme_val
                if theme_budget <= equity * 0.01:
                    logger.debug(
                        "Theme budget exhausted for %s (cap=%.1f%%), skipping %s",
                        theme,
                        self.theme_max_pct * 100,
                        ticker,
                    )
                    sector_budget_skips += 1
                    continue
                alloc_value = min(alloc_value, max(0.0, theme_budget))
                allocation_steps["post_theme_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                weak_theme_scale, weak_theme_detail = self._theme_health_scale(theme)
                alloc_value *= weak_theme_scale
                allocation_steps["post_weak_sleeve_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                if corr_stats is not None:
                    diversification_scale = np.clip(
                        1.0 - 0.35 * corr_stats["mean_abs_corr"],
                        0.70 if market_regime == 0 else (0.65 if market_regime == 1 else 0.60),
                        1.0,
                    )
                    alloc_value *= diversification_scale
                allocation_steps["post_correlation_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                # Constrain by penny budget
                if is_penny:
                    alloc_value = min(alloc_value, penny_budget)

                if price < 10.0:
                    current_low_price_val = (
                        low_price_values.get("sub_5", 0.0)
                        + low_price_values.get("5_to_10", 0.0)
                        + low_price_spent.get("sub_5", 0.0)
                        + low_price_spent.get("5_to_10", 0.0)
                    )
                    low_price_budget = equity * self.low_price_max_pct - current_low_price_val
                    if low_price_budget <= equity * 0.01:
                        logger.debug(
                            "Low-price budget exhausted (cap=%.1f%%), skipping %s",
                            self.low_price_max_pct * 100,
                            ticker,
                        )
                        penny_budget_skips += 1
                        continue
                    alloc_value = min(alloc_value, max(0.0, low_price_budget))
                allocation_steps["post_low_price_weight"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )
                allocation_steps["target_weight_post_caps"] = (
                    float(alloc_value / equity) if equity > 0 else np.nan
                )

                # Never spend more than 95% of remaining cash
                alloc_value = min(alloc_value, self.portfolio.cash * 0.95)

                # Pre-trade risk check
                if risk_engine is not None:
                    allowed, reason = risk_engine.check_pre_trade(alloc_value, self.portfolio)
                    if not allowed:
                        logger.debug(f"Pre-trade check blocked {ticker}: {reason}")
                        risk_blocked_skips += 1
                        continue
                    if "Capped to preserve cash floor" in reason:
                        effective_cash_floor = (
                            risk_engine._effective_cash_floor()
                            if hasattr(risk_engine, "_effective_cash_floor")
                            else risk_engine.cash_floor
                        )
                        max_spend = self.portfolio.cash - equity * effective_cash_floor
                        alloc_value = min(alloc_value, max(0.0, max_spend))

                final_weight = float(alloc_value / equity) if equity > 0 else np.nan
                allocation_steps["final_weight"] = final_weight
                prev_weight = target_weight_pre_caps
                drops = []
                for reason_name, key in [
                    ("sentiment_policy", "post_sentiment_weight"),
                    ("soft_signal_adjustment", "post_soft_signal_weight"),
                    ("volatility", "post_vol_weight"),
                    ("sector_cap", "post_sector_weight"),
                    ("theme_cap", "post_theme_weight"),
                    ("weak_sleeve", "post_weak_sleeve_weight"),
                    ("correlation_scale", "post_correlation_weight"),
                    ("low_price_or_penny_cap", "post_low_price_weight"),
                    ("cash_or_risk_cap", "final_weight"),
                ]:
                    cur_weight = allocation_steps[key]
                    if np.isfinite(prev_weight) and np.isfinite(cur_weight):
                        drops.append((prev_weight - cur_weight, reason_name))
                    if np.isfinite(cur_weight):
                        prev_weight = cur_weight
                material_drops = [drop for drop in drops if drop[0] > 0.0025]
                if material_drops:
                    allocation_steps["major_downweight_reason"] = max(material_drops)[1]
                def _impact(before: float, after: float) -> float:
                    if not (np.isfinite(before) and np.isfinite(after)):
                        return 0.0
                    return float(max(0.0, before - after))

                allocation_steps["sentiment_cap_impact"] = _impact(
                    target_weight_pre_caps,
                    allocation_steps["post_sentiment_weight"],
                )
                allocation_steps["volatility_cap_impact"] = _impact(
                    allocation_steps["post_soft_signal_weight"],
                    allocation_steps["post_vol_weight"],
                )
                allocation_steps["soft_signal_cap_impact"] = _impact(
                    allocation_steps["post_sentiment_weight"],
                    allocation_steps["post_soft_signal_weight"],
                )
                allocation_steps["sector_cap_impact"] = _impact(
                    allocation_steps["post_vol_weight"],
                    allocation_steps["post_sector_weight"],
                )
                allocation_steps["theme_cap_impact"] = _impact(
                    allocation_steps["post_sector_weight"],
                    allocation_steps["post_theme_weight"],
                )
                allocation_steps["correlation_cap_impact"] = _impact(
                    allocation_steps["post_weak_sleeve_weight"],
                    allocation_steps["post_correlation_weight"],
                )
                allocation_steps["weak_sleeve_cap_impact"] = _impact(
                    allocation_steps["post_theme_weight"],
                    allocation_steps["post_weak_sleeve_weight"],
                )
                allocation_steps["low_price_cap_impact"] = _impact(
                    allocation_steps["post_correlation_weight"],
                    allocation_steps["post_low_price_weight"],
                )
                allocation_steps["cash_or_risk_cap_impact"] = _impact(
                    allocation_steps["target_weight_post_caps"],
                    allocation_steps["final_weight"],
                )
                allocation_steps["total_cap_impact"] = _impact(
                    target_weight_pre_caps,
                    allocation_steps["final_weight"],
                )

                shares = alloc_value / price if price > 0 else 0
                if shares < 0.001 or alloc_value < 1.0:
                    tiny_alloc_skips += 1
                    continue

                # Build reason
                sent_label = sentiment_label
                soft_note = self._format_soft_signal_notes(
                    report.get("soft_signal_notes", {})
                )
                soft_reason = f" | SoftSignals={soft_note}" if soft_note else ""
                earnings_note = ""
                next_earnings = _get_next_earnings_date(ticker)
                if next_earnings:
                    days_to = (next_earnings - datetime.today().date()).days
                    earnings_note = f" | Earnings in {days_to}d"

                if self.rl_enabled and rl_score_val is not None:
                    corr_note = ""
                    if corr_stats is not None:
                        corr_note = f" | MaxCorr={corr_stats['max_abs_corr']:.2f}"
                    weak_note = ""
                    if weak_theme_scale < 1.0:
                        weak_note = (
                            f" | WeakSleeve={weak_theme_detail['weak_open_positions']}/"
                            f"{weak_theme_detail['open_positions']}"
                            f" avg={weak_theme_detail['avg_open_return_pct']:.1%}"
                            f" scale={weak_theme_scale:.2f}"
                        )
                    reason = (
                        f"logged_score={score:.4f} | rl_score={float(rl_rank_pct_val):.4f} | "
                        f"rl_rank_pct={float(rl_rank_pct_val):.4f} | "
                        f"rl_weight={float(rl_weight_val or 0.0):.4f} | "
                        f"rl_raw_weight={float(rl_raw_weight_val or 0.0):.6f} | "
                        f"composite_score={composite:.4f} | score_source=rl_rank_pct | "
                        f"target_weight_pre_caps={target_weight_pre_caps:.4f} | "
                        f"target_weight_post_caps={allocation_steps['target_weight_post_caps']:.4f} | "
                        f"final_weight={final_weight:.4f} | "
                        f"downweight_reason={allocation_steps['major_downweight_reason']} | "
                        f"rl_mode=true | Sector={sector} "
                        f"(target={target_alloc:.0%}) | Theme={theme} | "
                        f"Sentiment={sent_label}({self.sentiment_policy},scale={sentiment_scale:.2f})"
                        f"{soft_reason}{earnings_note}{corr_note}{weak_note} | "
                        f"{'PENNY ' if is_penny else ''}"
                        f"{(report.get('headlines') or [''])[0][:50]}"
                    )
                else:
                    corr_note = ""
                    if corr_stats is not None:
                        corr_note = f" | MaxCorr={corr_stats['max_abs_corr']:.2f}"
                    weak_note = ""
                    if weak_theme_scale < 1.0:
                        weak_note = (
                            f" | WeakSleeve={weak_theme_detail['weak_open_positions']}/"
                            f"{weak_theme_detail['open_positions']}"
                            f" avg={weak_theme_detail['avg_open_return_pct']:.1%}"
                            f" scale={weak_theme_scale:.2f}"
                        )
                    reason = (
                        f"logged_score={score:.4f} | composite_score={composite:.4f} | "
                        f"target_weight_pre_caps={target_weight_pre_caps:.4f} | "
                        f"target_weight_post_caps={allocation_steps['target_weight_post_caps']:.4f} | "
                        f"final_weight={final_weight:.4f} | "
                        f"downweight_reason={allocation_steps['major_downweight_reason']} | "
                        f"score_source=composite | Sector={sector} "
                        f"(target={target_alloc:.0%}) | Theme={theme} | "
                        f"Sentiment={sent_label}({self.sentiment_policy},scale={sentiment_scale:.2f})"
                        f"{soft_reason}{earnings_note}{corr_note}{weak_note} | "
                        f"{'PENNY ' if is_penny else ''}"
                        f"{(report.get('headlines') or [''])[0][:50]}"
                    )

                decisions.append(Decision(
                    action="BUY", ticker=ticker,
                    shares=shares, price=price,
                    score=score, reason=reason,
                ))

                sector_spent[sector] = sector_spent.get(sector, 0.0) + alloc_value
                theme_spent[theme] = theme_spent.get(theme, 0.0) + alloc_value
                low_price_spent[price_bucket] = low_price_spent.get(price_bucket, 0.0) + alloc_value
                if is_penny:
                    penny_value += alloc_value
                buyable_count += 1
                cycle_audit_rows.append(
                    {
                        "ticker": ticker,
                        "candidate_status": "buy_selected",
                        "logged_score": float(score),
                        "composite_score": float(composite),
                        "rl_rank_pct": float(rl_rank_pct_val) if rl_rank_pct_val is not None else np.nan,
                        "rl_weight": float(rl_weight_val) if rl_weight_val is not None else np.nan,
                        "rl_raw_weight": float(rl_raw_weight_val) if rl_raw_weight_val is not None else np.nan,
                        "sector": sector,
                        "theme_bucket": theme,
                        "low_price_bucket": price_bucket,
                        "alloc_value": float(alloc_value),
                        "alloc_pct": float(alloc_value / equity) if equity > 0 else np.nan,
                        "shares": float(shares),
                        "target_sector_alloc": float(target_alloc),
                        "max_position_pct": float(effective_max_position_pct),
                        "model_suggested_weight": allocation_steps["model_suggested_weight"],
                        "target_weight_pre_caps": allocation_steps["target_weight_pre_caps"],
                        "post_sentiment_weight": allocation_steps["post_sentiment_weight"],
                        "post_soft_signal_weight": allocation_steps["post_soft_signal_weight"],
                        "post_vol_weight": allocation_steps["post_vol_weight"],
                        "post_sector_weight": allocation_steps["post_sector_weight"],
                        "post_theme_weight": allocation_steps["post_theme_weight"],
                        "post_weak_sleeve_weight": allocation_steps["post_weak_sleeve_weight"],
                        "post_correlation_weight": allocation_steps["post_correlation_weight"],
                        "post_low_price_weight": allocation_steps["post_low_price_weight"],
                        "target_weight_post_caps": allocation_steps["target_weight_post_caps"],
                        "final_weight": allocation_steps["final_weight"],
                        "major_downweight_reason": allocation_steps["major_downweight_reason"],
                        "theme_effective_bet_count": float(theme_effective_bets),
                        "theme_cap": float(self.theme_max_pct),
                        "low_price_cap": float(self.low_price_max_pct),
                        "low_price_rank_policy": self.low_price_rank_policy,
                        "low_price_rank_scale": float(report.get("low_price_rank_scale", 1.0)),
                        "soft_signal_rank_scale": float(report.get("soft_signal_rank_scale", 1.0)),
                        "soft_signal_weight_scale": float(report.get("soft_signal_weight_scale", 1.0)),
                        "soft_signal_summary": self._format_soft_signal_notes(
                            report.get("soft_signal_notes", {})
                        ),
                        "llm_event_confidence": float(report.get("llm_event_confidence", 0.0) or 0.0),
                        "llm_event_trusted": bool(report.get("llm_event_trusted", False)),
                        "llm_thesis_impact": report.get("llm_thesis_impact", "unknown"),
                        "event_score": float(report.get("event_score", 0.0) or 0.0),
                        "event_risk_score": float(report.get("event_risk_score", 0.0) or 0.0),
                        "event_opportunity_score": float(
                            report.get("event_opportunity_score", 0.0) or 0.0
                        ),
                        "event_mention_count": int(report.get("event_mention_count", 0) or 0),
                        "event_top_types": ",".join(report.get("event_top_types", []) or []),
                        "crowd_mention_velocity": float(
                            report.get("crowd_mention_velocity", 0.0) or 0.0
                        ),
                        "pattern_score": float(report.get("pattern_score", 0.0) or 0.0),
                        "pattern_confidence": float(report.get("pattern_confidence", 0.0) or 0.0),
                        "primary_pattern": report.get("primary_pattern", "none"),
                        "rank_score": float(report.get("rank_score", score)),
                        "sentiment_label": sentiment_label,
                        "sentiment_policy": self.sentiment_policy,
                        "sentiment_weight_scale": float(sentiment_scale),
                        "sentiment_cap_impact": allocation_steps["sentiment_cap_impact"],
                        "soft_signal_cap_impact": allocation_steps["soft_signal_cap_impact"],
                        "volatility_cap_impact": allocation_steps["volatility_cap_impact"],
                        "sector_cap_impact": allocation_steps["sector_cap_impact"],
                        "theme_cap_impact": allocation_steps["theme_cap_impact"],
                        "weak_sleeve_cap_impact": allocation_steps["weak_sleeve_cap_impact"],
                        "correlation_cap_impact": allocation_steps["correlation_cap_impact"],
                        "low_price_cap_impact": allocation_steps["low_price_cap_impact"],
                        "cash_or_risk_cap_impact": allocation_steps["cash_or_risk_cap_impact"],
                        "total_cap_impact": allocation_steps["total_cap_impact"],
                    }
                )

            logger.debug(
                "Buy funnel: researched=%d slots=%d buys=%d penny_blocked=%d "
                "sector_blocked=%d corr_blocked=%d risk_blocked=%d tiny_alloc=%d",
                len(researched),
                n_slots,
                buyable_count,
                penny_budget_skips,
                sector_budget_skips,
                correlation_blocked_skips,
                risk_blocked_skips,
                tiny_alloc_skips,
            )
            if buyable_count == 0:
                logger.info(
                    "Buy funnel: researched=%d slots=%d buys=%d held=%d earnings=%d "
                    "no_research=%d threshold=%d sentiment_blocked=%d penny_blocked=%d sector_blocked=%d "
                    "corr_blocked=%d risk_blocked=%d tiny_alloc=%d",
                    len(researched),
                    n_slots,
                    buyable_count,
                    already_held_skips,
                    earnings_skips,
                    research_none_skips,
                    threshold_skips,
                    sentiment_blocked_skips,
                    penny_budget_skips,
                    sector_budget_skips,
                    correlation_blocked_skips,
                    risk_blocked_skips,
                    tiny_alloc_skips,
                )
            if cycle_audit_rows:
                self._last_cycle_audit = pd.DataFrame(cycle_audit_rows).sort_values(
                    ["candidate_status", "logged_score", "composite_score", "ticker"],
                    ascending=[True, False, False, True],
                ).reset_index(drop=True)
                self._last_cycle_audit = self._finalize_allocation_audit(
                    self._last_cycle_audit
                )

        # ── 8. Options decisions (suppressed when RL enabled) ─────────────────
        if not self.rl_enabled:
            options_decisions = self._evaluate_options(
                researched if n_slots > 0 and candidates else [],
                df_features,
            )
            decisions.extend(options_decisions)

        return self._reconcile_decisions(decisions)

    # ── Volatility-adjusted stop-loss ─────────────────────────────────────────

    def _research_with_fallback(
        self,
        ticker: str,
        df_features: pd.DataFrame,
    ) -> tuple[dict | None, bool]:
        report = research(ticker)
        if report is not None:
            self._attach_llm_sidecar_features(ticker, report)
            self._attach_event_sidecar_features(ticker, report)
            self._attach_pattern_features(ticker, report)
            return report, False

        fallback = research_from_features(df_features, ticker)
        if fallback is not None:
            self._attach_llm_sidecar_features(ticker, fallback)
            self._attach_event_sidecar_features(ticker, fallback)
            self._attach_pattern_features(ticker, fallback)
        return fallback, fallback is not None

    def _attach_llm_sidecar_features(self, ticker: str, report: dict) -> None:
        features = self.llm_sidecar_features.get(str(ticker).upper())
        if not isinstance(features, dict):
            return
        report["llm_sidecar"] = dict(features)
        report["llm_event_confidence"] = float(features.get("llm_event_confidence", 0.0) or 0.0)
        report["llm_event_trusted"] = bool(features.get("llm_event_trusted", False))
        report["llm_guidance_direction"] = features.get("guidance_direction", "unknown")
        report["llm_management_tone"] = features.get("management_tone", "unknown")
        report["llm_demand_outlook"] = features.get("demand_outlook", "unknown")
        report["llm_margin_outlook"] = features.get("margin_outlook", "unknown")
        report["llm_thesis_impact"] = features.get("thesis_impact", "unknown")
        report["llm_top_risks"] = list(features.get("top_risks") or [])[:5]
        report["llm_diagnostic_only"] = True
        report["llm_broker_influence"] = False
        report["llm_promotion_status"] = "unpromoted"
        if (
            self.llm_sidecar_broker_influence
            and self.allow_unpromoted_feature_influence
            and bool(report["llm_event_trusted"])
        ):
            score = self._llm_sidecar_soft_score(features)
            report["earnings_reaction_score"] = score
            report["llm_diagnostic_only"] = False
            report["llm_broker_influence"] = True

    def _attach_event_sidecar_features(self, ticker: str, report: dict) -> None:
        features = self.event_sidecar_features.get(str(ticker).upper())
        if not isinstance(features, dict):
            return
        report["event_sidecar"] = dict(features)
        report["event_score"] = float(features.get("event_score", 0.0) or 0.0)
        report["event_risk_score"] = float(features.get("event_risk_score", 0.0) or 0.0)
        report["event_opportunity_score"] = float(features.get("event_opportunity_score", 0.0) or 0.0)
        report["crowd_sentiment_score"] = float(features.get("crowd_sentiment_score", 0.0) or 0.0)
        report["crowd_mention_velocity"] = float(features.get("crowd_mention_velocity", 0.0) or 0.0)
        report["event_mention_count"] = int(features.get("mention_count", 0) or 0)
        report["event_source_count"] = int(features.get("source_count", 0) or 0)
        report["event_top_types"] = list(features.get("top_event_types") or [])[:5]
        report["event_top_events"] = list(features.get("top_events") or [])[:5]
        report["event_confidence"] = float(features.get("confidence", 0.0) or 0.0)
        report["event_diagnostic_only"] = True
        report["event_broker_influence"] = False
        report["event_promotion_status"] = "unpromoted"
        if self.event_sidecar_broker_influence:
            logger.warning("event_sidecar_broker_influence is diagnostics-only in this build.")

    def _attach_pattern_features(self, ticker: str, report: dict) -> None:
        features = self.pattern_features.get(str(ticker).upper())
        if not isinstance(features, dict):
            return
        report["pattern_sidecar"] = dict(features)
        report["pattern_score"] = float(features.get("pattern_score", 0.0) or 0.0)
        report["pattern_confidence"] = float(features.get("pattern_confidence", 0.0) or 0.0)
        report["primary_pattern"] = features.get("primary_pattern", "none")
        report["active_patterns"] = list(features.get("active_patterns") or [])[:5]
        report["pattern_diagnostic_only"] = True
        report["pattern_broker_influence"] = False
        report["pattern_promotion_status"] = "unpromoted"
        if self.pattern_sidecar_broker_influence:
            logger.warning("pattern_sidecar_broker_influence is diagnostics-only in this build.")

    @staticmethod
    def _llm_sidecar_soft_score(features: dict) -> float:
        pos = {"positive", "strengthens"}
        neg = {"negative", "weakens"}
        score = 0
        for key in (
            "guidance_direction",
            "management_tone",
            "demand_outlook",
            "margin_outlook",
            "thesis_impact",
        ):
            value = str(features.get(key, "unknown")).lower()
            if value in pos:
                score += 1
            elif value in neg:
                score -= 1
        return float(np.clip(score / 5.0, -1.0, 1.0))

    def _sentiment_label(self, report: dict) -> str:
        sent = report.get("sentiment", {})
        if isinstance(sent, dict):
            sent = sent.get("sentiment", "neutral")
        label = str(sent or "neutral").strip().lower()
        if label not in {"positive", "negative", "neutral"}:
            return "neutral"
        return label

    def _sentiment_weight_scale(self, label: str) -> float:
        if self.sentiment_policy in {"penalize", "penalize_negative"} and label == "negative":
            return float(np.clip(self.sentiment_negative_weight_mult, 0.0, 1.0))
        return 1.0

    def _sentiment_vetoes_entry(self, label: str, composite_score: float) -> bool:
        if self.sentiment_policy not in {"veto", "veto_negative"}:
            return False
        return label == "negative" and float(composite_score) < float(
            self.sentiment_veto_composite_floor
        )

    @staticmethod
    def _bounded_signal_score(value) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(score):
            return None
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def _scale_from_signal(score: float | None, strength: float) -> float:
        if score is None:
            return 1.0
        try:
            strength_val = abs(float(strength))
        except (TypeError, ValueError):
            strength_val = 0.0
        return float(np.clip(1.0 + float(score) * strength_val, 0.50, 1.50))

    def _earnings_reaction_score(self, report: dict) -> tuple[float | None, str]:
        for key in (
            "earnings_reaction_score",
            "post_earnings_reaction_score",
            "earnings_surprise_score",
        ):
            score = self._bounded_signal_score(report.get(key))
            if score is not None:
                return score, key

        surprise = report.get("earnings_surprise_pct")
        post_return = (
            report.get("post_earnings_return_1d")
            if report.get("post_earnings_return_1d") is not None
            else report.get("earnings_post_return_1d")
        )
        try:
            if surprise is not None and post_return is not None:
                surprise_score = np.tanh(float(surprise) / 0.10)
                reaction_score = np.tanh(float(post_return) / 0.05)
                return self._bounded_signal_score(
                    0.5 * surprise_score + 0.5 * reaction_score
                ), "surprise_plus_reaction"
        except (TypeError, ValueError):
            pass
        return None, "no_data"

    def _macro_regime_score(self, market_regime: int | None, report: dict) -> tuple[float | None, str]:
        for key in ("macro_risk_score", "market_risk_score", "regime_risk_score"):
            score = self._bounded_signal_score(report.get(key))
            if score is not None:
                return self._apply_macro_regime_mode(score, key, market_regime)

        if market_regime is None:
            return None, "no_data"
        regime_map = {
            0: 0.50,   # risk-on
            1: 0.20,   # constructive/neutral
            2: -0.10,  # choppy
            3: -0.60,  # risk-off
        }
        return self._apply_macro_regime_mode(
            float(regime_map.get(int(market_regime), 0.0)),
            f"regime_{market_regime}",
            market_regime,
        )

    def _apply_macro_regime_mode(
        self,
        score: float | None,
        source: str,
        market_regime: int | None,
    ) -> tuple[float | None, str]:
        if score is None:
            return None, source
        mode = self.macro_regime_mode
        score = float(score)
        if mode in {"standard", "", "none"}:
            return score, source
        if mode in {"risk_off_only", "no_bull_boost", "volatility_scaler_only"}:
            return min(score, 0.0), f"{source}:{mode}"
        if mode == "drawdown_guard_only":
            try:
                regime = int(market_regime) if market_regime is not None else None
            except (TypeError, ValueError):
                regime = None
            guarded = min(score, 0.0) if regime in {2, 3} else 0.0
            return guarded, f"{source}:{mode}"
        return score, source

    def _insider_signal_score(self, report: dict) -> tuple[float | None, str]:
        for key in (
            "insider_signal_score",
            "insider_net_buy_score",
            "insider_cluster_score",
            "insider_activity_score",
        ):
            score = self._bounded_signal_score(report.get(key))
            if score is not None:
                return score, key
        return None, "no_data"

    def _soft_signal_adjustments(
        self,
        report: dict,
        market_regime: int | None,
    ) -> tuple[float, float, dict]:
        notes: dict[str, dict] = {}
        rank_scale = 1.0
        weight_scale = 1.0

        score, source = self._earnings_reaction_score(report)
        if self.earnings_reaction_enabled or score is not None:
            influence = bool(self.earnings_reaction_enabled and self.allow_unpromoted_feature_influence)
            rank = self._scale_from_signal(score, self.earnings_reaction_rank_strength) if influence else 1.0
            weight = self._scale_from_signal(score, self.earnings_reaction_weight_strength) if influence else 1.0
            rank_scale *= rank
            weight_scale *= weight
            notes["earnings"] = {
                "score": score,
                "source": source,
                "rank": rank,
                "weight": weight,
                "diagnostic_only": not influence,
                "broker_influence": influence,
                "promotion_status": "unpromoted",
            }

        score, source = self._macro_regime_score(market_regime, report)
        if self.macro_regime_enabled or score is not None:
            influence = bool(self.macro_regime_enabled and self.allow_unpromoted_feature_influence)
            weight = self._scale_from_signal(score, self.macro_regime_weight_strength) if influence else 1.0
            weight_scale *= weight
            notes["macro"] = {
                "score": score,
                "source": source,
                "rank": 1.0,
                "weight": weight,
                "diagnostic_only": not influence,
                "broker_influence": influence,
                "promotion_status": "unpromoted",
            }

        score, source = self._insider_signal_score(report)
        if self.insider_adjustment_enabled or score is not None:
            influence = bool(self.insider_adjustment_enabled and self.allow_unpromoted_feature_influence)
            rank = self._scale_from_signal(score, self.insider_adjustment_rank_strength) if influence else 1.0
            weight = self._scale_from_signal(score, self.insider_adjustment_weight_strength) if influence else 1.0
            rank_scale *= rank
            weight_scale *= weight
            notes["insider"] = {
                "score": score,
                "source": source,
                "rank": rank,
                "weight": weight,
                "diagnostic_only": not influence,
                "broker_influence": influence,
                "promotion_status": "unpromoted",
            }

        return (
            float(np.clip(rank_scale, 0.50, 1.50)),
            float(np.clip(weight_scale, 0.50, 1.50)),
            notes,
        )

    @staticmethod
    def _format_soft_signal_notes(notes: dict) -> str:
        if not isinstance(notes, dict) or not notes:
            return ""
        parts = []
        for name in ("earnings", "macro", "insider"):
            detail = notes.get(name)
            if not isinstance(detail, dict):
                continue
            source = str(detail.get("source", "no_data"))
            score = detail.get("score")
            score_label = "no_data" if score is None else f"{float(score):+.2f}"
            weight = float(detail.get("weight", 1.0) or 1.0)
            parts.append(f"{name}:{source}:{score_label}:w{weight:.2f}")
        return ",".join(parts)

    def _theme_health_scale(self, theme: str) -> tuple[float, dict]:
        """
        Downweight new entries into a theme when the current open sleeve is
        already failing together. This converts the diagnostics signal into a
        conservative allocator response without forcing exits.
        """
        min_positions = max(1, int(self.weak_theme_min_positions))
        threshold = float(self.weak_theme_return_threshold)
        penalty = float(np.clip(self.weak_theme_penalty_mult, 0.0, 1.0))

        rows = []
        for held_ticker, pos in self.portfolio.positions.items():
            held_theme = theme_bucket(
                str(held_ticker),
                self._sector_map.get(str(held_ticker).upper(), "Unknown"),
            )
            if held_theme != theme:
                continue
            avg_cost = float(pos.get("avg_cost", 0.0) or 0.0)
            last_price = float(pos.get("last_price", 0.0) or 0.0)
            if avg_cost <= 0 or last_price <= 0:
                continue
            ret = (last_price - avg_cost) / avg_cost
            rows.append({"ticker": str(held_ticker).upper(), "return_pct": float(ret)})

        n_open = len(rows)
        weak_count = sum(1 for row in rows if row["return_pct"] < 0.0)
        avg_return = (
            float(sum(row["return_pct"] for row in rows) / n_open)
            if n_open else 0.0
        )
        is_weak = (
            n_open >= min_positions
            and weak_count >= min_positions
            and avg_return <= threshold
        )
        cooldown_cycles = max(0, int(self.weak_theme_cooldown_cycles))
        cooldown_min_hits = max(1, int(self.weak_theme_cooldown_min_hits))
        state = self._weak_theme_states.setdefault(
            theme,
            {
                "weak_hits": 0,
                "cooldown_remaining": 0,
                "last_cycle_id": -1,
            },
        )
        cycle_id = int(getattr(self, "_weak_theme_cycle_id", 0))
        if state.get("last_cycle_id") != cycle_id:
            if int(state.get("cooldown_remaining", 0)) > 0:
                state["cooldown_remaining"] = int(state["cooldown_remaining"]) - 1
            state["weak_hits"] = int(state.get("weak_hits", 0)) + 1 if is_weak else 0
            if (
                cooldown_cycles > 0
                and is_weak
                and int(state["weak_hits"]) >= cooldown_min_hits
            ):
                state["cooldown_remaining"] = max(
                    int(state.get("cooldown_remaining", 0)),
                    cooldown_cycles,
                )
            state["last_cycle_id"] = cycle_id
        cooldown_active = int(state.get("cooldown_remaining", 0)) > 0
        scale = 0.0 if cooldown_active else (penalty if is_weak else 1.0)
        detail = {
            "theme": theme,
            "open_positions": n_open,
            "weak_open_positions": weak_count,
            "avg_open_return_pct": avg_return,
            "threshold": threshold,
            "scale": scale,
            "tickers": [row["ticker"] for row in rows],
            "weak_hits": int(state.get("weak_hits", 0)),
            "cooldown_active": bool(cooldown_active),
            "cooldown_remaining": int(state.get("cooldown_remaining", 0)),
        }
        return scale, detail

    def _finalize_allocation_audit(self, audit_df: pd.DataFrame) -> pd.DataFrame:
        if audit_df.empty or "candidate_status" not in audit_df.columns:
            self._last_allocation_summary = {}
            return audit_df

        selected_mask = audit_df["candidate_status"].eq("buy_selected")
        selected = audit_df[selected_mask].copy()
        if selected.empty:
            self._last_allocation_summary = {}
            return audit_df

        signal_col = "rl_rank_pct" if "rl_rank_pct" in selected.columns else "logged_score"
        selected["signal_rank"] = selected[signal_col].rank(
            method="first",
            ascending=False,
        )
        selected["final_weight_rank"] = selected["final_weight"].rank(
            method="first",
            ascending=False,
        )
        selected["rank_weight_delta"] = selected["final_weight_rank"] - selected["signal_rank"]

        for idx, row in selected.iterrows():
            audit_df.at[idx, "signal_rank"] = float(row["signal_rank"])
            audit_df.at[idx, "final_weight_rank"] = float(row["final_weight_rank"])
            audit_df.at[idx, "rank_weight_delta"] = float(row["rank_weight_delta"])
            audit_df.at[idx, "rank_weight_mismatch"] = bool(abs(row["rank_weight_delta"]) >= 2)

        summary: dict[str, object] = {}
        selected["final_weight"] = selected["final_weight"].fillna(0.0).astype(float)
        selected["target_weight_pre_caps"] = selected[
            "target_weight_pre_caps"
        ].fillna(0.0).astype(float)

        final_weight_by_ticker = selected.set_index("ticker")["final_weight"].sort_values(
            ascending=False
        )
        top_positions = [
            {"ticker": str(ticker), "weight": float(weight)}
            for ticker, weight in final_weight_by_ticker.head(5).items()
        ]
        summary["top_positions"] = top_positions
        summary["largest_position"] = top_positions[0] if top_positions else None
        summary["top_1_concentration"] = (
            float(final_weight_by_ticker.iloc[0]) if not final_weight_by_ticker.empty else 0.0
        )
        summary["top_3_concentration"] = float(final_weight_by_ticker.head(3).sum())
        summary["effective_position_bet_count"] = effective_bet_count(
            exposure_weights({str(k): float(v) for k, v in final_weight_by_ticker.items()})
        )

        if "theme_bucket" in selected.columns:
            pre_cap_theme_exposure = (
                selected.groupby("theme_bucket")["target_weight_pre_caps"]
                .sum()
                .sort_values(ascending=False)
            )
            theme_exposure = (
                selected.groupby("theme_bucket")["final_weight"].sum().sort_values(ascending=False)
            )
            summary["pre_cap_theme_exposure"] = {
                str(k): float(v) for k, v in pre_cap_theme_exposure.items()
            }
            summary["theme_exposure"] = {
                str(k): float(v) for k, v in theme_exposure.items()
            }
            summary["theme_effective_bet_count"] = effective_bet_count(
                exposure_weights(summary["theme_exposure"])
            )
            summary["largest_theme"] = (
                {
                    "theme": str(theme_exposure.index[0]),
                    "weight": float(theme_exposure.iloc[0]),
                }
                if not theme_exposure.empty
                else None
            )
            summary["top_theme_concentration"] = (
                float(theme_exposure.iloc[0]) if not theme_exposure.empty else 0.0
            )
            summary["theme_overloads"] = {
                str(k): float(v)
                for k, v in theme_exposure.items()
                if float(v) > float(self.theme_max_pct) + 1e-9
            }

        if "sector" in selected.columns:
            pre_cap_sector_exposure = (
                selected.groupby("sector")["target_weight_pre_caps"]
                .sum()
                .sort_values(ascending=False)
            )
            sector_exposure = (
                selected.groupby("sector")["final_weight"].sum().sort_values(ascending=False)
            )
            summary["pre_cap_sector_exposure"] = {
                str(k): float(v) for k, v in pre_cap_sector_exposure.items()
            }
            summary["sector_exposure"] = {
                str(k): float(v) for k, v in sector_exposure.items()
            }
            summary["sector_effective_bet_count"] = effective_bet_count(
                exposure_weights(summary["sector_exposure"])
            )

        low_price_exposure = selected.loc[
            selected.get("low_price_bucket", pd.Series(index=selected.index)).isin(
                ["sub_5", "5_to_10"]
            ),
            "final_weight",
        ].sum()
        summary["low_price_exposure"] = float(low_price_exposure)

        cap_impact_cols = [
            "volatility_cap_impact",
            "sector_cap_impact",
            "theme_cap_impact",
            "weak_sleeve_cap_impact",
            "correlation_cap_impact",
            "low_price_cap_impact",
            "cash_or_risk_cap_impact",
            "sentiment_cap_impact",
            "total_cap_impact",
        ]
        cap_impact = {}
        cap_counts = {}
        for col in cap_impact_cols:
            if col in selected.columns:
                impacts = selected[col].fillna(0.0).astype(float)
                cap_impact[col] = float(impacts.sum())
                cap_counts[col] = int((impacts > 0.0025).sum())
        summary["cap_impact"] = cap_impact
        summary["cap_intervention_counts"] = cap_counts

        per_cap_cols = [col for col in cap_impact_cols if col != "total_cap_impact"]
        top_interventions = []
        for _, row in selected.iterrows():
            impacts = {
                col: float(row.get(col, 0.0) or 0.0)
                for col in per_cap_cols
            }
            if not impacts:
                continue
            cap_class, impact = max(impacts.items(), key=lambda item: item[1])
            if impact <= 0.0025:
                continue
            top_interventions.append(
                {
                    "ticker": str(row["ticker"]),
                    "cap_class": cap_class.replace("_impact", ""),
                    "impact": float(impact),
                    "pre_cap_weight": float(row["target_weight_pre_caps"]),
                    "final_weight": float(row["final_weight"]),
                    "major_downweight_reason": str(row.get("major_downweight_reason", "none")),
                }
            )
        summary["top_cap_interventions"] = sorted(
            top_interventions,
            key=lambda item: item["impact"],
            reverse=True,
        )[:5]

        mismatches = selected.reindex(
            selected["rank_weight_delta"].abs().sort_values(ascending=False).index
        ).head(5)
        summary["rank_weight_mismatches"] = [
            {
                "ticker": str(row["ticker"]),
                "signal_rank": float(row["signal_rank"]),
                "final_weight_rank": float(row["final_weight_rank"]),
                "rank_weight_delta": float(row["rank_weight_delta"]),
                "final_weight": float(row["final_weight"]),
            }
            for _, row in mismatches.iterrows()
            if abs(float(row["rank_weight_delta"])) >= 1
        ]

        self._last_allocation_summary = summary
        logger.info(
            "Allocation exposure summary: themes=%s low_price=%.1f%% "
            "top1=%.1f%% top3=%.1f%% top_theme=%.1f%% "
            "effective_theme_bets=%.2f effective_position_bets=%.2f cap_impact=%s",
            summary.get("theme_exposure", {}),
            float(summary.get("low_price_exposure", 0.0)) * 100,
            float(summary.get("top_1_concentration", 0.0)) * 100,
            float(summary.get("top_3_concentration", 0.0)) * 100,
            float(summary.get("top_theme_concentration", 0.0)) * 100,
            float(summary.get("theme_effective_bet_count", 0.0)),
            float(summary.get("effective_position_bet_count", 0.0)),
            summary.get("cap_impact", {}),
        )
        if summary.get("top_cap_interventions"):
            logger.info("Top cap interventions: %s", summary["top_cap_interventions"])
        if summary.get("rank_weight_mismatches"):
            logger.info("Rank/weight mismatches: %s", summary["rank_weight_mismatches"])
        if summary.get("theme_overloads"):
            logger.warning("Theme exposure exceeds cap: %s", summary["theme_overloads"])

        return audit_df

    def _get_stop_loss_pct(self, ticker: str, pos: dict) -> float:
        """
        Compute ATR-based stop-loss for a position.
        Falls back to floor if ATR unavailable.
        """
        try:
            data = fetch_ticker_data(ticker, days=30)
            if data is None or len(data) < 14:
                return self.stop_loss_pct_floor

            # ATR calculation
            h = data["high"].values
            l = data["low"].values
            c = data["close"].values
            prev_c = np.roll(c, 1); prev_c[0] = c[0]
            tr  = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
            atr = float(np.mean(tr[-14:]))

            entry_price = pos.get("avg_cost", c[-1])
            if entry_price <= 0:
                return self.stop_loss_pct_floor

            stop_pct = (self.stop_loss_atr_mult * atr) / entry_price
            return float(np.clip(stop_pct, self.stop_loss_pct_floor, self.stop_loss_pct_ceil))

        except Exception:
            return self.stop_loss_pct_floor

    # ── Earnings awareness ────────────────────────────────────────────────────

    def _near_earnings(self, ticker: str) -> bool:
        """Return True if earnings are within avoid_earnings_days."""
        if self.avoid_earnings_days <= 0:
            return False
        next_date = _get_next_earnings_date(ticker)
        if next_date is None:
            return False
        days_away = (next_date - datetime.today().date()).days
        return 0 <= days_away <= self.avoid_earnings_days

    # ── Sector map refresh ────────────────────────────────────────────────────

    def _maybe_refresh_sector_map(self, df_features: pd.DataFrame):
        """Refresh sector map weekly or on first run."""
        now = datetime.now()
        if (
            self._sector_cache_date is None
            or (now - self._sector_cache_date).days >= 7
        ):
            tickers = df_features.index.get_level_values("ticker").unique().tolist()
            # Also include held positions
            tickers += list(self.portfolio.positions.keys())
            tickers  = list(set(tickers))
            logger.info(f"Refreshing sector map for {len(tickers)} tickers...")
            self._sector_map = get_sectors_bulk(tickers)
            self._sector_cache_date = now

    # ── RL model validation ───────────────────────────────────────────────────

    def _assert_model_available(self) -> None:
        """
        Validate that the RL checkpoint exists and contains the required keys.

        The inference wrapper builds a dynamic observation tensor sized to the
        current shortlist, so we do NOT check n_assets == len(shortlist) here —
        that check is both brittle (shortlist length varies cycle to cycle) and
        wrong (the model accepts any asset_list length at inference time).

        Instead we verify:
          - The checkpoint file exists on disk.
          - The file loads without error.
          - The required keys (model_cfg, model_state) are present.

        Raises RuntimeError (logged as CRITICAL) on any failure.
        """
        if not self.rl_checkpoint_path or not os.path.exists(self.rl_checkpoint_path):
            msg = (
                f"RL checkpoint not found: '{self.rl_checkpoint_path}'. "
                "Aborting cycle — set rl_enabled=false or provide a valid checkpoint."
            )
            logger.critical(msg)
            raise RuntimeError(msg)

        try:
            ckpt = torch.load(self.rl_checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            msg = f"Failed to load RL checkpoint '{self.rl_checkpoint_path}': {exc}"
            logger.critical(msg)
            raise RuntimeError(msg) from exc

        missing = [k for k in ("model_cfg", "model_state") if k not in ckpt]
        if missing:
            msg = (
                f"RL checkpoint '{self.rl_checkpoint_path}' is missing required keys: "
                f"{missing}. Aborting cycle."
            )
            logger.critical(msg)
            raise RuntimeError(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_stock_exit_action(decision: Decision) -> bool:
        return decision.action in {"SELL", "SELL_PARTIAL"}

    @staticmethod
    def _exit_priority(decision: Decision) -> int:
        reason = str(getattr(decision, "reason", "") or "").strip().lower()

        if "stop-loss" in reason:
            return 0
        if "trailing stop" in reason:
            return 1
        if "full take-profit" in reason:
            return 2
        if decision.action == "SELL" and reason.startswith("rl exit"):
            return 3
        if decision.action == "SELL" and "signal deteriorated" in reason:
            return 4
        if decision.action == "SELL_PARTIAL" and "rl conviction drop" in reason:
            return 5
        if decision.action == "SELL_PARTIAL" and "partial take-profit" in reason:
            return 6
        if decision.action == "SELL":
            return 7
        return 8

    def _reconcile_decisions(self, decisions: list[Decision]) -> list[Decision]:
        """
        Keep at most one stock-exit action per ticker per cycle.

        Independent checks can fire on the same position during a single pass.
        The execution layer should only see the highest-priority exit.
        """
        best_exit_by_ticker: dict[str, tuple[tuple[int, int], Decision]] = {}
        exit_counts: dict[str, int] = {}

        for idx, decision in enumerate(decisions):
            if not self._is_stock_exit_action(decision):
                continue

            exit_counts[decision.ticker] = exit_counts.get(decision.ticker, 0) + 1
            rank = (self._exit_priority(decision), idx)
            incumbent = best_exit_by_ticker.get(decision.ticker)
            if incumbent is None or rank < incumbent[0]:
                best_exit_by_ticker[decision.ticker] = (rank, decision)

        if not best_exit_by_ticker:
            return decisions

        dropped = sum(count - 1 for count in exit_counts.values() if count > 1)
        if dropped > 0:
            logger.debug("Reconciled %d lower-priority duplicate stock exit(s).", dropped)

        reconciled: list[Decision] = []
        emitted_exit_tickers: set[str] = set()
        for decision in decisions:
            if not self._is_stock_exit_action(decision):
                reconciled.append(decision)
                continue

            winner = best_exit_by_ticker.get(decision.ticker)
            if winner is None:
                continue
            if decision is winner[1] and decision.ticker not in emitted_exit_tickers:
                reconciled.append(decision)
                emitted_exit_tickers.add(decision.ticker)

        return reconciled

    def _rl_exit_checks(
        self,
        held_tickers: list[str],
        rl_scores: "pd.Series",
    ) -> list[Decision]:
        """
        Phase 2: Compare each held ticker's current RL rank against the rank
        recorded at entry.

        RL scores from mode="rank" are already shortlist-relative rank
        percentiles, so Phase 2 compares those values directly:

          - SELL if current rank percentile < rl_exit_threshold
            (e.g. 0.20 means "bottom 20% of today's shortlist")
          - SELL_PARTIAL if rank dropped by more than rl_conviction_drop
            relative to entry rank percentile

        Tickers not present in rl_scores (fell off the shortlist) are skipped
        and deferred to heuristic exits.
        """
        exit_decisions: list[Decision] = []

        if rl_scores.empty:
            return exit_decisions

        # Higher rl_scores values already mean higher shortlist percentile.
        # Higher score → higher percentile (1.0 = top of shortlist).
        n = len(rl_scores)

        for ticker in held_tickers:
            pos = self.portfolio.positions.get(ticker)
            if pos is None:
                continue

            if ticker not in rl_scores.index:
                logger.debug(
                    "RL exit check: %s not in cycle rl_scores — deferring to heuristic exits",
                    ticker,
                )
                continue

            current_rank_pct = float(rl_scores[ticker])
            price = pos.get("last_price", 0.0)
            shares = pos.get("shares", 0.0)
            # Use rank-percentile at entry if available; fall back to raw score
            # converted to a rough percentile for backward compatibility.
            entry_rank_pct = pos.get("rl_rank_pct_at_entry")
            if entry_rank_pct is None and pos.get("rl_score_at_entry") is not None:
                # Legacy: approximate entry rank from raw score (rough but better than None)
                entry_rank_pct = float(pos["rl_score_at_entry"])

            # Check absolute rank threshold
            if current_rank_pct < self.rl_exit_threshold:
                drop_str = (
                    f"{entry_rank_pct - current_rank_pct:.4f}"
                    if entry_rank_pct is not None else "N/A (no entry rank)"
                )
                logger.info(
                    "RL exit (SELL) %s: entry_rank_pct=%s  current_rank_pct=%.4f  "
                    "drop=%s  threshold=rl_exit_threshold(%.2f)  shortlist_n=%d",
                    ticker,
                    f"{entry_rank_pct:.4f}" if entry_rank_pct is not None else "N/A",
                    current_rank_pct,
                    drop_str,
                    self.rl_exit_threshold,
                    n,
                )
                exit_decisions.append(Decision(
                    action="SELL",
                    ticker=ticker,
                    shares=shares,
                    price=price,
                    score=current_rank_pct,
                    reason=(
                        f"RL exit: rank_pct={current_rank_pct:.4f} < "
                        f"rl_exit_threshold={self.rl_exit_threshold:.2f} | "
                        f"entry_rank_pct={entry_rank_pct} | "
                        f"shortlist_n={n} | rl_mode=true"
                    ),
                ))
                continue

            # Check conviction drop threshold (only when entry rank is known)
            if entry_rank_pct is not None:
                drop = entry_rank_pct - current_rank_pct
                if drop > self.rl_conviction_drop:
                    half_shares = shares * 0.5
                    logger.info(
                        "RL exit (SELL_PARTIAL) %s: entry_rank_pct=%.4f  "
                        "current_rank_pct=%.4f  drop=%.4f  "
                        "threshold=rl_conviction_drop(%.2f)  shortlist_n=%d",
                        ticker,
                        entry_rank_pct,
                        current_rank_pct,
                        drop,
                        self.rl_conviction_drop,
                        n,
                    )
                    exit_decisions.append(Decision(
                        action="SELL_PARTIAL",
                        ticker=ticker,
                        shares=half_shares,
                        price=price,
                        score=current_rank_pct,
                        reason=(
                            f"RL conviction drop: entry_rank_pct={entry_rank_pct:.4f} | "
                            f"current_rank_pct={current_rank_pct:.4f} | "
                            f"drop={drop:.4f} > rl_conviction_drop={self.rl_conviction_drop:.2f} | "
                            f"shortlist_n={n} | rl_mode=true"
                        ),
                    ))

        return exit_decisions



    def _screen_candidates(
        self, df_features: pd.DataFrame, top_n: int = 100
    ) -> list[str]:
        import os
        from pipeline.screener import SCREENER_CKPT

        if os.path.exists(SCREENER_CKPT):
            try:
                from pipeline.screener import run_screener
                screener_device = self.device or torch.device("cpu")
                results = run_screener(
                    df_features,
                    device=screener_device,
                    top_n=top_n,
                )
                return self._filter_screened_tickers(results["ticker"].tolist(), top_n)
            except Exception as e:
                logger.warning(f"Screener failed, using rule-based fallback: {e}")

        try:
            dates     = sorted(df_features.index.get_level_values("date").unique())
            last_date = dates[-1]
            snap      = df_features.loc[last_date].copy()
            snap["_rank"] = (
                snap.get("ret_5d",    0) * 0.3 +
                snap.get("vol_ratio", 0) * 0.2 +
                snap.get("sent_net",  0) * 0.3 +
                snap.get("macd_hist", 0) * 0.2
            )
            return self._filter_screened_tickers(
                snap.nlargest(top_n, "_rank").index.tolist(),
                top_n,
            )
        except Exception:
            return []

    def _filter_screened_tickers(self, tickers: list[str], top_n: int) -> list[str]:
        """
        Prefer classified operating-company names over the local Unknown bucket.
        In practice this strips many ETF/fund/odd-instrument symbols from the
        shortlist while keeping ordering stable.
        """
        deduped: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            t = str(ticker).upper()
            if t in seen or t == "CASH":
                continue
            seen.add(t)
            deduped.append(t)

        classified = [
            ticker for ticker in deduped
            if self._sector_map.get(ticker, "Unknown") != "Unknown"
        ]
        if classified:
            dropped = len(deduped) - len(classified)
            if dropped > 0:
                logger.debug(
                    "Shortlist filter dropped %d Unknown-sector ticker(s) before research.",
                    dropped,
                )
            return classified[:top_n]
        return deduped[:top_n]

    def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for ticker in tickers:
            quote = fetch_latest_market_price(ticker)
            if quote.get("price"):
                prices[ticker] = float(quote["price"])
                continue
            data = fetch_ticker_data(ticker, days=5)
            if data is not None and not data.empty:
                prices[ticker] = float(data["close"].iloc[-1])
        return prices

    def _evaluate_options(
        self,
        researched: list[dict],
        df_features: pd.DataFrame,
    ) -> list[Decision]:
        """
        Evaluate options opportunities for top-scored stocks.
        Also checks existing option positions for expiry/close signals.
        """
        from broker.options import analyse_options, OptionsBook

        decisions = []
        equity    = self.portfolio.equity

        # ── Check existing option positions ───────────────────────────────────
        current_prices = self._get_current_prices(
            list({c.ticker for c in self.portfolio.options.positions.values()})
        )
        expired_keys = self.portfolio.options.check_expirations(
            current_prices, self.portfolio
        )
        if expired_keys:
            logger.info(f"  {len(expired_keys)} option(s) expired/assigned")

        # Close options where P&L > 50% of max profit (lock in gains)
        for key, contract in list(self.portfolio.options.positions.items()):
            spot = current_prices.get(contract.ticker, 0.0)
            if spot <= 0:
                continue
            pnl = contract.pnl(spot)
            max_profit = abs(contract.total_cost)
            if max_profit > 0 and pnl / max_profit >= 0.50:
                decisions.append(Decision(
                    action="CLOSE_OPTION", ticker=contract.ticker,
                    shares=0, price=spot, score=0.9,
                    reason=f"Option P&L at {pnl/max_profit:.0%} of max — closing",
                ))

        # ── Open new option positions for top candidates ───────────────────────
        # Only use up to 10% of equity for options total
        options_budget = equity * 0.10
        current_options_value = sum(
            abs(c.total_cost) for c in self.portfolio.options.positions.values()
            if c.position == "long"
        )
        remaining_options_budget = max(0, options_budget - current_options_value)

        if remaining_options_budget < 100:
            return decisions

        # Consider top 5 researched stocks for options
        for report in (researched or [])[:5]:
            ticker = report["ticker"]
            score  = report["composite_score"]
            price  = report["price"]

            # Skip penny stocks for options (usually no liquid chain)
            if price < 5.0:
                continue

            # Skip if already have an option on this ticker
            if any(c.ticker == ticker for c in self.portfolio.options.positions.values()):
                continue

            sent = report.get("sentiment", {})
            if isinstance(sent, dict):
                sent_net = sent.get("sent_net", 0.0)
            else:
                sent_net = 0.0

            atr_pct = abs(report.get("atr", 0.02))
            per_trade_budget = min(remaining_options_budget * 0.3, equity * 0.03)

            contracts = analyse_options(
                ticker       = ticker,
                current_price= price,
                signal_score = score,
                sentiment_net= sent_net,
                atr_pct      = atr_pct,
                budget       = per_trade_budget,
            )

            if contracts:
                for contract in contracts:
                    decisions.append(Decision(
                        action="OPEN_OPTION", ticker=ticker,
                        shares=contract.contracts, price=contract.premium_paid,
                        score=score,
                        reason=f"{contract.strategy} | DTE={contract.days_to_expiry}",
                    ))
                    # Store contract on decision for execution
                    decisions[-1]._option_contract = contract

        return decisions


# ── Earnings date helper ──────────────────────────────────────────────────────

def _get_next_earnings_date(ticker: str):
    """
    Fetch next earnings date from yfinance.
    Returns a date object or None.
    """
    try:
        import yfinance as yf
        import os, sys
        from contextlib import contextmanager

        @contextmanager
        def _quiet():
            with open(os.devnull, "w") as dn:
                old = sys.stderr; sys.stderr = dn
                try: yield
                finally: sys.stderr = old

        with _quiet():
            cal = yf.Ticker(ticker).calendar

        if cal is None or cal.empty:
            return None

        # calendar returns a DataFrame with 'Earnings Date' column
        if "Earnings Date" in cal.columns:
            dates = pd.to_datetime(cal["Earnings Date"], errors="coerce").dropna()
            future = [d.date() for d in dates if d.date() >= datetime.today().date()]
            return min(future) if future else None

        # Some versions return a dict
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                d = pd.to_datetime(ed, errors="coerce")
                if hasattr(d, "date"):
                    return d.date()
                if hasattr(d, "__iter__"):
                    dates = [pd.to_datetime(x).date() for x in d
                             if pd.to_datetime(x).date() >= datetime.today().date()]
                    return min(dates) if dates else None

    except Exception:
        pass
    return None
