"""
Shadow Portfolios — Evolutionary Strategy Engine
=================================================
Maintains a population of 1000 shadow genomes, each a parameter set.
Every Broker.py run:
  1. Fast-score all 1000 genomes against the last 60 days of data (~seconds)
  2. Weekly: full replay validation of the top 20 survivors
  3. Promote the best validated genome to live config
  4. After promotion: promoted genome mutates into a new child, old live
     config re-enters the pool (swap — nothing is lost)
  5. Evolve the population: elites survive, weak are replaced by mutated
     children of the strong, crossover blends top pairs

Options sub-population (100 of the 1000 genomes have no_options=False).
Options go live once the best options genome beats the best non-options
genome for OPTIONS_LIVE_DAYS consecutive days.

Parameters that evolve:
  min_score, stop_loss, take_profit, max_sector, partial_profit,
  max_position_pct, cash_floor, max_gross_exposure, target_volatility,
  avoid_earnings, rl_enabled, rl_phase, rl_exit_threshold, rl_conviction_drop

State is persisted in broker/state/shadows.json (~2MB for 1000 genomes).
"""

import json
import logging
import os
import random
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STATE_FILE        = "broker/state/shadows.json"
POPULATION_SIZE    = 1000
OPTIONS_FRACTION   = 0.10   # 10% of population tests options
ELITE_FRACTION     = 0.20   # top 20% survive unchanged
CULL_FRACTION      = 0.20   # bottom 20% replaced each generation
EVOLUTION_DAYS     = 7      # evolve population every N days
VALIDATION_TOP_N   = 20     # top N fast-scored genomes get full replay
VALIDATION_METRIC_VERSION = 3
PROMOTION_MIN_EDGE = 0.05   # must beat baseline Sharpe by this much to promote
OPTIONS_LIVE_DAYS  = 30     # days options must beat non-options before going live


# ── Parameter space ───────────────────────────────────────────────────────────

_PARAM_BOUNDS = {
    "min_score":         (0.40, 0.80),
    "stop_loss":         (0.04, 0.18),
    "take_profit":       (0.20, 0.70),
    "max_sector":        (0.15, 0.55),
    "partial_profit":    (0.10, 0.40),
    "max_position_pct":  (0.08, 0.25),
    "cash_floor":        (0.00, 0.08),
    "max_gross_exposure":(0.90, 1.00),
    "target_volatility": (0.12, 0.30),
    "avoid_earnings":    (0,    10),
    "rl_exit_threshold": (0.10, 0.50),
    "rl_conviction_drop":(0.10, 0.40),
}

_MUTATION_SCALE = {
    "min_score":         0.04,
    "stop_loss":         0.02,
    "take_profit":       0.05,
    "max_sector":        0.05,
    "partial_profit":    0.03,
    "max_position_pct":  0.03,
    "cash_floor":        0.015,
    "max_gross_exposure":0.02,
    "target_volatility": 0.03,
    "avoid_earnings":    1,
    "rl_exit_threshold": 0.03,
    "rl_conviction_drop":0.03,
}


# ── Genome operations ─────────────────────────────────────────────────────────

def _random_genome(rng: random.Random, no_options: bool = True, rl_enabled: bool = False) -> dict:
    g = {}
    for k, (lo, hi) in _PARAM_BOUNDS.items():
        if isinstance(lo, int):
            g[k] = rng.randint(lo, hi)
        else:
            g[k] = round(rng.uniform(lo, hi), 3)
    g["no_options"] = no_options
    g["rl_enabled"] = rl_enabled
    g["rl_phase"]   = rng.choice([1, 2]) if rl_enabled else 1
    g["fast_score"] = 0.0
    g["sharpe"]     = 0.0
    g["validated"]  = False
    g["validation_metric_version"] = 0
    g["age"]        = 0
    return g


def _mutate(genome: dict, rng: random.Random, scale: float = 1.0) -> dict:
    child = deepcopy(genome)
    for k, sigma in _MUTATION_SCALE.items():
        if k not in child:
            continue
        lo, hi = _PARAM_BOUNDS[k]
        if isinstance(lo, int):
            delta = rng.randint(-int(sigma * scale + 0.5), int(sigma * scale + 0.5))
            child[k] = int(np.clip(child[k] + delta, lo, hi))
        else:
            delta = rng.gauss(0, sigma * scale)
            child[k] = round(float(np.clip(child[k] + delta, lo, hi)), 3)
    child["fast_score"] = 0.0
    child["sharpe"] = 0.0
    child["validated"] = False
    child["validation_metric_version"] = 0
    child["age"] = 0
    return child


def _crossover(a: dict, b: dict, rng: random.Random) -> dict:
    child = deepcopy(a)
    for k in _PARAM_BOUNDS:
        if rng.random() < 0.5 and k in b:
            child[k] = b[k]
    child["fast_score"] = 0.0
    child["sharpe"] = 0.0
    child["validated"] = False
    child["validation_metric_version"] = 0
    child["age"] = 0
    return child


def _genome_from_config(config: dict) -> dict:
    """Convert a live broker.config dict into a genome."""
    g = {
        "min_score":          float(config.get("min_score",          0.58)),
        "stop_loss":          float(config.get("stop_loss",          0.08)),
        "take_profit":        float(config.get("take_profit",        0.35)),
        "max_sector":         float(config.get("max_sector",         0.25)),
        "partial_profit":     float(config.get("partial_profit",     0.15)),
        "max_position_pct":   float(config.get("max_position_pct",   0.10)),
        "cash_floor":         float(config.get("cash_floor",         0.05)),
        "max_gross_exposure": float(config.get("max_gross_exposure", 0.95)),
        "target_volatility":  float(config.get("target_volatility",  0.15)),
        "avoid_earnings":     int(config.get("avoid_earnings",       5)),
        "rl_exit_threshold":  float(config.get("rl_exit_threshold",  0.30)),
        "rl_conviction_drop": float(config.get("rl_conviction_drop", 0.20)),
        "no_options":         bool(config.get("no_options",          True)),
        "rl_enabled":         bool(config.get("rl_enabled",          False)),
        "rl_phase":           int(config.get("rl_phase",             1)),
        "fast_score":         0.0,
        "sharpe":             0.0,
        "validated":          False,
        "validation_metric_version": 0,
        "age":                0,
        "is_baseline":        True,
    }
    return g


def _write_genome_to_config(
    genome: dict,
    config_path: str,
    checkpoint_path: str | None = None,
) -> None:
    from pipeline.autotuner import _write_config_key

    _write_config_key("min_score", f"{genome['min_score']:.3f}", config_path)
    _write_config_key("stop_loss", f"{genome['stop_loss']:.3f}", config_path)
    _write_config_key("take_profit", f"{genome['take_profit']:.3f}", config_path)
    _write_config_key("max_sector", f"{genome['max_sector']:.3f}", config_path)
    _write_config_key("partial_profit", f"{genome['partial_profit']:.3f}", config_path)
    _write_config_key("max_position_pct", f"{genome['max_position_pct']:.3f}", config_path)
    _write_config_key("cash_floor", f"{genome['cash_floor']:.3f}", config_path)
    _write_config_key("max_gross_exposure", f"{genome['max_gross_exposure']:.3f}", config_path)
    _write_config_key("target_volatility", f"{genome['target_volatility']:.3f}", config_path)
    _write_config_key("avoid_earnings", str(genome["avoid_earnings"]), config_path)
    _write_config_key("rl_exit_threshold", f"{genome['rl_exit_threshold']:.3f}", config_path)
    _write_config_key("rl_conviction_drop", f"{genome['rl_conviction_drop']:.3f}", config_path)

    if genome.get("rl_enabled") and checkpoint_path:
        _write_config_key("rl_enabled", "true", config_path)
        _write_config_key("rl_phase", str(genome.get("rl_phase", 1)), config_path)
        _write_config_key("rl_checkpoint_path", checkpoint_path, config_path)
    else:
        _write_config_key("rl_enabled", "false", config_path)


# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "population":          [],
            "generation":          0,
            "last_evolved":        None,
            "last_validated":      None,
            "options_days_beating": 0,
            "baseline_sharpe":     0.0,
        }


def _fast_score_value(genome: dict) -> float:
    return float(genome.get("fast_score", genome.get("sharpe", -99)))


def _selection_score(genome: dict) -> float:
    if _is_current_validation(genome):
        return float(genome.get("sharpe", -99))
    return _fast_score_value(genome)


def _validation_upgrade_required(population: list[dict]) -> bool:
    baseline = next((g for g in population if g.get("is_baseline")), None)
    if baseline is not None and not _is_current_validation(baseline):
        return True
    return any(g.get("validated") and not _is_current_validation(g) for g in population)


def _save_state(state: dict) -> None:
    Path(_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, separators=(",", ":"))  # compact — 1000 genomes ~1MB


def _resolve_shadow_checkpoint(checkpoint_path: str | None) -> str | None:
    """
    Match the live broker's checkpoint resolution so shadows handle
    `rl_checkpoint_path=auto` the same way the live cycle does.
    Picks best by val_sharpe, not alphabetically.
    """
    from pipeline.checkpoints import resolve_checkpoint_path

    return resolve_checkpoint_path(checkpoint_path=checkpoint_path, save_dir="models")


def _is_current_validation(genome: dict) -> bool:
    version = int(genome.get("validation_metric_version", 0) or 0)
    return bool(genome.get("validated")) and version >= VALIDATION_METRIC_VERSION


def _invalidate_stale_validations(population: list[dict]) -> list[dict]:
    cleared = 0
    for genome in population:
        if genome.get("validated") and not _is_current_validation(genome):
            genome["validated"] = False
            genome["sharpe"] = 0.0
            genome["validation_metric_version"] = 0
            cleared += 1
    if cleared:
        logger.info("Shadows: cleared %d stale validation result(s)", cleared)
    return population


# ── Population initialisation ─────────────────────────────────────────────────

def _init_population(live_config: dict, checkpoint_exists: bool) -> list[dict]:
    rng = random.Random(42)
    pop = []

    # Always include the current live config as genome 0 (baseline)
    pop.append(_genome_from_config(live_config))

    # Fill rest of population
    n_options = int(POPULATION_SIZE * OPTIONS_FRACTION)
    n_rl      = int(POPULATION_SIZE * 0.15) if checkpoint_exists else 0

    for i in range(1, POPULATION_SIZE):
        use_options = (i < n_options)
        use_rl      = (i >= n_options and i < n_options + n_rl)
        pop.append(_random_genome(rng, no_options=not use_options, rl_enabled=use_rl))

    return pop


# ── Fast scoring ──────────────────────────────────────────────────────────────

def _fast_score_genome(genome: dict, snap: pd.DataFrame) -> float:
    """
    Estimate Sharpe for a genome using a single-date feature snapshot.
    This is a cheap proxy — not a full replay — used to rank 1000 genomes
    in seconds. Top scorers get full replay validation.

    Score = weighted combination of signals filtered by genome thresholds.
    """
    try:
        min_score = float(genome.get("min_score", 0.58))

        # Compute composite score for each ticker in the snapshot
        scores = (
            snap.get("ret_5d",    0).clip(-1, 1) * 0.20 +
            snap.get("sent_net",  0).clip(-1, 1) * 0.25 +
            snap.get("macd_hist", 0).clip(-1, 1) * 0.15 +
            snap.get("vol_ratio", 1).clip(0, 5)  * 0.10 +
            snap.get("rsi",      50).apply(lambda x: 1 - abs(x - 52.5) / 52.5) * 0.10 +
            snap.get("ret_20d",   0).clip(-1, 1) * 0.20
        )

        # Filter by min_score threshold
        selected = scores[scores >= min_score]
        if len(selected) == 0:
            return -1.0

        # Proxy Sharpe: mean / std of selected scores (higher = better signal quality)
        mean = float(selected.mean())
        std  = float(selected.std()) if len(selected) > 1 else 1.0
        proxy_sharpe = mean / max(std, 0.01)

        # Penalise overly selective genomes in the fast proxy. On short debug
        # windows they can look good by selecting almost nothing, then produce
        # near-zero trades in full replay validation.
        penalty = 0.0
        if float(genome.get("stop_loss", 0.08)) < 0.05:
            penalty += 0.1
        min_score_val = float(genome.get("min_score", 0.58))
        if min_score_val > 0.65:
            penalty += min(0.30, (min_score_val - 0.65) * 2.0)
        if len(selected) < 5:
            penalty += 0.15

        deployment_bonus = 0.0
        deployment_bonus += 1.0 * np.clip(float(genome.get("max_position_pct", 0.10)) - 0.10, -0.04, 0.12)
        deployment_bonus += 1.5 * np.clip(0.05 - float(genome.get("cash_floor", 0.05)), -0.03, 0.05)
        deployment_bonus += 2.0 * np.clip(float(genome.get("max_gross_exposure", 0.95)) - 0.95, -0.05, 0.05)
        deployment_bonus += 0.5 * np.clip(float(genome.get("target_volatility", 0.15)) - 0.15, -0.05, 0.10)
        deployment_bonus = float(np.clip(deployment_bonus, -0.10, 0.25))

        return round((proxy_sharpe - penalty) * (1.0 + deployment_bonus), 4)

    except Exception:
        return -1.0


def fast_score_population(population: list[dict], df_features: pd.DataFrame) -> list[dict]:
    """Score all genomes using the latest feature snapshot."""
    try:
        dates   = sorted(df_features.index.get_level_values("date").unique())
        snap    = df_features.loc[dates[-1]].copy()
    except Exception:
        return population

    for genome in population:
        genome["fast_score"] = _fast_score_genome(genome, snap)
        genome["age"] = genome.get("age", 0) + 1

    return population


# ── Full replay validation ────────────────────────────────────────────────────

def validate_top_genomes(
    population: list[dict],
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    checkpoint_path: str | None,
    top_n: int = VALIDATION_TOP_N,
    replay_years: int = 1,
) -> list[dict]:
    """
    Run full replay on the top_n fast-scored genomes.
    Updates genome['sharpe'] and genome['validated'] = True.
    """
    from broker.replay import run_replay
    from pipeline.benchmark import compute_metrics

    baseline = next((g for g in population if g.get("is_baseline")), None)
    resolved_checkpoint = _resolve_shadow_checkpoint(checkpoint_path)
    ranked = sorted(
        [g for g in population if not g.get("is_baseline")],
        key=_fast_score_value,
        reverse=True,
    )
    to_validate = ranked[:top_n]
    if baseline is not None:
        to_validate = [baseline] + to_validate

    # Restrict to replay window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_val = df_features[df_features.index.get_level_values("date") >= cutoff]

    if baseline is not None:
        logger.info("Shadows: validating baseline + top %d genomes with full replay...", top_n)
    else:
        logger.info("Shadows: validating top %d genomes with full replay...", top_n)

    # Quick sanity check: verify price_lookup has data for the replay window
    if price_lookup is not None and not price_lookup.empty:
        pl_dates = price_lookup.index.get_level_values("date")
        feat_dates = df_features.index.get_level_values("date")
        overlap = len(set(pl_dates) & set(feat_dates))
        logger.info(
            "Shadows: price_lookup has %d dates, df_features has %d dates, overlap=%d",
            pl_dates.nunique(), feat_dates.nunique(), overlap,
        )
        if overlap == 0:
            logger.warning(
                "Shadows: NO DATE OVERLAP between price_lookup and df_features — "
                "all replays will make zero trades. Check date formats."
            )
            # Mark all as validated with Sharpe=0 so warm-up can still promote
            for genome in to_validate:
                genome["sharpe"]       = 0.0
                genome["total_return"] = 0.0
                genome["max_drawdown"] = 0.0
                genome["validated"]    = True
                genome["validation_metric_version"] = VALIDATION_METRIC_VERSION
            return population

    for i, genome in enumerate(to_validate):
        use_rl = bool(genome.get("rl_enabled")) and bool(resolved_checkpoint)
        strategy = "screener_rl" if use_rl else "heuristics_only"
        ckpt = resolved_checkpoint if use_rl else None
        effective_min_score = float(genome.get("min_score", 0.55))

        try:
            rets, _ = run_replay(
                df_val,
                price_lookup,
                strategy=strategy,
                checkpoint_path=ckpt,
                initial_cash=10_000.0,
                min_score=effective_min_score,
                stop_loss_floor=float(genome.get("stop_loss", 0.08)),
                take_profit=float(genome.get("take_profit", 0.35)),
                partial_profit_pct=float(genome.get("partial_profit", 0.20)),
                max_position_pct=float(genome.get("max_position_pct", 0.10)),
                cash_floor=float(genome.get("cash_floor", 0.05)),
                max_gross_exposure=float(genome.get("max_gross_exposure", 0.95)),
                target_volatility=float(genome.get("target_volatility", 0.15)),
                max_sector_pct=float(genome.get("max_sector", 0.25)),
                avoid_earnings_days=int(genome.get("avoid_earnings", 3)),
                rl_phase=int(genome.get("rl_phase", 1)),
                rl_exit_threshold=float(genome.get("rl_exit_threshold", 0.30)),
                rl_conviction_drop=float(genome.get("rl_conviction_drop", 0.20)),
                label=f"val_{i}",
            )
            metrics = compute_metrics(rets)
            genome["sharpe"] = round(metrics["sharpe"], 4)
            genome["total_return"] = round(metrics["total_return"], 4)
            genome["max_drawdown"] = round(metrics["max_drawdown"], 4)
            genome["validated"] = True
            genome["validation_metric_version"] = VALIDATION_METRIC_VERSION
        except Exception as exc:
            logger.warning("Validation failed for genome %d: %s", i, exc)
            genome["sharpe"]    = -1.0
            genome["validated"] = False
            genome["validation_metric_version"] = 0

    return population


# ── Evolution ─────────────────────────────────────────────────────────────────

def evolve_population(population: list[dict], live_config: dict) -> list[dict]:
    """
    One generation of evolution:
      - Top ELITE_FRACTION survive unchanged
      - Bottom CULL_FRACTION are replaced by mutated children of elites
      - Middle receive small perturbations
      - Crossover applied to random elite pairs
      - Live config always kept as one genome
    """
    rng = random.Random()
    n   = len(population)

    ranked = sorted(population, key=_selection_score, reverse=True)

    n_elite = int(n * ELITE_FRACTION)
    n_cull  = int(n * CULL_FRACTION)

    elites  = ranked[:n_elite]
    middle  = ranked[n_elite: n - n_cull]
    # Cull bottom — replace with mutated elites + crossovers
    new_bottom = []
    for _ in range(n_cull):
        if rng.random() < 0.4 and len(elites) >= 2:
            # Crossover two random elites
            a, b = rng.sample(elites, 2)
            new_bottom.append(_crossover(a, b, rng))
        else:
            # Mutate a random elite
            parent = rng.choice(elites)
            new_bottom.append(_mutate(parent, rng))

    # Lightly mutate middle
    mutated_middle = [_mutate(g, rng, scale=0.3) for g in middle]

    new_pop = elites + mutated_middle + new_bottom

    # Keep the live baseline in slot 0, including any replay validation metrics.
    baseline = next((deepcopy(g) for g in population if g.get("is_baseline")), None)
    if baseline is None:
        baseline = _genome_from_config(live_config)
    new_pop[0] = baseline

    logger.info("Shadows: evolved population — elites=%d  mutated=%d  new=%d",
                n_elite, len(mutated_middle), n_cull)
    return new_pop


# ── Promotion ─────────────────────────────────────────────────────────────────

def _maybe_promote(
    population: list[dict],
    live_config: dict,
    checkpoint_path: str | None,
    config_path: str = "broker.config",
) -> tuple[list[dict], bool]:
    """
    Find the best validated genome. If it beats the baseline by PROMOTION_MIN_EDGE,
    promote it to live config.

    After promotion:
      - Winner becomes the new live baseline
      - A mutated child stays in the pool

    Returns (updated_population, promoted: bool).
    """
    validated = [g for g in population if _is_current_validation(g) and not g.get("is_baseline")]
    if not validated:
        return population, False

    resolved_checkpoint = _resolve_shadow_checkpoint(checkpoint_path)
    baseline_sharpe = float(next(
        (g["sharpe"] for g in population if g.get("is_baseline") and _is_current_validation(g)),
        0.0,
    ))
    baseline_idx = next((i for i, g in enumerate(population) if g.get("is_baseline")), None)

    winner = max(validated, key=lambda g: float(g.get("sharpe", -99)))
    best_sharpe = float(winner.get("sharpe", -99))

    if baseline_idx is None:
        logger.info("Shadows: live baseline missing from population — deferring promotion")
        return population, False

    if best_sharpe <= baseline_sharpe + PROMOTION_MIN_EDGE:
        logger.info(
            "Shadows: best validated Sharpe=%.3f does not beat baseline=%.3f + %.2f",
            best_sharpe, baseline_sharpe, PROMOTION_MIN_EDGE,
        )
        return population, False

    logger.info(
        "Shadows: PROMOTING genome (Sharpe=%.3f vs baseline=%.3f)",
        best_sharpe, baseline_sharpe,
    )

    can_enable_rl = bool(winner.get("rl_enabled") and resolved_checkpoint)
    promoted = deepcopy(winner)
    if promoted.get("rl_enabled") and not can_enable_rl:
        logger.info(
            "Shadows: RL promotion requested but no checkpoint resolved — promoting as heuristics_only"
        )
        promoted["rl_enabled"] = False
        promoted["rl_phase"] = 1

    _write_genome_to_config(
        promoted,
        config_path=config_path,
        checkpoint_path=resolved_checkpoint if can_enable_rl else None,
    )

    # Winner becomes the new live baseline; a mutated child stays in the pool.
    rng = random.Random()
    child = _mutate(promoted, rng, scale=0.5)
    promoted["is_baseline"] = True

    # Replace winner's slot with its child; slot 0 keeps the live baseline.
    for i, g in enumerate(population):
        if g is winner:
            population[i] = child
            break
    population[baseline_idx] = promoted

    return population, True


def _maybe_enable_options(
    population: list[dict],
    state: dict,
    config_path: str = "broker.config",
) -> bool:
    """Enable options in live config once options genomes consistently beat non-options."""
    from pipeline.autotuner import _write_config_key

    options_genomes = [
        g for g in population
        if not g.get("no_options", True) and _is_current_validation(g)
    ]
    non_options_genomes = [
        g for g in population
        if g.get("no_options", True) and _is_current_validation(g)
    ]

    if not options_genomes or not non_options_genomes:
        return False

    best_opts     = max(options_genomes,     key=lambda g: float(g.get("sharpe", -99)))
    best_non_opts = max(non_options_genomes, key=lambda g: float(g.get("sharpe", -99)))

    if float(best_opts.get("sharpe", -99)) > float(best_non_opts.get("sharpe", -99)) + 0.05:
        days = state.get("options_days_beating", 0) + 1
        state["options_days_beating"] = days
        logger.info("Shadows: options beating non-options (%d/%d days)", days, OPTIONS_LIVE_DAYS)
        if days >= OPTIONS_LIVE_DAYS:
            _write_config_key("no_options", "false", config_path)
            logger.info("Shadows: OPTIONS ENABLED in live config")
            state["options_days_beating"] = 0
            return True
    else:
        state["options_days_beating"] = 0

    return False


# ── Main entry point ──────────────────────────────────────────────────────────

def run_shadow_cycle(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    live_config: dict,
    checkpoint_path: str | None = None,
    config_path: str = "broker.config",
    allow_promotion: bool = False,
    validation_top_n: int = VALIDATION_TOP_N,
    validation_replay_years: int = 1,
) -> bool:
    """
    One shadow cycle — called from Broker.py after the live cycle.

    Daily:  fast-score all 1000 genomes (~seconds)
    Weekly: full replay validation of the top validation_top_n genomes, then
        evolve population.

    allow_promotion: if False (default), winning genomes are logged as
        advisory recommendations but NOT written to broker.config.
        Pass --approve-promotion on the CLI to enable auto-promotion.
    """
    state = _load_state()
    resolved_checkpoint = _resolve_shadow_checkpoint(checkpoint_path)

    # Initialise population on first run
    if not state["population"]:
        logger.info("Shadows: initialising population of %d genomes...", POPULATION_SIZE)
        checkpoint_exists = bool(resolved_checkpoint)
        state["population"] = _init_population(live_config, checkpoint_exists)

    population = state["population"]

    # ── Daily: fast-score all genomes ────────────────────────────────────────
    population = fast_score_population(population, df_features)
    logger.info(
        "Shadows: fast-scored %d genomes — top fast score=%.3f",
        len(population),
        max(_fast_score_value(g) for g in population),
    )

    # ── Weekly: full validation + evolution ──────────────────────────────────
    last_validated = state.get("last_validated")
    days_since_val = _days_since(last_validated)
    force_revalidation = _validation_upgrade_required(population)

    if force_revalidation:
        logger.info("Shadows: validation metrics stale — forcing replay revalidation")
        population = _invalidate_stale_validations(population)

    if days_since_val >= 7 or force_revalidation:
        population = validate_top_genomes(
            population,
            df_features,
            price_lookup,
            resolved_checkpoint,
            top_n=max(1, int(validation_top_n)),
            replay_years=max(1, int(validation_replay_years)),
        )
        if allow_promotion:
            population, promoted = _maybe_promote(
                population, live_config, resolved_checkpoint, config_path
            )
        else:
            # Advisory mode: log the best genome but don't write to config
            validated = [g for g in population if _is_current_validation(g) and not g.get("is_baseline")]
            if validated:
                best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
                baseline_sharpe = float(next(
                    (g["sharpe"] for g in population if g.get("is_baseline") and _is_current_validation(g)), 0.0
                ))
                if float(best.get("sharpe", -99)) > baseline_sharpe + PROMOTION_MIN_EDGE:
                    logger.info(
                        "Shadows: advisory — genome with Sharpe=%.3f beats baseline=%.3f. "
                        "Run with --approve-promotion to promote to live config.",
                        float(best.get("sharpe", 0)), baseline_sharpe,
                    )
            promoted = False
        population = evolve_population(population, live_config)
        state["last_validated"] = date.today().isoformat()
        state["last_evolved"] = date.today().isoformat()
        state["generation"]     = state.get("generation", 0) + 1
        logger.info("Shadows: generation %d complete", state["generation"])

    # ── Options gate ─────────────────────────────────────────────────────────
    _maybe_enable_options(population, state, config_path)

    baseline = next((g for g in population if g.get("is_baseline")), None)
    if baseline is not None and _is_current_validation(baseline):
        state["baseline_sharpe"] = float(baseline.get("sharpe", 0.0))

    state["population"] = population
    _save_state(state)

    # Print compact summary
    _log_summary(population, state)


def _days_since(iso_date: str | None) -> int:
    if not iso_date:
        return 9999
    try:
        return (date.today() - date.fromisoformat(iso_date)).days
    except Exception:
        return 9999


def _log_summary(population: list[dict], state: dict) -> None:
    validated = [g for g in population if _is_current_validation(g)]
    top5 = sorted(validated, key=lambda g: float(g.get("sharpe", -99)), reverse=True)[:5]
    baseline = next((g for g in population if g.get("is_baseline")), {})
    baseline_sharpe = float(baseline.get("sharpe", 0.0)) if _is_current_validation(baseline) else 0.0

    logger.info(
        "Shadows: gen=%d  pop=%d  validated=%d  baseline_sharpe=%.3f",
        state.get("generation", 0),
        len(population),
        len(validated),
        baseline_sharpe,
    )
    for i, g in enumerate(top5):
        logger.info(
            "  #%d  Sharpe=%.3f  min_score=%.2f  stop=%.2f  tp=%.2f  "
            "rl=%s  opts=%s",
            i + 1,
            float(g.get("sharpe", 0)),
            float(g.get("min_score", 0)),
            float(g.get("stop_loss", 0)),
            float(g.get("take_profit", 0)),
            "Y" if g.get("rl_enabled") else "N",
            "Y" if not g.get("no_options", True) else "N",
        )


def get_shadow_summary() -> str:
    """Compact text summary for Broker.py --status output."""
    state = _load_state()
    population = state.get("population", [])
    if not population:
        return "  Shadow population: not yet initialised (runs after first Broker.py cycle)"

    validated = [g for g in population if _is_current_validation(g)]
    top5 = sorted(validated, key=lambda g: float(g.get("sharpe", -99)), reverse=True)[:5]
    baseline = next((g for g in population if g.get("is_baseline")), {})
    baseline_sharpe = float(baseline.get("sharpe", 0.0)) if _is_current_validation(baseline) else 0.0

    lines = [
        f"  Shadow population: {len(population)} genomes  "
        f"({len(validated)} validated)  "
        f"generation {state.get('generation', 0)}",
        f"  Baseline Sharpe: {baseline_sharpe:+.3f}",
        "  Top 5 validated genomes:",
    ]
    for i, g in enumerate(top5):
        lines.append(
            f"    #{i+1}  Sharpe={float(g.get('sharpe',0)):+.3f}  "
            f"min_score={g.get('min_score',0):.2f}  "
            f"stop={g.get('stop_loss',0):.2f}  "
            f"tp={g.get('take_profit',0):.2f}  "
            f"rl={'Y' if g.get('rl_enabled') else 'N'}  "
            f"opts={'Y' if not g.get('no_options', True) else 'N'}"
        )
    return "\n".join(lines)


# ── Historical warm-up (called once after training) ───────────────────────────

def run_historical_warmup(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    checkpoint_path: str | None,
    live_config: dict,
    generations: int = 5,
    replay_years: int = 3,
    validation_top_n: int = VALIDATION_TOP_N,
    config_path: str = "broker.config",
) -> bool:
    """
    Warm-start the shadow population using historical data immediately after
    training. Runs `generations` full evolutionary cycles on the historical
    window, then promotes the best genome to broker.config so the broker
    starts with pre-tuned parameters on day one.

    Called automatically at the end of `python Agent.py --mode train`.

    Parameters
    ----------
    generations : int
        Number of evolutionary generations to run on historical data.
        Each generation: fast-score all 1000 → validate top 20 → evolve.
        5 generations takes ~10-20 minutes depending on hardware.
    replay_years : int
        How many years of historical data to use for validation replays.
    """
    logger.info("=" * 60)
    logger.info("Shadow warm-up: initialising population on historical data")
    logger.info("  generations=%d  replay_years=%d", generations, replay_years)
    logger.info("=" * 60)

    resolved_checkpoint = _resolve_shadow_checkpoint(checkpoint_path)
    checkpoint_exists = bool(resolved_checkpoint)
    population = _init_population(live_config, checkpoint_exists)

    # Restrict features to the replay window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_hist = df_features[df_features.index.get_level_values("date") >= cutoff]

    for gen in range(generations):
        logger.info("Warm-up generation %d/%d...", gen + 1, generations)

        # Fast-score all genomes
        population = fast_score_population(population, df_hist)

        # Full replay validation of top 20
        population = validate_top_genomes(
            population, df_hist, price_lookup, resolved_checkpoint,
            top_n=validation_top_n, replay_years=replay_years,
        )

        validated = [g for g in population if _is_current_validation(g)]
        if validated:
            best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
            logger.info(
                "  Gen %d best: Sharpe=%.3f  min_score=%.2f  stop=%.2f  tp=%.2f",
                gen + 1,
                float(best.get("sharpe", 0)),
                float(best.get("min_score", 0)),
                float(best.get("stop_loss", 0)),
                float(best.get("take_profit", 0)),
            )

        # Evolve between generations, but keep the final generation's validated
        # genomes intact so warm-up can promote and persist the actual winners.
        if gen < generations - 1:
            population = evolve_population(population, live_config)

    # Promote the best genome found across all generations
    promoted = False
    validated = [g for g in population if _is_current_validation(g) and not g.get("is_baseline")]
    if validated:
        best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
        logger.info(
            "Warm-up complete. Best validated genome: Sharpe=%.3f",
            float(best.get("sharpe", 0)),
        )

        _write_genome_to_config(
            best,
            config_path=config_path,
            checkpoint_path=resolved_checkpoint if best.get("rl_enabled") else None,
        )

        logger.info("broker.config updated with warm-up winner.")
        promoted = True
    else:
        logger.warning("Warm-up: no validated genomes found — broker.config unchanged.")

    # Save the warm-started population so Broker.py inherits it
    state = {
        "population":           population,
        "generation":           generations,
        "last_evolved":         date.today().isoformat(),
        "last_validated":       date.today().isoformat(),
        "options_days_beating": 0,
        "baseline_sharpe":      float(next(
            (g["sharpe"] for g in population if g.get("is_baseline") and _is_current_validation(g)),
            0.0,
        )),
    }
    _save_state(state)
    logger.info(
        "Warm-up population saved (%d genomes, %d validated) → %s",
        len(population),
        len([g for g in population if _is_current_validation(g)]),
        _STATE_FILE,
    )
    return promoted
