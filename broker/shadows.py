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
PROMOTION_MIN_EDGE = 0.05   # must beat baseline Sharpe by this much to promote
OPTIONS_LIVE_DAYS  = 30     # days options must beat non-options before going live


# ── Parameter space ───────────────────────────────────────────────────────────

_PARAM_BOUNDS = {
    "min_score":         (0.40, 0.80),
    "stop_loss":         (0.04, 0.18),
    "take_profit":       (0.20, 0.70),
    "max_sector":        (0.15, 0.55),
    "partial_profit":    (0.10, 0.40),
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
    g["sharpe"]     = 0.0
    g["validated"]  = False
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
    child["sharpe"]    = 0.0
    child["validated"] = False
    child["age"]       = 0
    return child


def _crossover(a: dict, b: dict, rng: random.Random) -> dict:
    child = deepcopy(a)
    for k in _PARAM_BOUNDS:
        if rng.random() < 0.5 and k in b:
            child[k] = b[k]
    child["sharpe"]    = 0.0
    child["validated"] = False
    child["age"]       = 0
    return child


def _genome_from_config(config: dict) -> dict:
    """Convert a live broker.config dict into a genome."""
    g = {
        "min_score":          float(config.get("min_score",          0.58)),
        "stop_loss":          float(config.get("stop_loss",          0.08)),
        "take_profit":        float(config.get("take_profit",        0.35)),
        "max_sector":         float(config.get("max_sector",         0.25)),
        "partial_profit":     float(config.get("partial_profit",     0.15)),
        "avoid_earnings":     int(config.get("avoid_earnings",       5)),
        "rl_exit_threshold":  float(config.get("rl_exit_threshold",  0.30)),
        "rl_conviction_drop": float(config.get("rl_conviction_drop", 0.20)),
        "no_options":         bool(config.get("no_options",          True)),
        "rl_enabled":         bool(config.get("rl_enabled",          False)),
        "rl_phase":           int(config.get("rl_phase",             1)),
        "sharpe":             0.0,
        "validated":          False,
        "age":                0,
        "is_baseline":        True,
    }
    return g


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


def _save_state(state: dict) -> None:
    Path(_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, separators=(",", ":"))  # compact — 1000 genomes ~1MB


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

        # Penalise extreme parameters (overfitting risk)
        penalty = 0.0
        if float(genome.get("stop_loss", 0.08)) < 0.05:
            penalty += 0.1
        if float(genome.get("min_score", 0.58)) > 0.75:
            penalty += 0.1

        return round(proxy_sharpe - penalty, 4)

    except Exception:
        return -1.0


def fast_score_population(population: list[dict], df_features: pd.DataFrame) -> list[dict]:
    """Score all genomes using the latest feature snapshot. Updates genome['sharpe'] in-place."""
    try:
        dates   = sorted(df_features.index.get_level_values("date").unique())
        snap    = df_features.loc[dates[-1]].copy()
    except Exception:
        return population

    for genome in population:
        genome["sharpe"] = _fast_score_genome(genome, snap)
        genome["age"]    = genome.get("age", 0) + 1

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

    # Sort by fast score, take top_n
    ranked = sorted(population, key=lambda g: float(g.get("sharpe", -99)), reverse=True)
    to_validate = ranked[:top_n]

    # Restrict to replay window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_val = df_features[df_features.index.get_level_values("date") >= cutoff]

    logger.info("Shadows: validating top %d genomes with full replay...", top_n)

    for i, genome in enumerate(to_validate):
        rl_enabled = genome.get("rl_enabled", False)
        ckpt = checkpoint_path if (rl_enabled and checkpoint_path and os.path.exists(checkpoint_path)) else None
        strategy = "screener_rl" if ckpt else "heuristics_only"

        try:
            rets, _ = run_replay(
                df_val,
                price_lookup,
                strategy=strategy,
                checkpoint_path=ckpt,
                initial_cash=10_000.0,
                min_score=float(genome.get("min_score", 0.58)),
                stop_loss_floor=float(genome.get("stop_loss", 0.08)),
                take_profit=float(genome.get("take_profit", 0.35)),
                max_sector_pct=float(genome.get("max_sector", 0.25)),
                label=f"val_{i}",
            )
            m = genome["sharpe"] = round(compute_metrics(rets)["sharpe"], 4)
            genome["total_return"] = round(compute_metrics(rets)["total_return"], 4)
            genome["max_drawdown"] = round(compute_metrics(rets)["max_drawdown"], 4)
            genome["validated"]    = True
        except Exception as exc:
            logger.debug("Validation failed for genome %d: %s", i, exc)
            genome["sharpe"]    = -1.0
            genome["validated"] = False

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

    ranked = sorted(population, key=lambda g: float(g.get("sharpe", -99)), reverse=True)

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

    # Always ensure live config is in the population
    baseline = _genome_from_config(live_config)
    new_pop[0] = baseline  # slot 0 is always the live baseline

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
      - Winner gets mutated into a new child (continues evolving)
      - Old live config re-enters pool as a new genome (swap)

    Returns (updated_population, promoted: bool).
    """
    from pipeline.autotuner import _write_config_key

    validated = [g for g in population if g.get("validated") and not g.get("is_baseline")]
    if not validated:
        return population, False

    baseline_sharpe = float(next(
        (g["sharpe"] for g in population if g.get("is_baseline")), 0.0
    ))

    best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
    best_sharpe = float(best.get("sharpe", -99))

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

    # Write winner to config
    _write_config_key("min_score",          f"{best['min_score']:.3f}",          config_path)
    _write_config_key("stop_loss",          f"{best['stop_loss']:.3f}",          config_path)
    _write_config_key("take_profit",        f"{best['take_profit']:.3f}",        config_path)
    _write_config_key("max_sector",         f"{best['max_sector']:.3f}",         config_path)
    _write_config_key("partial_profit",     f"{best['partial_profit']:.3f}",     config_path)
    _write_config_key("avoid_earnings",     str(best["avoid_earnings"]),          config_path)
    _write_config_key("rl_exit_threshold",  f"{best['rl_exit_threshold']:.3f}",  config_path)
    _write_config_key("rl_conviction_drop", f"{best['rl_conviction_drop']:.3f}", config_path)
    if best.get("rl_enabled"):
        _write_config_key("rl_enabled", "true",              config_path)
        _write_config_key("rl_phase",   str(best["rl_phase"]), config_path)
        if checkpoint_path:
            _write_config_key("rl_checkpoint_path", checkpoint_path, config_path)
    else:
        _write_config_key("rl_enabled", "false", config_path)

    # Swap: winner mutates into a child, old live config re-enters pool
    rng = random.Random()
    child = _mutate(best, rng, scale=0.5)
    old_baseline = _genome_from_config(live_config)

    # Replace winner's slot with its child; add old baseline back
    for i, g in enumerate(population):
        if g is best:
            population[i] = child
            break
    population[0] = old_baseline  # slot 0 always holds the current live baseline

    return population, True


def _maybe_enable_options(
    population: list[dict],
    state: dict,
    config_path: str = "broker.config",
) -> bool:
    """Enable options in live config once options genomes consistently beat non-options."""
    from pipeline.autotuner import _write_config_key

    options_genomes    = [g for g in population if not g.get("no_options", True) and g.get("validated")]
    non_options_genomes = [g for g in population if g.get("no_options", True)  and g.get("validated")]

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
) -> None:
    """
    One shadow cycle — called from Broker.py after the live cycle.

    Daily:  fast-score all 1000 genomes (~seconds)
    Weekly: full replay validation of top 20, then evolve population
    """
    state = _load_state()

    # Initialise population on first run
    if not state["population"]:
        logger.info("Shadows: initialising population of %d genomes...", POPULATION_SIZE)
        checkpoint_exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
        state["population"] = _init_population(live_config, checkpoint_exists)

    population = state["population"]

    # ── Daily: fast-score all genomes ────────────────────────────────────────
    population = fast_score_population(population, df_features)
    logger.info(
        "Shadows: fast-scored %d genomes — top Sharpe=%.3f",
        len(population),
        max(float(g.get("sharpe", -99)) for g in population),
    )

    # ── Weekly: full validation + evolution ──────────────────────────────────
    last_validated = state.get("last_validated")
    days_since_val = _days_since(last_validated)

    if days_since_val >= 7:
        population = validate_top_genomes(
            population, df_features, price_lookup, checkpoint_path
        )
        population, promoted = _maybe_promote(
            population, live_config, checkpoint_path, config_path
        )
        population = evolve_population(population, live_config)
        state["last_validated"] = date.today().isoformat()
        state["generation"]     = state.get("generation", 0) + 1
        logger.info("Shadows: generation %d complete", state["generation"])

    # ── Options gate ─────────────────────────────────────────────────────────
    _maybe_enable_options(population, state, config_path)

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
    validated = [g for g in population if g.get("validated")]
    top5 = sorted(validated, key=lambda g: float(g.get("sharpe", -99)), reverse=True)[:5]
    baseline = next((g for g in population if g.get("is_baseline")), {})

    logger.info(
        "Shadows: gen=%d  pop=%d  validated=%d  baseline_sharpe=%.3f",
        state.get("generation", 0),
        len(population),
        len(validated),
        float(baseline.get("sharpe", 0)),
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

    validated = [g for g in population if g.get("validated")]
    top5 = sorted(validated, key=lambda g: float(g.get("sharpe", -99)), reverse=True)[:5]
    baseline = next((g for g in population if g.get("is_baseline")), {})

    lines = [
        f"  Shadow population: {len(population)} genomes  "
        f"({len(validated)} validated)  "
        f"generation {state.get('generation', 0)}",
        f"  Baseline Sharpe: {float(baseline.get('sharpe', 0)):+.3f}",
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
    config_path: str = "broker.config",
) -> None:
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

    checkpoint_exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
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
            population, df_hist, price_lookup, checkpoint_path,
            top_n=VALIDATION_TOP_N, replay_years=replay_years,
        )

        # Evolve
        population = evolve_population(population, live_config)

        validated = [g for g in population if g.get("validated")]
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

    # Promote the best genome found across all generations
    validated = [g for g in population if g.get("validated") and not g.get("is_baseline")]
    if validated:
        best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
        baseline_sharpe = float(next(
            (g["sharpe"] for g in population if g.get("is_baseline")), 0.0
        ))

        logger.info(
            "Warm-up complete. Best genome: Sharpe=%.3f (baseline=%.3f)",
            float(best.get("sharpe", 0)), baseline_sharpe,
        )

        # Always promote after warm-up — this is the best we found on history
        from pipeline.autotuner import _write_config_key
        _write_config_key("min_score",          f"{best['min_score']:.3f}",          config_path)
        _write_config_key("stop_loss",          f"{best['stop_loss']:.3f}",          config_path)
        _write_config_key("take_profit",        f"{best['take_profit']:.3f}",        config_path)
        _write_config_key("max_sector",         f"{best['max_sector']:.3f}",         config_path)
        _write_config_key("partial_profit",     f"{best['partial_profit']:.3f}",     config_path)
        _write_config_key("avoid_earnings",     str(best["avoid_earnings"]),          config_path)
        _write_config_key("rl_exit_threshold",  f"{best['rl_exit_threshold']:.3f}",  config_path)
        _write_config_key("rl_conviction_drop", f"{best['rl_conviction_drop']:.3f}", config_path)
        if best.get("rl_enabled") and checkpoint_path:
            _write_config_key("rl_enabled",         "true",          config_path)
            _write_config_key("rl_phase",            str(best.get("rl_phase", 1)), config_path)
            _write_config_key("rl_checkpoint_path",  checkpoint_path, config_path)
        else:
            _write_config_key("rl_enabled", "false", config_path)

        logger.info("broker.config updated with warm-up winner.")
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
            (g["sharpe"] for g in population if g.get("is_baseline")), 0.0
        )),
    }
    _save_state(state)
    logger.info(
        "Warm-up population saved (%d genomes, %d validated) → %s",
        len(population),
        len([g for g in population if g.get("validated")]),
        _STATE_FILE,
    )
