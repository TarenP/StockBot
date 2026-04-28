# StockBot — AI Stock Predictor & Autonomous Broker

---

## Agent vs Broker — what's the difference?

Think of it like this:

**Agent (`Agent.py`) = the researcher and trainer.**
It studies the market, learns patterns from 60+ years of historical data,
and tells you what stocks look promising. It does not manage money. It does
not buy or sell anything. It's the brain that gets trained.

**Broker (`Broker.py`) = your daily research assistant.**
It uses what the Agent learned, plus live data and news, to analyze the market
and tell you exactly what to do with your portfolio. Every morning it produces
a **Daily Briefing** — a plain-English action list: what to buy, what to sell,
and why. You execute those trades manually in your own brokerage account.

It also maintains a **paper portfolio** that tracks exactly what it would have
done, so you can compare your real results to its recommendations over time.

In practice:
- You train the Agent once (takes hours), then occasionally retrain it
- You run the Broker every day (takes minutes), and it handles everything

Most of the time you only need one command:
```bash
python Broker.py
```

---

## How it works

Every stock is scored using 29 features computed daily:

**Technical signals (10):** returns over 1/5/20 days, RSI, MACD histogram,
Bollinger Band position, ATR volatility, volume ratio, volume z-score,
52-week price position.

**Sentiment signals (9):** all derived from FinBERT-scored news headlines —
net sentiment score, 3/7/14-day rolling averages, sentiment surprise (today
vs 14-day baseline), sentiment acceleration, 7-day trend slope, raw positive
confidence, negativity spike detection.

**Market context (3):** SPY 20-day return, VIX level, market breadth (fraction
of stocks above 200-day moving average).

**Fundamental signals (3):** trailing P/E ratio, revenue growth, short interest
percentage. Fetched from yfinance with a 24-hour cache.

**Regime features (4):** one-hot encoded market regime (calm bull / trending /
choppy / risk-off) derived from SPY realised volatility and cross-sectional
return dispersion.

All features are cross-sectionally z-scored against the full universe each
day, so a value of +1.5 means that stock is 1.5 standard deviations above
the average stock on that signal.

### Two-stage ML pipeline

```
11,500+ tickers
    ↓  Stage 1: TickerScorer (screener)
~50-100 shortlist
    ↓  Stage 2: PortfolioTransformer (RL agent)
Final ranked candidates → broker execution
```

**Stage 1 — Screener (`pipeline/screener.py`):**
A bidirectional GRU with attention pooling trained to identify stocks likely
to be in the top decile of 20-day forward returns. Trained on 60+ years of
cross-sectionally normalised features across all 7,000+ tickers. Cuts the
universe from 11,500 to ~50-100 candidates before the RL agent sees anything.

Key design choices: labels are built from raw forward returns computed from
raw `close` prices, not from already-normalised features; the model consumes
the engineered feature window directly without a second normalisation pass;
sample validity is based on real history coverage rather than `!= 0` checks;
and checkpoints are selected on shortlist quality (`precision@k`,
`recall@k`, lift, and mean forward return at `k`) rather than AUC alone.
That keeps the screener aligned with its real job in the bot: narrowing the
universe to the best broker shortlist, not just maximizing classifier metrics.

**Stage 2 — RL agent (`pipeline/model.py`):**
A PortfolioTransformer trained via PPO walk-forward to allocate weights across
the screener's shortlist. Trained cross-sectionally — it sees all shortlist
tickers simultaneously and learns relative conviction, not absolute scores.

---

## Setup

```bash
pip install -r requirements.txt
```

FinBERT (~500MB) downloads automatically the first time sentiment scoring runs.

---

## First time setup

```bash
# Step 1: Train the model (takes 2-6 hours on CPU, 30-60 min on GPU)
python Agent.py --mode train --folds 10

# Step 2: Set your starting cash in broker.config (only thing you need to edit)
# cash = 10000

# Step 3: Run the broker
python Broker.py
```

After step 3, run `python Broker.py` every morning. The broker handles price updates, sentiment, and trade decisions automatically. Model finetuning, parameter optimisation, and RL mode switching also run on a schedule — but they depend on data quality. If data sources are degraded, new entries are blocked and the manifest records why. Check `logs/broker.log` after each run to see what happened.

---

## broker.config — Your persistent settings

Edit `broker.config` once. These become your defaults every run — no flags needed.

```
# Universe — broad tradable U.S. equities, SPY as benchmark only
universe_mode  = tradable_us  # broad tradable U.S. equity universe
                               # options: tradable_us, sp500, sp1500, custom
live_target_size = 1500        # breadth to maintain in the price/news universe
allow_etfs     = false         # ETFs excluded from tradable names; SPY is benchmark only
allow_otc      = false
allow_preferreds = false
allow_warrants = false
benchmark_symbols = SPY        # SPY stays in parquet for benchmark tracking
freeze_universe_snapshot = false  # set true for reproducible replay
universe_snapshot_path = plots/live_universe_snapshot.json
min_broad_universe_size = 1000    # fail loudly if universe resolves below this

# Freshness gates — block new entries when data quality degrades
min_fresh_price_coverage    = 0.90   # require 90% of universe to have today's prices
min_fresh_sentiment_coverage = 0.50  # require 50% sentiment coverage on latest slice

# Portfolio
cash           = 10000     # starting cash (first run only — ignored after that)
max_positions  = 20        # more positions = more capital deployed
stop_loss      = 0.115     # ATR-adjusted per stock; 8% floor avoids noise clips
take_profit    = 1.000     # disable hard cap; trailing exits manage the rest
partial_profit = 0.350     # trim at +35%, let leaders compound
trailing_stop  = 0.120     # exit a winner after it gives back 12% from peak
trailing_activation = 0.180 # start trailing once position has 18% cushion
min_score      = 0.58      # heuristic entry floor (RL path uses rl_min_score)
penny_pct      = 0.03      # minimal speculative exposure
max_sector     = 0.400     # hard cap per sector
max_correlation = 0.80     # blocks adding names too correlated with held positions
avoid_earnings = 4         # skip stocks within N days of earnings
max_daily_loss = 0.025     # 2.5% daily halt
max_drawdown   = 0.12      # 12% circuit breaker
execution_spread = 0.001   # base paper execution cost; low-price tiers can be higher
no_options     = true      # disabled until stock-side edge is proven

# RL integration
rl_enabled            = true
rl_checkpoint_path    = auto   # auto = best checkpoint by val_sharpe
rl_phase              = 1      # 1=ranking, 2=ranking+exits
rl_exit_threshold     = 0.150  # Phase 2: sell if rank_pct drops below this
rl_conviction_drop    = 0.250  # Phase 2: sell 50% if rank_pct drops by this much
rl_min_score          = 0.0    # Phase 1: min rl_score to enter (0 = top-k only)
```

**Important operational notes:**
- `universe_mode = tradable_us` (the default) builds a broad investable U.S. equity universe from live sources. ETFs, OTC names, preferreds, and warrants are excluded. SPY is retained in the parquet as a benchmark reference only — it is never a tradable candidate.
- `universe_mode = sp500` is available as a stricter alternative that enforces exact S&P 500 membership. Use it if you want index-constrained behavior.
- New entries are blocked automatically when `fresh_price_coverage` or `fresh_sentiment_coverage` fall below their thresholds — the broker continues exits and holds but makes no new buys. The manifest records exactly which gate triggered and why.
- If the benchmark (SPY) is unavailable, relative metrics (alpha, beta, information ratio) are omitted from the report but trading continues normally.
- `rl_checkpoint_path = auto` always picks the checkpoint with the highest validation Sharpe, not the most recently trained one.
- Set `freeze_universe_snapshot = true` for reproducible replay — the universe is frozen to a dated snapshot file so re-running the same replay always uses the same ticker set.

### Degraded-mode behavior

| Condition | New entries | Exits | Status run |
|---|---|---|---|
| Price coverage < threshold | Blocked | Allowed | Allowed |
| Sentiment coverage < threshold | Blocked | Allowed | Allowed |
| Held ticker missing from fresh slice | Blocked | Allowed | Allowed |
| Benchmark (SPY) unavailable | Allowed | Allowed | Allowed (no relative metrics) |
| Universe below min_broad_universe_size | Error — run aborts | — | — |
| Risk halt (daily loss / drawdown) | Blocked | Allowed | Allowed |

Each blocked condition is named in the manifest and printed to the console.

---

## Broker — Running and monitoring

### Run a cycle
```bash
python Broker.py
```

Every run produces a **Daily Briefing** at the end — a plain-English action list:

```
================================================================
  DAILY BRIEFING  —  Monday, March 29 2026
================================================================

  TODAY'S ACTIONS
  --------------------------------------------------------------
  SELL  NVDA    12.54 shares @ $167.52
        Reason: Full take-profit (+57.3%)
  BUY   MSFT    8.21 shares @ $412.30
        Reason: Score=0.74 | Strong momentum + positive sentiment

  CURRENT POSITIONS
  --------------------------------------------------------------
  Ticker   Shares    Price      Value      P&L
  AAPL      12.54  $199.34   $2,500      +0.0%
  LLY        2.85  $878.24   $2,500      +0.0%
  MSFT       8.21  $412.30   $3,385      +0.0%

  PAPER PORTFOLIO PERFORMANCE
  --------------------------------------------------------------
  Equity:       $12,450.00
  Cash:          $2,065.00  (17% of portfolio)
  Total return:     +24.5%
  SPY return:       +16.3%  (same period)
  Alpha vs SPY:      +8.2%  (beats SPY: YES)

  PARAMETER RECOMMENDATION (advisory)
  --------------------------------------------------------------
  Sharpe 1.24 vs baseline 1.01 | stop_loss -> 0.12, take_profit -> 0.65
  run with --approve-promotion to apply
================================================================
```

You execute the BUY/SELL actions manually in your own brokerage account.
The paper portfolio tracks what the system would have done so you can
compare your real results to its recommendations over time.

### Check portfolio without trading
```bash
python Broker.py --status
# or
python Broker.py --snapshot
# refresh the broader local price cache too, still no trading
python Broker.py --status --refresh-prices
```
Shows current holdings, cash, EOD return, drawdown, SPY benchmark, concentration,
paper execution drag, cap-history status, replay/live parity, and current shadow
portfolio standings. This command fetches the latest price for each current
holding, validates the marks, credits known cash dividends for held shares,
updates `broker/state/portfolio.json`, records a no-trade equity snapshot, and
prints the refreshed status. It does not buy or sell. Add `--refresh-prices`
when you also want it to update the broader local price cache before printing.

### See trade history
```bash
python Broker.py --trades
```

---

## How the broker makes decisions

### Price validation
Any price move greater than 30% in a single day is cross-checked across
three sources (yfinance re-fetch, Finviz quote, news headlines) before being
accepted. If 2 of 3 sources confirm the move it's real. Otherwise the
previous price is used and a warning is logged.

### Exit logic (checked every run on every position)
- **Stop-loss:** ATR-adjusted per stock. A volatile stock gets a wider stop
  than a stable one. Floor set by `stop_loss` in config (default 7%).
  Ceiling is 25% regardless of volatility.
- **Partial take-profit:** At the `partial_profit` threshold (default +20%),
  sells 50% of the position and locks in profit. The remaining 50% runs on.
- **Full take-profit:** At the `take_profit` threshold (default +45%),
  exits the remaining shares entirely. The company is fully exited.
- **Signal deterioration:** If the composite score of a held stock drops
  below 0.35, the position is sold regardless of P&L.

To exit everything in one shot instead of two stages, set
`partial_profit` higher than `take_profit` in `broker.config`.

### Risk engine (checked every run)
- **Daily loss limit:** If the portfolio is down more than `max_daily_loss`
  (default 3%) in the current session, no new positions are opened.
- **Drawdown circuit breaker:** If the portfolio is down more than
  `max_drawdown` (default 15%) from its peak equity, no new positions
  are opened until it recovers.
- **Cash floor:** Always keeps 5% of equity in cash as a buffer.
- **Volatility scaling:** When recent portfolio volatility is elevated,
  position sizes are automatically scaled down.

### Sector allocation
The broker scores all 11 GICS sectors each run based on average momentum,
sentiment, breadth (% of stocks up), and volume surge. It converts these
scores into target allocations using a softmax with a quadratic
diversification penalty — the more concentrated you already are in a sector,
the harder it becomes to add more. The `max_sector` config sets a hard cap.

### Theme and price-bucket controls
Sector labels are not the only concentration check. The broker also tags
candidates with a `theme_bucket` such as `consumer_credit_finance`,
`precious_metals_miners`, or `speculative_growth_turnaround`. `theme_max_pct`
caps any one economic theme, and `low_price_max_pct` caps aggregate sub-$10
exposure. The score audit records pre-cap, post-cap, and final weights so
rank-to-weight inversions can be traced after each rebalance. Each cycle also
logs aggregate pre-cap and final theme/sector exposure, effective theme and
position bet counts, top-name/top-theme concentration, cap-impact totals and
counts by class, the largest cap interventions, and the largest rank-to-weight
mismatches.

### Paper diagnostics
Each completed paper cycle appends `broker/state/portfolio_history.jsonl` with
cash, holdings, top-name concentration, theme exposure, low-price exposure,
execution drag, and the latest allocation summary. It also writes:

- `broker/state/cap_impact_summary.json` - rolling cap-impact totals and counts
- `broker/state/performance_attribution.json` - realized/unrealized P&L and execution drag
- `broker/state/replay_live_parity.json` - live settings compared with replay settings

Paper fills charge an explicit spread/slippage cost in cash and record it on
each trade. The trade log keeps both decision price and execution cost so P&L
does not silently assume free execution.

Sentiment policy is explicit in `broker.config`. `sentiment_policy =
informational` means sentiment is logged and available to the scorer, but it
is not a standalone buy veto. `penalize_negative` haircuts negative-sentiment
position sizes by `sentiment_negative_weight_mult`, while `veto_negative`
blocks negative-sentiment entries whose composite score is below
`sentiment_veto_composite_floor`.

### Candidate selection and sizing
1. Screens 1000 stocks using the trained screener (or rule-based fallback)
2. Skips any stock within `avoid_earnings` days of earnings
3. Deep-researches top candidates: fetches live OHLCV + FinBERT news
4. Scores each on 19 signals → composite score 0–1
5. Skips anything below `min_score`
6. Sizes position by conviction: higher score = larger allocation
7. Applies sector, theme, low-price, penny, cash-floor, and volatility controls
8. Applies execution cost estimate (spread model) before buying

### RL integration (opt-in)

Set `rl_enabled = true` in `broker.config` (or pass `--rl_enabled`) to
activate the RL model for live decisions. The broker operates in hard
model-required mode — if the checkpoint is missing or fails to load, the
cycle aborts rather than silently falling back to heuristics.

**Checkpoint contract:** on startup the broker verifies the checkpoint file
exists, loads without error, and contains the required `model_cfg` and
`model_state` keys. It does not check `n_assets` against the shortlist size
— the inference wrapper builds a dynamic observation tensor sized to the
current shortlist each cycle, so the model works with any shortlist length.

**Phase 1 — RL as ranking controller (default when enabled):**
The screener narrows the universe to a shortlist. The RL model scores the
entire shortlist in one cross-sectional forward pass and ranks candidates by
conviction. `rl_min_score` (default `0.0`) sets a floor on RL scores — at
the default of 0 the broker simply takes the top-k by RL rank with no
absolute cutoff. Set it to e.g. `0.05` if you want to filter out the
bottom of the distribution. All existing risk controls (sector cap, penny
cap, ATR stop, cash floor) still apply after RL ranking. Options are
suppressed while RL is active.

**Phase 2 — RL conviction-drop exits (`rl_phase = 2`):**
After the shortlist is scored, held positions that appear in the cycle's RL
scores are checked for exit signals. Scoring uses the same cross-sectional
pass as entry ranking — held names are evaluated in the live opportunity set,
not in isolation. If the current RL score drops below `rl_exit_threshold`
(default 0.30), the position is sold. If it drops by more than
`rl_conviction_drop` (default 0.20) relative to the score at entry, 50% of
the position is sold. Held names that fell off the shortlist entirely are
skipped and handled by heuristic exits instead.

```bash
# Enable RL for one run
python Broker.py --rl_enabled --rl_checkpoint_path models/best_fold9.pt

# Enable Phase 2 exits, with a score floor for entries
python Broker.py --rl_enabled --rl_phase 2 --rl_min_score 0.05
```

Run the ablation study before enabling RL in production to verify it
actually improves on the heuristic baseline.

### Options trading
After stock decisions, evaluates top-scored stocks for options:
|-----------------|----------|-----|
| Score > 0.75 + strong positive sentiment | Long Call | Leveraged bullish bet |
| Score > 0.65 + mild positive sentiment | Bull Call Spread | Cheaper bullish, defined risk |
| Score < 0.35 + negative sentiment | Long Put | Bearish hedge |
| Score > 0.60, neutral | Cash-Secured Put | Collect premium, buy stock cheaper if assigned |

Options are capped at 10% of equity. All strategies are defined-risk —
maximum loss is always the premium paid. Cash-secured puts fully reserve
the strike price in cash. Assignment and exercise correctly update stock
positions and cash. Positions auto-close when P&L reaches 50% of max profit.

Set `no_options = true` in `broker.config` to disable.

---

## Agent — Research and training commands

You only need these occasionally. The broker runs daily on its own.

### Get weekly stock picks
```bash
python Agent.py --mode predict
```
Outputs a ranked table of top picks with portfolio weights and signal
breakdown. Useful for a manual sanity check on what the model thinks.

```bash
python Agent.py --mode predict --top_k 20      # show top 20 instead of 10
```

### Screen all stocks including penny stocks
```bash
# Train the screener first (one time only)
python Agent.py --mode train_screener

# Scan all 11,500+ tickers
python Agent.py --mode screen

# Penny stocks only (under $5)
python Agent.py --mode screen --penny

# Custom price range
python Agent.py --mode screen --min_price 1 --max_price 10 --screener_top_n 100
```

The screener has no universe size limit — it scores every ticker
individually including sub-$5 stocks. In live screening and screener training,
raw `close`/`volume` columns are loaded alongside engineered features so
price, liquidity, and history-coverage filters are enforced consistently.

### Train the RL model from scratch
```bash
python Agent.py --mode train --folds 10
```

Walk-forward training: each fold trains on 8 years of data, validates on 1
year, and tests on 1 year. No data leakage between folds. The best
checkpoint per fold is saved to `models/`.

If you stop mid-training (Ctrl+C), progress is saved every 5,000 steps.
Re-run the exact same command to resume — completed folds are skipped.

```bash
python Agent.py --mode train --folds 10 --total_steps 200000   # longer training
python Agent.py --mode train --folds 5  --top_n 500            # faster, smaller universe
```

Training time: ~2–6 hours on CPU per fold, ~30–60 min on GPU.
Check if you have a GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Fine-tune on recent data
```bash
python Agent.py --mode finetune
```
Takes the best existing checkpoint and continues training on the most recent
2 years of data. Run this after major market regime changes. Takes ~20 min.

### Backtest the RL agent vs SPY
```bash
python Agent.py --mode backtest
```
Evaluates the trained RL policy on the held-out test period. SPY is fetched
automatically. Saves a 4-panel chart to `plots/backtest.png`.

### Broker replay backtest
```bash
python Agent.py --mode replay
```
Runs the broker over historical data using the same decision engine the live
system uses — same screening, research, sector allocation, stop/take-profit
logic, risk-engine checks, and execution-cost-adjusted fills.

**Replay is deterministic by default.** Every replay run automatically freezes
the universe to a dated snapshot file (`plots/live_universe_snapshot.json`)
unless `freeze_universe_snapshot = true` is already set in `broker.config`.
Re-running the same replay with the same snapshot and checkpoint produces the
same trade log. The snapshot path is recorded in the manifest.

```bash
python Agent.py --mode replay --replay_years 5     # use 5 years of history
python Agent.py --mode replay --sensitivity        # also run sensitivity sweep
```

**Friction sensitivity** is printed after every replay — three execution-cost
regimes so you can see whether results depend on cost assumptions:

```
========================================================================
  Friction Sensitivity (optimistic / base / stressed execution cost)
========================================================================
  Regime       Spread    Return    Sharpe     MaxDD
  ------------------------------------------------------------
  optimistic    0.02%    +18.4%     1.312    -14.2%
  base          0.10%    +17.9%     1.287    -14.5%
  stressed      0.30%    +16.8%     1.201    -15.1%
========================================================================
```

If stressed Sharpe drops more than 0.3 below base, a warning is logged.

**Replay invariants** are checked at the end of every run:
```
Replay invariants: all 5 checks passed.
```
Invariants checked: no fill before signal date, fills within replay window,
no negative shares, no zero-price fills, all actions are known types.
Any failure logs `INVARIANT FAIL` with details.

**Replay manifest** (`plots/replay_manifest.json`) records:
- `code_version` — git commit hash
- `config_hash` — hash of broker.config at run time
- `checkpoint_path` — which model was used
- `resolved_universe_hash` — hash of the ticker set
- `snapshot_path` — frozen universe file used
- `benchmark.available` — whether SPY data was present
- `friction` — execution cost assumptions
- `watchlist_included` — whether watchlist tickers were included

### RL ablation study
```bash
python Agent.py --mode ablation --rl_checkpoint models/best_fold9.pt
```
Runs all four strategy variants over the same historical period and produces
a side-by-side comparison:

| Variant | Screener | Ranking |
|---|---|---|
| `heuristics_only` | Rule-based | Composite score (baseline) |
| `screener_heuristics` | TickerScorer ML | Composite score |
| `screener_rl` | TickerScorer ML | RL score |
| `rl_weights` | TickerScorer ML | Direct RL weights |

Saves `plots/ablation_report.csv` and `plots/ablation.png`. Prints a gate
result (`PASSED`/`FAILED`) — the RL variant must beat the heuristic baseline
by at least 0.10 Sharpe without worsening max drawdown by more than 5pp
before the integration is considered production-ready.

### Update data manually
```bash
python Agent.py --mode update
```
The broker does this automatically on every run, so you only need this
manually if you want to update data without running a full broker cycle.

### Background scheduler (optional)
```bash
python Agent.py --mode schedule
```
Runs continuously in the background and handles everything automatically:

- **17:00 Mon–Fri** — fetches latest prices and news sentiment
- **Sunday 20:00** — fine-tunes the RL model on recent data, then immediately runs an auto-tune pass that:
  - Searches the parameter grid (min_score, stop_loss, take_profit, max_sector) over the last 2 years of replay data and writes the best combination back to `broker.config`
  - Runs the ablation gate — if `screener_rl` beats `heuristics_only` by the required margin, sets `rl_enabled = true` automatically; if not, sets it to `false`
  - Logs every decision with the data that drove it to `logs/autotuner.log`

Once the scheduler is running, `broker.config` is a live file managed by the system. You don't need to edit it manually for performance parameters or RL mode — the system decides based on data. Note: the scheduler still depends on live data sources. If sources are degraded, the auto-tune pass may produce conservative parameters or skip RL mode. Check `logs/autotuner.log` to see what drove each decision.

---

## Seeding an initial portfolio

If you want to start with specific positions rather than letting the broker
build from scratch:

1. Edit `seed_portfolio.py` — set `STARTING_CASH` and fill in `POSITIONS`:
```python
STARTING_CASH = 10000

POSITIONS = {
    "AAPL": 2500,   # invest $2500 in Apple at today's price
    "NVDA": 2500,
    "MSFT": 2500,
    "AMZN": 2500,
}
```

2. Run it:
```bash
python seed_portfolio.py
```

It fetches today's last close price for each ticker, calculates shares,
and writes the portfolio state. The broker takes over from there.

---

## SPY benchmark — how performance is measured

Every broker run fetches the current SPY price and tracks it alongside your
equity. Shown automatically at the end of every run and via `--status`:

- **Beta** — how much your portfolio moves relative to SPY (1.0 = moves with market)
- **Alpha** — annualised return above what beta alone would predict
- **Information ratio** — consistency of outperformance (higher = more consistent)
- **Upside capture** — how much of SPY's up days you capture
- **Downside capture** — how much of SPY's down days you suffer (lower = better)
- **Beats SPY** — simple yes/no on total return

The goal is positive alpha with downside capture below 1.0 — meaning you
make more than SPY on good days and lose less on bad days.

---

## File structure

```
Broker.py                     Run this to trade
Agent.py                      Research, training, backtesting, screening
broker.config                 Your persistent broker settings
requirements.txt
README.md

pipeline/
  data.py                     Load parquet, filter universe, merge sentiment
  features.py                 29 features: technical + sentiment + market context + regime
  environment.py              RL training environment
  model.py                    Transformer policy (temporal + cross-asset attention)
  train.py                    PPO walk-forward training with resume support
  backtest.py                 RL agent backtest vs SPY
  benchmark.py                SPY benchmark metrics (beta, alpha, IR, capture)
  screener.py                 Bidirectional GRU screener — cuts universe to ~50-100 candidates
  rl_inference.py             RL inference wrapper (get_rl_targets, ensemble support)
  autotuner.py                Auto-tunes broker params and RL mode from replay data
  maintenance.py              Staleness checks — runs from Broker.py on each cycle
  sentiment.py                News scraper + FinBERT scorer
  updater.py                  yfinance price fetcher + universe resolver
  universe_resolver.py        Shared universe resolution (tradable_us / sp500 / custom)
  run_manifest.py             Replay/live manifest helpers and schema enforcement
  checkpoints.py              Checkpoint resolution by val_sharpe
  scheduler.py                Daily/weekly automation

broker/
  broker.py                   Core broker logic + freshness gates + manifest emission
  brain.py                    Decision engine (exits, sectors, buys, RL ranking)
  portfolio.py                Cash, positions, options book, P&L tracking
  analyst.py                  On-demand stock research (live price + news)
  options.py                  Options strategies, Greeks, cash-secured accounting
  risk.py                     Portfolio risk engine + startup validation
  replay.py                   Broker replay + friction sensitivity + invariant checks
  shadows.py                  Shadow portfolio engine — evolutionary parameter search
  sectors.py                  Dynamic sector scoring and allocation
  validator.py                Data quality cross-verification (30%+ move check)
  universe.py                 New stock discovery (disabled in benchmark-constrained modes)
  journal.py                  Trade log, equity curve, SPY benchmark tracking

MasterDS/
  stooq_panel.parquet         Historical OHLCV (broad U.S. equity universe)

Sentiment/
  analyst_ratings_with_sentiment.csv   FinBERT-scored headlines

models/
  best_fold*.pt               Best Transformer checkpoint per training fold
  screener.pt                 Trained screener checkpoint
  universe_snapshots/         Dated universe snapshots for replay determinism

broker/state/                 Generated on first run — do not edit manually
  portfolio.json              Live portfolio (cash, positions, trade history)
  journal.jsonl               Full trade log with reasoning
  equity_curve.csv            Equity over time with SPY prices
  maintenance.json            Staleness timestamps for each auto-maintenance task
  shadows.json                Shadow portfolio state and standings
  last_live_manifest.json     Most recent live cycle manifest

plots/                        Charts and manifests saved here
  replay_manifest.json        Replay run manifest (universe hash, checkpoint, friction)
  live_universe_snapshot.json Frozen universe snapshot for replay determinism
logs/                         broker.log, autotuner.log, maintenance.log
tests/                        191+ tests covering universe, replay, manifests, exits
```

---

## Typical workflow

**Daily (morning):**
```bash
python Broker.py
```
One command. Price updates, sentiment, and trade decisions run automatically. Finetuning and parameter optimisation run on a weekly schedule. New entries are blocked if data quality falls below configured thresholds — check `logs/broker.log` to see what ran and why.

**First time only:**
```bash
python Agent.py --mode train --folds 10
```
Train the initial model. After that, the broker retrains itself.

**Optional — check in without trading:**
```bash
python Broker.py --status
```
Shows portfolio, SPY benchmark, and shadow portfolio standings.

---

## How to run a reproducible evaluation

To produce a result you can audit and reproduce later:

```bash
# 1. Freeze the universe snapshot
# In broker.config:
#   freeze_universe_snapshot = true
#   universe_snapshot_path = plots/live_universe_snapshot.json

# 2. Run the replay
python Agent.py --mode replay --replay_years 3

# 3. Archive the artifacts
# plots/replay_manifest.json  — config hash, universe hash, checkpoint, benchmark status
# plots/replay.png            — equity curve
# plots/replay_score_audit.csv — per-ticker decision trace
# plots/live_universe_snapshot.json — frozen universe used
```

The manifest records: timestamp, code version (git commit), config hash, checkpoint path, universe hash and size, benchmark availability, freshness coverage, and friction regime assumptions. Re-running with the same snapshot and checkpoint will produce the same trade log.

---

## Tips

- The sentiment surprise signal is the strongest indicator — sudden positive
  news before price moves is what the model looks for most
- High cash allocation in `--mode predict` means the model sees low
  conviction across the universe — often precedes volatile periods
- If the broker makes no trades, it's usually because nothing scored above
  `min_score`. Lower it in `broker.config` or wait for better setups
- When RL is enabled, `rl_min_score` controls the entry floor (default 0 = pure top-k ranking). `min_score` only applies to the heuristic path
- Run in paper mode for at least 30-90 days before using real capital
- The broker's backtest (`--mode replay`) and the RL agent's backtest
  (`--mode backtest`) measure different things — the replay is the honest
  number for what the broker actually does
- After any replay, check `plots/replay_manifest.json` to confirm which
  universe, checkpoint, and friction assumptions were used

---

## Testing

```bash
python -m pytest tests/ -q
```

191+ tests covering: universe resolution, replay determinism, manifest schema,
freshness gates, exit deduplication, RL exit checks, replay invariants,
friction sensitivity, and broker cycle behavior.

Two pre-existing failures are known and unrelated to core broker logic:
- `test_screener.py::test_build_samples_uses_raw_close_forward_returns`
- `test_regime_classifier.py::test_regime_broadcast_consistency`

Key test files by purpose:
- `tests/test_manifest_schema.py` — manifest required fields and schema enforcement
- `tests/test_freshness_gates.py` — adversarial freshness gate behavior
- `tests/test_replay_determinism.py` — replay produces identical results on same inputs
- `tests/test_replay_invariants.py` — impossible states (fill before signal, zero prices, etc.)
- `tests/test_tier2_tier3_audit.py` — live/replay parity, friction sensitivity, stale-state contamination
- `tests/test_rl_exit_checks.py` — rank-percentile exit logic
- `tests/test_exit_deduplication.py` — one exit action per ticker per cycle

---

## Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended for 1000-stock universe)
- GPU optional but significantly speeds up training
- Internet connection for daily data updates
