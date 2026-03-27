# StockBot — AI Stock Predictor & Autonomous Broker

---

## Agent vs Broker — what's the difference?

Think of it like this:

**Agent (`Agent.py`) = the researcher and trainer.**
It studies the market, learns patterns from 60+ years of historical data,
and tells you what stocks look promising. It does not manage money. It does
not buy or sell anything. It's the brain that gets trained.

**Broker (`Broker.py`) = the portfolio manager.**
It uses what the Agent learned, plus live data and news, to actually manage
a portfolio. It buys stocks, sells them, handles options, enforces risk
limits, and tracks performance vs SPY. This is the thing that runs daily.

In practice:
- You train the Agent once (takes hours), then occasionally retrain it
- You run the Broker every day (takes minutes), and it handles everything

Most of the time you only need one command:
```bash
python Broker.py
```

---

## How it works

Every stock is scored using 19 features computed daily:

**Technical signals (10):** returns over 1/5/20 days, RSI, MACD histogram,
Bollinger Band position, ATR volatility, volume ratio, volume z-score,
52-week price position.

**Sentiment signals (9):** all derived from FinBERT-scored news headlines —
net sentiment score, 3/7/14-day rolling averages, sentiment surprise (today
vs 14-day baseline), sentiment acceleration, 7-day trend slope, raw positive
confidence, negativity spike detection.

The sentiment surprise signal is the most forward-looking — a sudden jump in
positive news before price moves is the strongest buy signal in the system.

All features are cross-sectionally z-scored against the full universe each
day, so a value of +1.5 means that stock is 1.5 standard deviations above
the average stock on that signal.

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

# Step 2: Seed your portfolio with starting positions (optional)
# Edit seed_portfolio.py first, then:
python seed_portfolio.py

# Step 3: Run the broker
python Broker.py
```

After step 3, just run `python Broker.py` whenever you want an update.
Everything else is automatic.

---

## broker.config — Your persistent settings

Edit `broker.config` once. These become your defaults every run — no flags needed.

```
cash           = 10000    # starting cash (first run only — ignored after that)
max_positions  = 10       # focused portfolio — 10 names on $10k is enough
stop_loss      = 0.08     # ATR-adjusted upward per stock; 8% floor avoids noise clips
take_profit    = 0.35     # full exit — realistic for a daily-checked system
partial_profit = 0.15     # take half off at +15%, let rest run to take_profit
min_score      = 0.58     # high enough to filter mediocre setups
penny_pct      = 0.03     # minimal speculative exposure until system is proven
max_sector     = 0.25     # prevents correlated sector drawdowns
avoid_earnings = 5        # earnings risk starts well before the report date
top_n          = 500      # sufficient universe for a $10k daily broker
max_daily_loss = 0.025    # tighter during paper trading — 2.5% daily halt
max_drawdown   = 0.12     # tighter during paper trading — 12% circuit breaker
no_options     = true     # disabled until stock-side edge is proven in paper trading

# RL integration (opt-in — defaults to heuristic mode)
rl_enabled            = false              # set true to activate RL ranking
rl_checkpoint_path    = models/best_fold9.pt
rl_phase              = 1                  # 1=ranking, 2=ranking+exits, 3=weights (future)
rl_exit_threshold     = 0.30              # Phase 2: sell if rl_score drops below this
rl_conviction_drop    = 0.20              # Phase 2: sell 50% if score drops by this much
rl_min_score          = 0.0               # Phase 1: min rl_score to enter (0 = top-k only)
```

---

## Broker — Running and monitoring

### Run a cycle
```bash
python Broker.py
```

Every run automatically:
1. Fetches latest prices from yfinance
2. Scrapes fresh news headlines and scores them with FinBERT
3. Validates portfolio state and data freshness
4. Checks exits on all held positions (stops, take-profits, signal deterioration)
5. Scores all 11 market sectors and sets allocation targets
6. Screens 1000+ stocks for buy candidates
7. Deep-researches top candidates with live price + sentiment data
8. Executes buy/sell/options decisions
9. Logs everything and compares to SPY
10. Exits

At the end of every run you automatically see:
- Full portfolio summary (all positions, cash, total return, unrealised P&L per stock)
- SPY benchmark comparison (beta, alpha, information ratio, beats SPY yes/no)
- What changed this cycle — what was bought, what was sold, and why

Your portfolio balance, positions, and trade history persist between runs
in `broker/state/`. Nothing resets. The `cash` value in `broker.config`
is only used on the very first run ever.

### Check portfolio without trading
```bash
python Broker.py --status
```
Shows the same full output as after a normal run, but without executing
any trades. Use this any time you want to check in without triggering a cycle.

### See more trade history
```bash
python Broker.py --trades
```
Shows the last 30 trades with full reasoning. The normal run shows 10 —
use this when you want to dig deeper into trade history.

### Override a setting for one run
```bash
python Broker.py --min_score 0.55      # stricter this run only
python Broker.py --penny_pct 0.20      # more penny exposure this run
python Broker.py --no_options          # skip options this run
python Broker.py --no_market_hours     # run outside market hours (testing)
```

### Schedule it
Run once daily after market close. The broker enforces market hours
internally — new entries are blocked outside 9:30am–4:00pm ET, but
exits and price updates still run.

Windows Task Scheduler:
- Program: `python`
- Arguments: `Broker.py`
- Start in: `C:\path\to\StockBot`
- Trigger: Daily at 4:30pm, Mon–Fri

Cron (Mac/Linux):
```
30 16 * * 1-5 cd /path/to/StockBot && python Broker.py
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

### Candidate selection and sizing
1. Screens 1000 stocks using the trained screener (or rule-based fallback)
2. Skips any stock within `avoid_earnings` days of earnings
3. Deep-researches top candidates: fetches live OHLCV + FinBERT news
4. Scores each on 19 signals → composite score 0–1
5. Skips anything below `min_score`
6. Sizes position by conviction: higher score = larger allocation
7. Applies sector budget, penny cap, cash floor, and volatility scaling
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
individually including sub-$5 stocks.

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
Runs the actual broker decision logic over historical data — same screening,
sector allocation, stops, take-profits, and execution costs as the live
broker. This is the honest performance number for the broker.

```bash
python Agent.py --mode replay --replay_years 5     # use 5 years of history
python Agent.py --mode replay --sensitivity        # also run sensitivity sweep
```

The sensitivity sweep runs the replay across 13 parameter combinations to
test whether results are robust or collapse under parameter changes.

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
Runs continuously: fetches prices + news at 17:00 Mon–Fri, fine-tunes the
model every Sunday at 20:00. Only needed if you want the model to keep
improving automatically without manual retraining.

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
seed_portfolio.py             One-time portfolio seeder
requirements.txt
README.md
AUDIT.md                      Full production readiness checklist

pipeline/
  data.py                     Load parquet, filter universe, merge sentiment
  features.py                 19 features: 10 technical + 9 sentiment
  environment.py              RL training environment
  model.py                    Transformer policy (temporal + cross-asset attention)
  train.py                    PPO walk-forward training with resume support
  backtest.py                 RL agent backtest vs SPY
  benchmark.py                SPY benchmark metrics (beta, alpha, IR, capture)
  screener.py                 Per-ticker buy signal scorer (all 11,500+ tickers)
  rl_inference.py             RL inference wrapper (get_rl_targets, WeightAdapter stub)
  sentiment.py                News scraper + FinBERT scorer
  updater.py                  yfinance price fetcher
  scheduler.py                Daily/weekly automation

broker/
  broker.py                   Core broker logic
  brain.py                    Decision engine (exits, sectors, buys, options)
  portfolio.py                Cash, positions, options book, P&L tracking
  analyst.py                  On-demand stock research (live price + news)
  options.py                  Options strategies, Greeks, cash-secured accounting
  risk.py                     Portfolio risk engine + startup validation
  replay.py                   Broker replay backtest + sensitivity sweep + ablation
  sectors.py                  Dynamic sector scoring and allocation
  validator.py                Data quality cross-verification (30%+ move check)
  universe.py                 New stock discovery (Finviz + Yahoo trending)
  journal.py                  Trade log, equity curve, SPY benchmark tracking

MasterDS/
  stooq_panel.parquet         Historical OHLCV (26M rows, 1962-present)

Sentiment/
  analyst_ratings_with_sentiment.csv   FinBERT-scored headlines (1.5M+ rows, growing)

models/
  best_fold0-9.pt             Best Transformer checkpoint per training fold

broker/state/                 Generated on first run — do not edit manually
  portfolio.json              Live portfolio (cash, positions, trade history)
  journal.jsonl               Full trade log with reasoning
  equity_curve.csv            Equity over time with SPY prices
  sector_cache.json           Cached GICS sector classifications
  watchlist.csv               Discovered new tickers

plots/                        Charts saved here (backtest.png, replay.png, etc.)
logs/                         broker.log, scheduler.log
```

---

## Typical workflow

**Daily (after market close):**
```bash
python Broker.py
```
That's it. Automatically updates data, makes decisions, shows you everything.

**Weekly (Monday morning, optional):**
```bash
python Agent.py --mode predict
```
See the model's top picks for the week with signal breakdown.

**Monthly (optional):**
```bash
python Agent.py --mode replay --sensitivity
```
Check whether the broker is actually beating SPY and whether results hold
up across parameter changes.

**After major market events:**
```bash
python Agent.py --mode finetune
```
Adapt the model to the new market regime.

---

## Tips

- The sentiment surprise signal is the strongest indicator — sudden positive
  news before price moves is what the model looks for most
- High cash allocation in `--mode predict` means the model sees low
  conviction across the universe — often precedes volatile periods
- If the broker makes no trades, it's usually because nothing scored above
  `min_score`. Lower it in `broker.config` or wait for better setups
- When RL is enabled, `rl_min_score` controls the entry floor (default 0 = pure top-k ranking). `min_score` only applies to the heuristic path
- AUDIT.md contains the full production readiness checklist — read it
  before committing real money
- Run in paper mode for at least 30-90 days before using real capital
- The broker's backtest (`--mode replay`) and the RL agent's backtest
  (`--mode backtest`) measure different things — the replay is the honest
  number for what the broker actually does

---

## Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended for 1000-stock universe)
- GPU optional but significantly speeds up training
- Internet connection for daily data updates
