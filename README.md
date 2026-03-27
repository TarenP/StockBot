# Stock Predictor & Autonomous Broker

A reinforcement learning portfolio agent and an autonomous broker that manages
a real portfolio across stocks, penny stocks, and options — driven by technical
analysis and live news sentiment scored by FinBERT.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the model (first time only, takes a few hours)
python Agent.py --mode train --folds 10

# 2. Run the broker whenever you want an update
python Broker.py --cash 10000
```

The broker is one-shot — run it manually, schedule it via Task Scheduler or
cron, or just run it whenever you feel like checking in. It does its work
and exits.

---

## How it works

**Agent** (`Agent.py`) trains a Transformer model on 60+ years of price
history, scores every stock in the universe weekly, and outputs ranked picks.

**Broker** (`Broker.py`) manages a live portfolio. Each run it:
1. Validates data freshness and portfolio state
2. Checks exits on all held positions (ATR-adjusted stops, take-profits)
3. Scores all 11 market sectors and decides its own allocation targets
4. Screens the full universe for buy candidates
5. Deep-researches top candidates with live price data + FinBERT news scoring
6. Executes buy/sell/options decisions with execution cost applied
7. Logs everything and compares performance to SPY
8. Exits

Both use the same signal engine: 19 features per stock (10 technical + 9
sentiment) cross-sectionally z-scored against the universe each day.

---

## Agent — Commands

### Get weekly stock picks
```bash
python Agent.py --mode predict
```
```
========================================================================
  Weekly Portfolio Picks  —  as of 2026-03-27
  Universe: 500 stocks  |  Showing top 10
========================================================================
  #    Ticker   Weight   Mom20d     RSI  Sentiment  Vol Ratio
  ------------------------------------------------------------------
  1    NVDA      8.43%    +1.82   +0.94      +1.21      +2.10
  2    META      7.21%    +1.44   +0.71      +0.88      +1.33
  ...
  ------------------------------------------------------------------
  CASH           12.50%
========================================================================
```
Signal values are z-scored — +1.5 means 1.5 standard deviations above the
universe average on that signal.

```bash
python Agent.py --mode predict --top_k 20      # show top 20
python Agent.py --mode predict --top_n 300     # smaller universe
```

---

### Screen all stocks (including penny stocks)
```bash
# Train the screener once before using it
python Agent.py --mode train_screener

# Scan all 11,500+ tickers
python Agent.py --mode screen

# Penny stocks only (under $5)
python Agent.py --mode screen --penny

# Custom price range
python Agent.py --mode screen --min_price 1 --max_price 10 --screener_top_n 100
```

---

### Fetch latest data
```bash
python Agent.py --mode update
```
Pulls today's OHLCV from yfinance and fetches + scores fresh news headlines
with FinBERT. Safe to run multiple times.

```bash
python Agent.py --mode update --force_refresh  # re-download last 30 days
```

Optional — NewsAPI key for higher-quality news (100 req/day free tier):
```bash
set NEWSAPI_KEY=your_key_here        # Windows
export NEWSAPI_KEY=your_key_here     # Mac/Linux
```

---

### Train from scratch
```bash
python Agent.py --mode train --folds 10
```
Walk-forward training: each fold trains on 8 years, validates on 1, tests on 1.
No data leakage. Saves best checkpoint per fold to `models/`.

If you stop mid-fold, progress is saved every 5,000 steps. Re-run the same
command to resume — completed folds are skipped automatically.

```bash
python Agent.py --mode train --folds 10 --total_steps 200000   # longer training
python Agent.py --mode train --folds 5  --top_n 300            # faster, smaller universe
```

Training time: ~2–6 hours on CPU, ~30–60 min on GPU.

---

### Fine-tune on recent data
```bash
python Agent.py --mode finetune
```
Continues training the best checkpoint on the most recent 2 years. Run this
after major market regime changes. Takes ~20 min.

---

### Backtest
```bash
python Agent.py --mode backtest
```
Evaluates the RL agent on the held-out test period vs SPY and equal-weight.
SPY is fetched automatically. Saves a 4-panel chart to `plots/backtest.png`.

---

### Broker replay backtest
```bash
python Agent.py --mode replay
```
Runs the **actual broker decision logic** over historical data — same
screening, sector allocation, stop-losses, take-profits, and execution costs
as the live broker. This is the honest performance number, not the RL agent's
idealized backtest.

```bash
python Agent.py --mode replay --replay_years 5     # longer history
python Agent.py --mode replay --sensitivity        # also run sensitivity sweep
```

The sensitivity sweep runs the replay across 13 parameter combinations
(varying min_score, stop-loss, take-profit, and execution spread) and reports
whether results hold up or collapse — the key test for overfitting.

Output:
```
=======================================================================
  Benchmark Report — Broker Replay vs SPY
=======================================================================
  Metric                  Broker Replay           SPY   Equal-Weight
  ─────────────────────────────────────────────────────────────────
  total_return               +47.3%           +38.1%        +29.4%
  ann_return                 +13.6%           +11.4%         +9.0%
  sharpe                      1.21             0.89          0.74
  max_drawdown               -14.2%           -23.8%        -28.1%
  ...
  Beta                         0.72
  Alpha (ann)                 +3.8%
  Beats SPY (return)            YES
=======================================================================
```

---

### Keep data up to date automatically
```bash
python Agent.py --mode schedule
```
Runs in the background: fetches prices + news at 17:00 Mon–Fri, fine-tunes
the model every Sunday at 20:00. Logs to `logs/scheduler.log`.

---

## Broker — Commands

### Run a cycle
```bash
python Broker.py
```
Does one full cycle and exits. Every run automatically:
- Fetches latest prices from yfinance
- Scrapes fresh news headlines and scores them with FinBERT
- Makes buy/sell/options decisions
- Logs results vs SPY

Portfolio state (cash, positions, trade history) persists between runs in
`broker/state/`. Your balance carries over every time — nothing resets.

**First run only** — set your starting cash:
```bash
python Broker.py --cash 10000
```
After that, just run `python Broker.py` with no flags. The `--cash` argument
is ignored once a portfolio exists.

```bash
python Broker.py                       # standard run — just use this every time
python Broker.py --max_positions 30    # hold more stocks
python Broker.py --penny_pct 0.40      # more aggressive penny stock allocation
python Broker.py --min_score 0.70      # higher conviction threshold
python Broker.py --no_options          # stocks only
python Broker.py --no_market_hours     # skip market hours check (for testing)
```

### Check status without trading
```bash
python Broker.py --status    # portfolio summary + SPY benchmark comparison
python Broker.py --trades    # recent trade history with full reasoning
```

### Schedule it
Run once daily after market close. Examples:

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

### Broker settings

| Flag | Default | Description |
|------|---------|-------------|
| `--cash` | 10000 | Starting cash (first run only) |
| `--max_positions` | 20 | Max simultaneous stock positions |
| `--stop_loss` | 0.07 | ATR stop-loss floor (actual stop adjusted upward by volatility) |
| `--take_profit` | 0.45 | Full exit threshold (partial exit at +20%) |
| `--min_score` | 0.50 | Minimum signal score to buy (0–1) |
| `--penny_pct` | 0.20 | Max % of portfolio in penny stocks |
| `--max_sector` | 0.40 | Hard cap per sector |
| `--avoid_earnings` | 3 | Skip stocks within N days of earnings |
| `--top_n` | 1000 | Universe size for screening |
| `--max_daily_loss` | 0.03 | Halt new entries if down 3% in one session |
| `--max_drawdown` | 0.15 | Circuit breaker: no new entries if drawdown > 15% |
| `--no_options` | off | Disable options trading |
| `--no_market_hours` | off | Skip market hours check |

---

### How the broker makes decisions

Each run:

1. **Startup validation** — refuses to trade if price data is >3 days stale,
   portfolio state is inconsistent, or options cash reservation exceeds cash.

2. **Price validation** — any move >30% is cross-checked across yfinance,
   Finviz, and news headlines before being accepted. Bad data is rejected.

3. **Exit checks** on every held position:
   - Stop-loss: ATR-adjusted per stock (volatile stocks get wider stops)
   - Partial take-profit: sells 50% at +20%, lets the rest run
   - Full take-profit: exits remaining position at +45%
   - Signal deterioration: exits if composite score drops below 0.35

4. **Risk engine** — halts new entries if daily loss limit or drawdown
   circuit breaker is triggered. Scales position sizes down when portfolio
   volatility is elevated.

5. **Sector scoring** — scores all 11 GICS sectors on momentum, sentiment,
   breadth, and volume. Converts to target allocations with a quadratic
   diversification penalty (concentration is increasingly punished).

6. **Screening** — runs the trained screener across the full universe.
   Skips stocks within `--avoid_earnings` days of earnings.

7. **Research** — fetches live price data and FinBERT-scored news for each
   candidate. Computes composite score from 19 signals.

8. **Position sizing** — conviction-scaled allocation, constrained by sector
   budget, penny cap, cash floor, and volatility scaling. Execution cost
   (bid-ask spread) applied before buying.

9. **Options** — evaluates top candidates for options strategies:
   - Long Call (strong bullish + positive sentiment)
   - Bull Call Spread (moderate bullish)
   - Long Put (bearish hedge)
   - Cash-Secured Put (neutral-bullish income, full cash reserved)

10. **SPY benchmark** — fetches SPY price every run, tracks alpha, beta,
    information ratio, upside/downside capture. Shown in `--status`.

State persists across runs in `broker/state/`.

---

## Sentiment signals

9 FinBERT-derived features used by both agent and broker:

| Signal | What it captures |
|--------|-----------------|
| `sent_net` | Positive − negative score |
| `sent_ma3/7/14` | Rolling sentiment momentum |
| `sent_surprise` | Today vs 14-day baseline — catches sudden news shifts |
| `sent_accel` | Short MA crossing medium MA |
| `sent_trend` | 7-day slope |
| `sent_pos_raw` | Raw positive confidence |
| `sent_neg_spike` | Sudden negativity vs recent average |

The sentiment surprise signal is the strongest forward-looking indicator —
a sudden positive news shift before price moves is the model's top buy signal.

---

## File structure

```
Agent.py                      Prediction, training, screening, scheduling
Broker.py                     Autonomous portfolio manager (one-shot)
requirements.txt
README.md
AUDIT.md                      Full gap analysis and production readiness checklist

pipeline/
  data.py                     Load parquet, filter universe, merge sentiment
  features.py                 19 features: 10 technical + 9 sentiment
  environment.py              RL training environment
  model.py                    Transformer policy
  train.py                    PPO walk-forward training with resume support
  backtest.py                 Backtesting vs SPY and equal-weight
  benchmark.py                SPY benchmark metrics (beta, alpha, IR, capture ratios)
  screener.py                 Per-ticker buy signal scorer (all 11,500+ tickers)
  sentiment.py                News scraper + FinBERT scorer
  updater.py                  yfinance price fetcher
  scheduler.py                Daily/weekly automation

broker/
  broker.py                   Main entry point (one-shot)
  brain.py                    Decision engine
  portfolio.py                Cash, stock positions, options book, P&L
  analyst.py                  On-demand stock research (price + news)
  options.py                  Options strategies, Greeks, cash-secured accounting
  risk.py                     Portfolio risk engine + startup validation
  replay.py                   Broker replay backtest + sensitivity sweep
  sectors.py                  Dynamic sector scoring and allocation
  validator.py                Data quality cross-verification
  universe.py                 New stock discovery
  journal.py                  Trade log, equity curve, SPY benchmark tracking

MasterDS/
  stooq_panel.parquet         Historical OHLCV (26M rows, 1962–present)

Sentiment/
  analyst_ratings_with_sentiment.csv   FinBERT-scored headlines (growing)

models/
  best_fold0-9.pt             Best Transformer checkpoint per training fold

plots/                        Backtest and benchmark charts (generated)
logs/                         Broker and scheduler logs (generated)
broker/state/                 Live portfolio state (generated on first run)
```

---

## Tips

- Run `python Agent.py --mode update` before `Broker.py` to get fresh data
- Run `python Broker.py --status` any time to see portfolio + SPY comparison
- The sentiment surprise signal is the strongest buy indicator — sudden
  positive news before price moves is what the model looks for most
- High cash allocation in `--mode predict` = model sees low conviction,
  often precedes volatile periods
- After a major market event, run `python Agent.py --mode finetune` to
  adapt the model to the new regime
- `AUDIT.md` contains the full production readiness checklist — read it
  before committing real money

---

## Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- GPU optional but speeds up training significantly
- Internet connection for data updates
