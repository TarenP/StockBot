# Stock Predictor & Autonomous Broker

A reinforcement learning portfolio agent combined with an autonomous broker that
manages a real portfolio across stocks, penny stocks, and options — driven by
technical analysis and live news sentiment.

---

## How it works

The system has two independent components:

**Agent** (`Agent.py`) — a research and prediction tool. Trains a Transformer
model on 60+ years of price history, scores every stock in the universe each
week, and outputs ranked picks with signal breakdowns.

**Broker** (`Broker.py`) — an autonomous portfolio manager. Runs continuously,
makes its own buy/sell/options decisions, manages risk, and adapts to market
conditions without manual input.

Both use the same underlying signal engine: 19 features per stock (10 technical
indicators + 9 sentiment signals) scored by FinBERT on live news headlines.

---

## Setup

```bash
pip install -r requirements.txt

# First-time training
python Agent.py --mode train

# Get this week's picks
python Agent.py --mode predict
```

FinBERT (~500MB) downloads automatically on first sentiment run.

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
  Signals are cross-sectionally z-scored (0 = universe avg).
  Rebalance weekly. Not financial advice.
```

Signal values are z-scored against the full universe — +1.5 means that stock
is 1.5 standard deviations above the average on that signal.

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

The screener has no universe size limit — it scores every ticker individually
so it handles the full parquet including penny stocks. Output includes a
buy-signal score (0–1), momentum, RSI, sentiment surprise, and volume ratio.

---

### Fetch latest data
```bash
python Agent.py --mode update
```
Pulls today's OHLCV from yfinance and fetches + scores fresh news headlines
with FinBERT. Safe to run multiple times (deduplicates automatically).

```bash
python Agent.py --mode update --force_refresh  # re-download last 30 days
```

**Optional — NewsAPI key** for higher-quality news (100 req/day free):
```bash
set NEWSAPI_KEY=your_key_here        # Windows
export NEWSAPI_KEY=your_key_here     # Mac/Linux
```

---

### Train from scratch
```bash
python Agent.py --mode train
```
Walk-forward training: each fold trains on 8 years, validates on 1, tests on 1.
No data leakage. Saves best checkpoint per fold to `models/`.

```bash
python Agent.py --mode train --folds 10         # recommended for robustness
python Agent.py --mode train --total_steps 200000
python Agent.py --mode train --top_n 300        # smaller = faster
```

Training time: ~2–6 hours on CPU, ~30–60 min on GPU.

If you stop mid-fold, progress is saved automatically every 5,000 steps.
Re-run the same command to resume — completed folds are skipped.

---

### Fine-tune on recent data
```bash
python Agent.py --mode finetune
```
Continues training the best checkpoint on the most recent 2 years. Run this
after major market regime changes or every few months. Takes ~20 min.

---

### Backtest
```bash
python Agent.py --mode backtest
```
Evaluates the best checkpoint on the held-out test period vs equal-weight.
Saves equity curve to `plots/backtest.png`.

```
=================================================================
Metric               Policy   Equal-Weight
-----------------------------------------------------------------
total_return          +84.3%        +41.2%
ann_return            +12.1%         +7.3%
sharpe                  1.43          0.81
sortino                 2.11          1.02
max_drawdown          -18.4%        -31.7%
calmar                  0.66          0.23
win_rate               54.2%         51.1%
=================================================================
```

---

### Keep everything up to date automatically
```bash
python Agent.py --mode schedule
```

| Time          | Action                                        |
|---------------|-----------------------------------------------|
| 17:00 Mon–Fri | Fetch new prices + news, score with FinBERT   |
| Sunday 20:00  | Fine-tune model on recent 2 years of data     |

Logs to `logs/scheduler.log`. To run permanently on Windows, add to Task
Scheduler pointing to `python Agent.py --mode schedule`.

---

## Autonomous Broker — Commands

The broker manages a real portfolio autonomously. It runs on a schedule,
makes buy/sell/options decisions, and handles all risk management itself.

### Start the broker
```bash
# $10,000 portfolio, trade daily
python Broker.py --cash 10000 --interval daily

# Larger portfolio, trade every 4 hours
python Broker.py --cash 50000 --interval 4hour --max_positions 30

# Aggressive penny stock allocation
python Broker.py --cash 10000 --interval daily --penny_pct 0.40

# Conservative — tighter stops, higher conviction threshold
python Broker.py --cash 10000 --interval daily --stop_loss 0.07 --min_score 0.70

# Stocks only, no options
python Broker.py --cash 10000 --interval daily --no_options
```

### Check status and history
```bash
python Broker.py --status      # portfolio summary + P&L + open options
python Broker.py --trades      # recent trade history with full reasoning
python Broker.py --once        # run one cycle and exit
```

### Broker settings

| Flag | Default | Description |
|------|---------|-------------|
| `--cash` | 10000 | Starting cash (only used on first run) |
| `--interval` | daily | Trade frequency: hourly, 2hour, 4hour, daily, weekly |
| `--max_positions` | 20 | Max simultaneous stock positions |
| `--stop_loss` | 0.12 | Stop-loss floor — actual stop is ATR-adjusted above this |
| `--take_profit` | 0.45 | Full exit threshold (partial exit at +20%) |
| `--min_score` | 0.60 | Minimum signal score to buy (0–1) |
| `--penny_pct` | 0.20 | Max % of portfolio in penny stocks |
| `--max_sector` | 0.40 | Hard cap per sector (broker self-adjusts below this) |
| `--avoid_earnings` | 3 | Skip stocks within N days of earnings (0 = disabled) |
| `--top_n` | 500 | Universe size for screening |
| `--no_options` | off | Disable options trading |

---

### How the broker makes decisions

Each cycle the broker:

1. Validates price updates — any move >30% is cross-checked across yfinance,
   Finviz, and news before being accepted. Bad data is rejected, real moves
   are confirmed and flagged.

2. Checks exits on every held position:
   - Stop-loss: ATR-adjusted per stock (volatile stocks get wider stops)
   - Partial take-profit: sells 50% at +20%, lets the rest run
   - Full take-profit: exits remaining position at +45%
   - Signal deterioration: exits if composite score drops below 0.35

3. Scores all 11 market sectors dynamically — momentum, sentiment, breadth,
   and volume surge. Converts scores into target allocations using a softmax
   with a quadratic diversification penalty (concentration is punished
   increasingly as it grows). The broker decides its own sector weights.

4. Screens the full universe for buy candidates using the trained screener.

5. Skips any stock with earnings within 3 days (configurable).

6. Deep-researches top candidates: fetches live price data + news headlines,
   computes all 19 signals, scores with FinBERT.

7. Sizes positions by conviction: higher score = larger allocation, constrained
   by sector budget, penny cap, and available cash.

8. Evaluates options opportunities for top-scored stocks (see below).

Every Sunday it discovers new stocks by scraping Finviz and Yahoo Finance
trending, validates them via yfinance, and adds their price history to the
parquet.

State persists across restarts — portfolio, positions, options, and trade
journal are saved to `broker/state/`.

---

### Options trading

The broker trades options automatically alongside stocks. All strategies are
defined-risk — maximum loss is always the premium paid.

**Strategies used:**

| Signal | Strategy | Why |
|--------|----------|-----|
| Score > 0.75, strong positive sentiment | Long Call | Strong bullish — leveraged upside |
| Score > 0.65, mild positive sentiment | Bull Call Spread | Moderate bullish — cheaper than outright call |
| Score < 0.35, negative sentiment | Long Put | Bearish hedge |
| Score > 0.60, neutral-bullish | Cash-Secured Put | Collect premium; buy stock cheaper if assigned |

Options budget is capped at 10% of total equity. Positions auto-close when
P&L reaches 50% of maximum profit. Expiry is handled automatically — ITM
options are exercised, OTM options expire worthless, short put assignments
are logged as stock purchases.

Portfolio summary shows options alongside stocks:
```
  Positions: 12 stocks, 3 options
  ────────────────────────────────────────────────────────
  Contract                        DTE    Delta    Theta
  ────────────────────────────────────────────────────────
  LONG CALL NVDA $950             21d   +0.420   -0.85/d
  BULL CALL SPREAD AAPL $195      14d   +0.310   -0.62/d
  CASH-SECURED PUT MSFT $380       7d   -0.180   -0.42/d
```

---

## Sentiment signals

Both the agent and broker use 9 sentiment features derived from FinBERT scores:

| Signal | What it captures |
|--------|-----------------|
| `sent_net` | Raw positive − negative score |
| `sent_ma3/7/14` | Rolling sentiment momentum |
| `sent_surprise` | Today vs 14-day baseline — catches sudden news shifts |
| `sent_accel` | Short MA crossing medium MA |
| `sent_trend` | 7-day slope of sentiment |
| `sent_pos_raw` | Raw positive confidence |
| `sent_neg_spike` | Sudden negativity vs recent average |

The sentiment surprise signal is the most forward-looking — a sudden jump in
positive news before price moves is the strongest buy signal in the model.

---

## File structure

```
Agent.py                      Prediction, training, screening, scheduling
Broker.py                     Autonomous portfolio manager
requirements.txt
README.md

pipeline/
  data.py                     Load parquet, filter universe, merge sentiment
  features.py                 19 features: 10 technical + 9 sentiment
  environment.py              RL training environment (Sharpe reward + tx costs)
  model.py                    Transformer policy (temporal + cross-asset attention)
  train.py                    PPO walk-forward training with resume support
  backtest.py                 Performance metrics + equity curve plots
  screener.py                 Per-ticker buy signal scorer (all 11,500+ tickers)
  sentiment.py                News scraper + FinBERT scorer
  updater.py                  yfinance price fetcher → appends to parquet
  scheduler.py                Daily/weekly automation loop

broker/
  broker.py                   Main loop + CLI
  brain.py                    Decision engine (exits, sectors, buys, options)
  portfolio.py                Cash, stock positions, options book, P&L
  analyst.py                  On-demand stock research (price + news)
  options.py                  Options strategies, Greeks, OptionsBook
  sectors.py                  Dynamic sector scoring + allocation
  validator.py                Data quality cross-verification
  universe.py                 New stock discovery (Finviz + Yahoo trending)
  journal.py                  Trade log + equity curve

MasterDS/
  stooq_panel.parquet         Historical OHLCV (26M rows, 1962–present)

Sentiment/
  analyst_ratings_with_sentiment.csv   FinBERT-scored headlines (growing)

models/
  best_fold0-9.pt             Best Transformer checkpoint per training fold

plots/
  backtest.png                Equity curve from last backtest

logs/
  broker.log                  Broker activity log
  scheduler.log               Scheduler activity log

broker/state/
  portfolio.json              Live portfolio state (persists across restarts)
  journal.jsonl               Full trade history
  equity_curve.csv            Equity over time
  sector_cache.json           Cached sector classifications
  watchlist.csv               Discovered tickers
```

---

## Tips

- Run `--mode predict` every Monday before market open for the week's picks
- Run `--mode update` before predicting to get fresh sentiment data
- The sentiment surprise signal is the strongest forward-looking indicator —
  a sudden positive news shift before price moves is the model's top buy signal
- High cash allocation in predict output = model sees low conviction across
  the universe, often precedes volatile periods
- After a major market event, run `--mode finetune` to adapt to the new regime
- For the broker, `--interval 4hour` is a good balance between responsiveness
  and avoiding overtrading
- Penny stocks are capped at `--penny_pct` of equity — raise it if you want
  more exposure, but be aware of the higher volatility

---

## Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended for full 500-stock universe)
- GPU optional but significantly speeds up training
- Internet connection for daily data updates
