# StockBot — AI Stock Predictor & Autonomous Broker

A complete AI-powered stock trading system built on reinforcement learning,
transformer models, and live news sentiment. It has two main components that
work together:

- **Agent** (`Agent.py`) — trains a model, generates weekly stock picks,
  backtests performance, and screens the full market universe
- **Broker** (`Broker.py`) — autonomously manages a live portfolio, makes
  buy/sell/options decisions, tracks performance vs SPY, and handles all
  risk management

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
# Install all dependencies
pip install -r requirements.txt
```

FinBERT (~500MB) downloads automatically the first time sentiment scoring runs.

---

## First time setup

```bash
# Step 1: Train the model (takes 2-6 hours on CPU, 30-60 min on GPU)
python Agent.py --mode train --folds 10

# Step 2: Seed your portfolio with starting positions (optional)
# Edit seed_portfolio.py first, then run:
python seed_portfolio.py

# Step 3: Run the broker
python Broker.py
```

After step 3, just run `python Broker.py` whenever you want an update.
Everything else is automatic.

---

## broker.config — Your persistent settings

Edit `broker.config` to set your preferences. These become the defaults
every time you run `python Broker.py`. You never need to pass flags unless
you want a one-off override.

```
cash           = 10000    # starting cash (first run only — ignored after that)
max_positions  = 20       # max stocks held at once
stop_loss      = 0.07     # minimum stop-loss (ATR-adjusted upward per stock)
take_profit    = 0.45     # full exit at +45% (partial exit at +20%)
min_score      = 0.50     # minimum signal score to buy (0-1)
penny_pct      = 0.10     # max % of portfolio in $2-$5 stocks
max_sector     = 0.40     # hard cap per sector
avoid_earnings = 3        # skip stocks within N days of earnings
top_n          = 1000     # how many stocks to screen
max_daily_loss = 0.03     # halt new entries if down 3% today
max_drawdown   = 0.15     # circuit breaker: no new entries if down 15% from peak
no_options     = false    # set to true to disable options trading
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

Your portfolio balance, positions, and trade history persist between runs
in `broker/state/`. Nothing resets. The `cash` value in `broker.config`
is only used on the very first run ever.

### Check your portfolio
```bash
python Broker.py --status
```
Shows full portfolio summary including:
- Current positions with unrealised P&L per stock
- Open options positions with Greeks
- Total return vs SPY (beta, alpha, information ratio, upside/downside capture)
- Max drawdown, Sharpe ratio, win rate

### See recent trades
```bash
python Broker.py --trades
```
Shows the last 30 trades with full reasoning — what signal triggered each
buy or sell, the score at time of decision, and the price.

### Override a setting for one run
```bash
python Broker.py --min_score 0.55      # stricter this run only
python Broker.py --penny_pct 0.20      # more penny exposure this run
python Broker.py --no_options          # skip options this run
python Broker.py --no_market_hours     # run outside market hours (testing)
```

### Schedule it
Run once daily after market close. The broker enforces market hours
internally — new entries are blocked outside 9:30am–4:00pm ET.

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
- **Partial take-profit:** At +20% gain, sells 50% of the position and
  locks in profit. The remaining 50% continues running.
- **Full take-profit:** At +45% gain, exits the remaining position entirely.
- **Signal deterioration:** If the composite score of a held stock drops
  below 0.35, the position is sold regardless of P&L.

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

### Options trading
After stock decisions, evaluates top-scored stocks for options:

| Signal condition | Strategy | Why |
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

### Get weekly stock picks
```bash
python Agent.py --mode predict
```
Outputs a ranked table of top picks with portfolio weights and signal
breakdown. Run this Monday morning for the week's recommendations.

```bash
python Agent.py --mode predict --top_k 20      # show top 20 instead of 10
python Agent.py --mode predict --top_n 500     # use smaller universe
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
python Agent.py --mode screen --min_price 1 --max_price 10

# Show more results
python Agent.py --mode screen --penny --screener_top_n 100
```

The screener has no universe size limit — it scores every ticker
individually so it handles the full parquet including sub-$5 stocks.

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
Check if you have a GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Fine-tune on recent data
```bash
python Agent.py --mode finetune
```
Takes the best existing checkpoint and continues training on the most recent
2 years of data. Run this after major market regime changes (crashes, rate
cycles, etc.). Takes ~20 minutes.

```bash
python Agent.py --mode finetune --finetune_steps 30000
```

### Backtest the RL agent vs SPY
```bash
python Agent.py --mode backtest
```
Evaluates the trained RL policy on the held-out test period. SPY is fetched
automatically. Produces a full metrics table and saves a 4-panel chart to
`plots/backtest.png` showing:
- Equity curves (policy vs SPY vs equal-weight)
- Drawdown comparison vs SPY
- Rolling 3-month relative performance
- Rolling 12-month relative performance

### Broker replay backtest
```bash
python Agent.py --mode replay
```
Runs the **actual broker decision logic** over historical data — same
screening, sector allocation, stops, take-profits, and execution costs as
the live broker. This is the honest performance number for the broker, not
the RL agent's idealized backtest.

```bash
python Agent.py --mode replay --replay_years 5     # use 5 years of history
python Agent.py --mode replay --sensitivity        # also run sensitivity sweep
```

The sensitivity sweep runs the replay across 13 parameter combinations
(varying min_score, stop-loss, take-profit, execution spread) to test
whether results are robust or collapse under parameter changes.

### Update data manually
```bash
python Agent.py --mode update
```
Fetches latest prices and scores fresh news headlines. The broker does this
automatically on every run, so you only need this manually if you want to
update without running a full broker cycle.

```bash
python Agent.py --mode update --force_refresh   # re-download last 30 days
```

### Keep data updated automatically (background scheduler)
```bash
python Agent.py --mode schedule
```
Runs continuously in the background:
- 17:00 Mon–Fri: fetch new prices + news, score with FinBERT
- Sunday 20:00: fine-tune model on recent 2 years of data

Logs to `logs/scheduler.log`. Use this if you want the model to keep
improving automatically. Otherwise the broker's built-in auto-update
on each run is sufficient.

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
equity. `python Broker.py --status` shows:

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
  replay.py                   Broker replay backtest + sensitivity sweep
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
That's it. Everything else is automatic.

**Weekly (Monday morning, optional):**
```bash
python Agent.py --mode predict
```
See the model's top picks for the week with signal breakdown.

**Monthly (optional):**
```bash
python Agent.py --mode replay --sensitivity
```
Check whether the broker is actually beating SPY and whether results are
robust across parameter changes.

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
