# System Audit — Current State

**Last updated:** March 2026
**Status:** Ready for paper trading. Not yet ready for real money.

---

## What is fully implemented

### Core trading pipeline
- Walk-forward training with no data leakage (8-year train / 1-year val / 1-year test)
- Resume-on-interrupt with periodic checkpointing
- Two-stage ML architecture: TickerScorer screener → PortfolioTransformer RL agent
- RL integration with hard model-required mode (aborts cycle if checkpoint missing)
- Phase 1 RL ranking (replaces heuristic composite score for entries)
- Phase 2 RL conviction-drop exits (cross-sectional, uses cycle-level scoring pass)
- Separate `rl_min_score` threshold decoupled from heuristic `min_score`

### Risk management
- ATR-adjusted stop-losses per position (floor + ceiling)
- Partial take-profit at configurable threshold, full exit at take-profit
- Signal deterioration exit (composite score < 0.35)
- Delisted/halted ticker detection (force-close if yfinance returns no data)
- Daily loss limit circuit breaker (halt new entries if down X% today)
- Drawdown circuit breaker (halt new entries if DD from peak > threshold)
- Gross exposure cap (max % of equity in positions)
- Cash floor enforcement (always keep minimum cash buffer)
- Volatility scaling (scale position sizes down when realized vol is elevated)
- Pre-trade risk check before every BUY
- Execution cost model (spread estimate by price tier + market impact)

### Options trading
- Long Call, Bull Call Spread, Long Put, Cash-Secured Put strategies
- Market IV sourced from yfinance options chain (`impliedVolatility` column)
- Liquidity filters: minimum open interest, volume, and max bid-ask spread
- Cash-secured puts: full cash reservation at strike price enforced
- Assignment: correctly deducts cash and adds shares to portfolio
- Exercise: correctly deducts cash and adds/removes shares
- Portfolio-level Greeks: `total_delta`, `total_theta`, `total_reserved_cash`
- Options auto-enable: shadow population proves options value before going live

### Benchmarking and observability
- SPY fetched automatically every cycle and stored in equity curve
- Full benchmark report: beta, alpha, information ratio, upside/downside capture
- Rolling 3-month and 12-month outperformance vs SPY charts
- Per-cycle journal with full decision reasoning
- Live performance chart updated after every run
- `daily_integrity_check()` validates portfolio state each cycle

### Data quality
- Price move cross-verification (>30% move checked against yfinance + Finviz + news)
- Startup validation: price data freshness, sentiment freshness, portfolio consistency
- Portfolio state validation on load: negative cash reset, invalid positions removed
- Options cash reservation consistency check at startup

### Automation
- Single command: `python Broker.py` — everything else is automatic
- Maintenance checks on every run: prices, sentiment, model, parameters
- Shadow population: 1000 genomes evolving in parallel, best promotes to live config
- Historical warm-up: shadow population pre-tuned on 3 years of history after training
- Auto-tuner: parameter grid search + RL ablation gate, writes winners to broker.config
- Options auto-enable: 30 days of shadow proof required before going live
- Duplicate run prevention: lock file prevents running twice in one day
- Screener trained automatically as part of `python Agent.py --mode train`

---

## What is still missing before real money

### 1. Survivorship bias in training data
The parquet contains currently-traded stocks. Stocks that were delisted,
went bankrupt, or were acquired are underrepresented. This inflates backtest
returns. Magnitude is unknown without a point-in-time database.

**Mitigation:** Run paper trading for 90+ days and compare live results to
backtest. If live Sharpe is significantly below backtest Sharpe, survivorship
bias is likely a contributor.

### 2. Agent-broker architecture mismatch
The RL agent was trained to maximize portfolio Sharpe over a fixed universe
with continuous weight rebalancing. The broker uses discrete buy/sell decisions
on a dynamic screened shortlist. The `--mode backtest` numbers reflect the RL
environment, not the broker's actual execution logic.

**Mitigation:** Use `--mode replay` for honest broker performance numbers.
The ablation gate (`--mode ablation`) measures whether RL ranking actually
improves on heuristics in the broker's execution model.

### 3. No intraday data for options
Options pricing with daily OHLCV is imprecise. IV from the chain is correct
at fetch time but stale by the next cycle. Theta decay is computed correctly
but mark-to-market between cycles uses stale IV.

**Mitigation:** Options are disabled by default (`no_options = true`) and
only auto-enable after 30 days of shadow proof. The shadow proof uses the
same daily data, so the bar is consistent.

### 4. Point-in-time correctness in sentiment
Headlines published after market close are used in the same day's features.
In live trading, those headlines arrive after the close and should only
affect the next day's decisions. This is a subtle look-ahead leak in training.

**Mitigation:** Small effect in practice since most headlines are intraday.
Monitor whether live sentiment signals match backtest expectations.

### 5. No correlation cap between positions
The broker can hold 10 highly correlated tech stocks. This is not 10
independent bets — it's concentrated sector exposure with extra steps.
The sector cap (`max_sector`) partially addresses this but doesn't measure
pairwise correlation.

**Mitigation:** The sector cap (default 25%) limits single-sector exposure.
The shadow population evolves `max_sector` automatically. Monitor sector
concentration in `--status` output.

---

## Paper trading checklist

Before committing real money, complete all of these:

- [ ] 90 days of paper trading with documented results
- [ ] Paper trading Sharpe > 1.0 annualised
- [ ] Paper trading beats SPY on total return
- [ ] Paper trading max drawdown < 20%
- [ ] Downside capture ratio < 0.80
- [ ] Shadow population has run at least 2 full promotion cycles
- [ ] RL ablation gate has passed at least once (if using RL mode)
- [ ] Options shadow has beaten baseline for 30 days before enabling options
- [ ] Reviewed at least 10 losing trades to understand failure modes
- [ ] Verified startup validation catches stale data correctly
- [ ] Tested disaster recovery: kill process mid-cycle, restart, verify state

---

## Real money checklist

In addition to the paper trading checklist:

- [ ] 90+ days of paper trading complete
- [ ] Live results within 20% of replay backtest Sharpe (survivorship bias check)
- [ ] Position size limits appropriate for account size (no single position > 10%)
- [ ] Maximum total options exposure < 10% of equity (enforced by broker)
- [ ] Tax implications understood (frequent trading = short-term capital gains)
- [ ] Emergency manual override tested: can you close all positions manually?
- [ ] Broker API integration tested with paper account if using real broker
- [ ] Understood that past performance does not guarantee future results

---

## Architecture notes for developers

### Two-stage ML pipeline
```
11,500 tickers
    ↓  TickerScorer (screener) — per-ticker binary classifier, trained on forward returns
~50-100 shortlist
    ↓  PortfolioTransformer (RL agent) — cross-sectional, portfolio weights via PPO
Final ranked candidates → broker execution
```

### RL integration design
- `_assert_model_available()` validates checkpoint exists and has required keys.
  Does NOT check n_assets vs shortlist size — the model accepts dynamic asset lists.
- `_rl_exit_checks()` uses the cycle-level `rl_scores` Series (not per-ticker inference)
  so held positions are evaluated in the same cross-sectional context as entries.
- `rl_min_score` (default 0.0) is separate from `min_score` — RL path uses top-k
  ranking by default with no absolute floor.

### Shadow population
- 1000 genomes, daily fast-score (~seconds), weekly full replay validation of top 20
- Promotion requires beating baseline by 0.05 Sharpe
- After promotion: winner mutates into child, old live config re-enters pool (swap)
- Options sub-population (100 genomes): options go live after 30 days of proof
- Historical warm-up: 5 generations on 3 years of history after initial training

### broker.config is managed by the system
After initial training, `broker.config` is written by the auto-tuner and shadow
promotion logic. The only values you should set manually are `cash` and
`max_positions`. Everything else is data-driven.
