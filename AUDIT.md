# System Audit: Gap Analysis & Implementation Roadmap

**Auditor perspective:** Senior quant engineer, trading systems architect, risk reviewer.
**Assumption:** This system could lose real money. Every gap is evaluated on that basis.

---

## 1. Executive Summary

This system has a solid foundation — walk-forward training, sentiment integration,
ATR-adjusted stops, sector-aware allocation, options support, and data validation.
However, it has critical gaps that would cause it to fail in paper trading and
potentially cause significant losses in live trading.

The three most dangerous problems right now:

1. **The RL agent and the broker are completely different decision processes.**
   You trained one thing and deployed another. The backtest does not reflect
   what the broker actually does. Any performance numbers from the backtest
   are not predictive of broker performance.

2. **There is no SPY benchmark anywhere in the broker's live operation.**
   You have no way to know if the broker is adding value or just riding the
   market. A system that returns +15% in a year where SPY returned +25% is
   a failure, not a success.

3. **The options accounting is broken.** Cash-secured puts do not actually
   reserve cash. Assignment does not update stock positions. Exercise does
   not update cash. The options P&L is computed using a constant IV proxy
   instead of market IV. This will produce incorrect equity calculations
   and potentially allow the broker to over-allocate capital.

---

## 2. What Already Exists and Is Solid

- Walk-forward training with no data leakage (genuine strength)
- Resume-on-interrupt with periodic checkpointing
- ATR-adjusted stop-losses per position (correct approach)
- Partial profit-taking at +20%, full exit at +45%
- Sector-aware allocation with quadratic diversification penalty
- Data quality cross-verification for suspicious price moves (>30%)
- Earnings calendar avoidance
- FinBERT sentiment with 9 derived features including surprise and acceleration
- Dynamic universe discovery (Finviz + Yahoo trending)
- Persistent portfolio state across restarts
- Defined-risk options strategies only (no naked selling)
- Per-cycle journal with equity curve tracking

---

## 3. Critical Missing Features

### 3A. SPY Benchmark — Completely Absent from Live Broker

The backtest accepts `spy_rets` as an optional parameter that defaults to None.
In `run_backtest_mode` in Agent.py, SPY is never fetched. In the broker, SPY
is never referenced at all.

This is not a cosmetic issue. Without SPY comparison you cannot answer:
- Is this system generating alpha or just beta?
- Should I just buy SPY instead?
- Is the system's risk-adjusted return better than the index?

**Required metrics vs SPY:**
- Beta: `cov(portfolio_rets, spy_rets) / var(spy_rets)`
- Alpha (Jensen's): `ann_return - (rf + beta * (spy_ann_return - rf))`
- Information ratio: `mean(portfolio_rets - spy_rets) / std(portfolio_rets - spy_rets) * sqrt(252)`
- Upside capture: `mean(portfolio_rets[spy_rets > 0]) / mean(spy_rets[spy_rets > 0])`
- Downside capture: `mean(portfolio_rets[spy_rets < 0]) / mean(spy_rets[spy_rets < 0])`
- Rolling 3/6/12-month relative performance vs SPY

Beating equal-weight is not enough. Equal-weight of 500 liquid US stocks
is itself a form of market exposure. You need to prove you beat the index
on a risk-adjusted basis after costs.

### 3B. Agent-Broker Architecture Mismatch

The RL environment (`pipeline/environment.py`) and the broker (`broker/brain.py`)
are completely different systems:

| Dimension | RL Environment | Broker |
|-----------|---------------|--------|
| Action space | Continuous weights over fixed 500-asset universe | Discrete buy/sell on screened candidates |
| Position sizing | Softmax over all assets simultaneously | Conviction-scaled per-stock allocation |
| Universe | Fixed at training time | Dynamic, changes every cycle |
| Rebalancing | Every step (daily) | Only when signal triggers |
| Transaction costs | 10 bps flat on weight change | Not modeled in broker |
| Stop-losses | None | ATR-adjusted |
| Partial exits | None | At +20% |
| Options | None | Yes |
| Sector constraints | None | Yes |
| Earnings avoidance | None | Yes |
| Cash handling | Cash is one asset in the weight vector | Separate cash balance |

The backtest in `pipeline/backtest.py` runs the RL environment, not the broker
logic. So the backtest numbers tell you how the RL policy performs in its
idealized environment — not how the broker will perform in practice.

This is the most dangerous architectural flaw. You are training one system
and deploying a different one, with no validation that they produce similar
outcomes.

### 3C. Options Accounting Is Broken

**Cash-secured puts do not reserve cash:**
In `options.py`, `_cash_secured_put` calculates `cash_per_contract = strike * 100`
but this is only used to determine `n_contracts`. The actual cash reservation
never happens — `portfolio.cash` is not reduced by `strike * 100 * n_contracts`
when the short put is opened. Only the premium received is credited.

This means the broker can sell puts requiring $50,000 in cash backing while
only having $5,000 available. If assigned, it cannot buy the stock.

**Assignment does not update stock positions:**
In `OptionsBook.check_expirations`, when a short put is assigned, the code
logs the assignment but does not:
- Deduct `strike * contracts * 100` from cash
- Add `contracts * 100` shares to `portfolio.positions`

The stock just appears in the log and nowhere else.

**Exercise does not update cash:**
When a long call is exercised ITM, the code logs the intrinsic value but does
not actually buy the shares or deduct cash.

**IV is an ATR proxy, not market IV:**
`iv = min(max(atr_pct * (252 ** 0.5), 0.10), 2.0)` — this is a rough
approximation. For options valuation, you should use the implied volatility
from the options chain itself (`impliedVolatility` column in yfinance).
Using ATR-derived IV will systematically misprice options.

**Option P&L uses stale IV:**
`current_value` uses `self.greeks.get("iv", 0.30)` — the IV at open time.
As IV changes (e.g., IV crush after earnings), the P&L calculation becomes
increasingly wrong.

### 3D. No Portfolio-Level Risk Engine

The system has per-position risk controls (ATR stops, take-profits) but no
portfolio-level controls:

- No maximum daily loss limit (e.g., halt if portfolio drops >3% in one day)
- No maximum drawdown circuit breaker (e.g., go to cash if DD > 15%)
- No gross exposure limit (sum of all position values / equity)
- No volatility targeting (scale positions down when realized vol is high)
- No correlation cap (can hold 10 highly correlated tech stocks)
- No turnover limit (can churn the entire portfolio every cycle)
- No cash floor (can be 100% invested with no buffer)

### 3E. No Kill-Switches or Safety Systems

The broker runs indefinitely with no conditions that halt trading:

- No circuit breaker after N consecutive losing trades
- No halt if sentiment feed is stale (currently warns but continues trading)
- No halt if price feeds disagree (validator rejects bad data but doesn't pause)
- No halt if model checkpoint is missing or corrupted
- No "do not trade" enforcement outside market hours
- No duplicate order prevention (if the scheduler fires twice, two cycles run)
- No startup validation (broker starts even if portfolio state is inconsistent)

### 3F. Execution Realism Is Missing

The broker executes all trades at `last_price` with no friction:

- No bid-ask spread (for penny stocks this can be 5-10% of price)
- No slippage (especially critical for penny stocks with thin books)
- No partial fills (assumes 100% fill at target price)
- No market impact (large orders move price)
- No overnight gap handling (position opened at close, gapped down at open)
- No trading halt handling (suspended tickers still get price updates)
- No delisted ticker handling (positions in delisted stocks are never closed)

The RL environment uses 10 bps flat transaction cost. The broker uses nothing.
These are not the same system.

---

## 4. Important But Secondary Missing Features

### 4A. Validation Methodology Weaknesses

- **Hyperparameter leakage:** `min_score=0.60`, `stop_loss_pct_floor=0.07`,
  `partial_profit_pct=0.20` were chosen by the developer. If these were tuned
  by looking at backtest results, they are overfit. There is no sensitivity
  analysis showing performance across a range of parameter values.

- **Universe selection bias:** The universe is selected based on recent liquidity.
  Stocks that became liquid recently may have done so because they performed well.
  This introduces a subtle look-ahead bias.

- **No regime coverage analysis:** The backtest covers whatever time period the
  test fold happens to land on. There is no explicit analysis of performance
  during 2008, 2020 COVID crash, 2022 rate hike cycle, etc.

- **No Monte Carlo or bootstrap confidence intervals:** A single backtest number
  has no error bars. You don't know if Sharpe=1.43 is statistically different
  from Sharpe=0.81 (equal-weight) given the sample size.

- **Survivorship bias:** The parquet contains currently-traded stocks. Stocks
  that were delisted, went bankrupt, or were acquired are likely underrepresented.
  This inflates backtest returns.

### 4B. Data Quality Gaps

- No corporate actions handling: stock splits will cause apparent 50% price
  drops that the validator will flag as suspicious but may confirm as real
  (yfinance adjusts, Finviz may not, creating a false disagreement).

- No point-in-time correctness: sentiment data is joined by date, but a
  headline published at 4pm is used in the same day's features. In live
  trading, that headline arrives after the close. This is a subtle
  look-ahead leak in the training data.

- No duplicate headline detection beyond exact string matching. The same
  story syndicated across multiple outlets will be counted multiple times,
  inflating sentiment scores.

### 4C. Missing Observability

- No feature importance or signal attribution (which features drive decisions?)
- No P&L attribution by signal (is momentum or sentiment driving returns?)
- No trade-level post-mortem (why did this trade lose money?)
- No model drift detection (are feature distributions shifting?)
- No health dashboard showing data freshness, model status, portfolio state

---

## 5. Why SPY Comparison Must Be a First-Class Benchmark

Beating equal-weight is a low bar. Equal-weight of 500 liquid US stocks has
high correlation with SPY (typically 0.85-0.95). If SPY is up 25%, equal-weight
is probably up 20-22%. Beating equal-weight by 5% in that environment means
you underperformed SPY.

The correct question is: **does this system generate alpha above SPY after
all costs?**

Required benchmark metrics (implement in `pipeline/backtest.py` and
`broker/journal.py`):

```python
def benchmark_vs_spy(portfolio_rets, spy_rets, rf_rate=0.05/252):
    n = min(len(portfolio_rets), len(spy_rets))
    p = portfolio_rets[:n]
    s = spy_rets[:n]

    beta  = np.cov(p, s)[0,1] / np.var(s)
    alpha = (p.mean() - rf_rate) - beta * (s.mean() - rf_rate)
    alpha_ann = alpha * 252

    active_rets = p - s
    ir = active_rets.mean() / (active_rets.std() + 1e-9) * np.sqrt(252)

    up_mask   = s > 0
    down_mask = s < 0
    up_cap    = p[up_mask].mean()   / s[up_mask].mean()   if up_mask.any()   else np.nan
    down_cap  = p[down_mask].mean() / s[down_mask].mean() if down_mask.any() else np.nan

    return {
        "beta": beta, "alpha_ann": alpha_ann,
        "information_ratio": ir,
        "upside_capture": up_cap, "downside_capture": down_cap,
        "tracking_error": active_rets.std() * np.sqrt(252),
        "beats_spy": (np.prod(1+p) > np.prod(1+s)),
    }
```

SPY data should be fetched automatically via yfinance at backtest time and
stored alongside the equity curve in the broker journal.

---

## 6. Architecture Mismatches Between Agent, Backtest, and Broker

The fundamental problem is that three different execution models exist:

**Training environment** (`pipeline/environment.py`):
- Fixed universe, continuous weight rebalancing, Sharpe reward, 10 bps TC

**Backtest** (`pipeline/backtest.py`):
- Runs the RL policy in the training environment — same fixed universe,
  same continuous rebalancing. This is internally consistent with training
  but has nothing to do with how the broker operates.

**Broker** (`broker/brain.py`):
- Dynamic universe, discrete buy/sell decisions, conviction sizing,
  ATR stops, partial exits, sector constraints, options, earnings avoidance.
  No transaction costs modeled.

**The fix:** Create a unified `ExecutionEngine` class that both the backtest
and the broker use. The backtest should simulate the broker's actual decision
logic — screener → research → sector check → conviction sizing → stops —
not the RL environment's continuous rebalancing.

Until this is done, backtest numbers are not predictive of broker performance.

---

## 7. Missing Risk and Execution Infrastructure

### Portfolio Risk Engine (implement in `broker/risk.py`):

```python
class PortfolioRiskEngine:
    def __init__(self,
        max_daily_loss:     float = 0.03,   # halt if down 3% today
        max_drawdown:       float = 0.15,   # go to cash if DD > 15%
        max_gross_exposure: float = 0.95,   # max 95% invested
        target_volatility:  float = 0.15,   # annualised vol target
        max_correlation:    float = 0.70,   # max pairwise correlation
        max_turnover_daily: float = 0.20,   # max 20% portfolio turnover/day
        cash_floor:         float = 0.05,   # always keep 5% cash
    ): ...

    def check_pre_trade(self, decision, portfolio) -> tuple[bool, str]:
        """Return (allowed, reason). Called before every trade."""

    def check_portfolio_health(self, portfolio, spy_price) -> str:
        """Return 'ok', 'warning', or 'halt'. Called each cycle."""

    def scale_position_for_vol(self, alloc_value, realized_vol) -> float:
        """Scale down position size when vol is elevated."""
        vol_scalar = min(1.0, self.target_volatility / (realized_vol + 1e-9))
        return alloc_value * vol_scalar
```

### Execution Model (implement in `broker/execution.py`):

```python
def estimate_execution_cost(
    ticker: str,
    shares: float,
    price: float,
    avg_daily_volume: float,
    is_penny: bool,
) -> float:
    """
    Estimate total execution cost including spread and market impact.
    Returns cost as fraction of trade value.
    """
    # Bid-ask spread estimate
    if is_penny:
        spread_pct = 0.02   # 2% for penny stocks
    elif price < 20:
        spread_pct = 0.005  # 0.5% for low-price stocks
    else:
        spread_pct = 0.001  # 0.1% for liquid stocks

    # Market impact (square-root model)
    participation_rate = (shares * price) / (avg_daily_volume * price + 1e-9)
    impact_pct = 0.1 * np.sqrt(participation_rate)

    return spread_pct + impact_pct
```

---

## 8. Missing Options Infrastructure

### Fix cash-secured put accounting:

```python
# In OptionsBook.open():
if contract.position == "short" and contract.option_type == "put":
    # Reserve cash = strike * 100 * contracts
    cash_reserved = contract.strike * 100 * contract.contracts
    if cash_reserved > cash_available:
        return False, 0.0
    # Cash delta = premium received - cash reserved
    cash_delta = contract.premium_paid * contract.contracts * 100 - cash_reserved
    self._cash_reserved[contract.contract_key] = cash_reserved
```

### Fix assignment handling:

```python
# In OptionsBook.check_expirations(), short put assignment:
if contract.option_type == "put" and spot < contract.strike:
    shares_to_buy = contract.contracts * 100
    cost = contract.strike * shares_to_buy
    # Actually update portfolio
    portfolio.cash -= cost
    portfolio.buy(contract.ticker, shares_to_buy, contract.strike,
                  f"Assigned from short put ${contract.strike}")
    # Release reserved cash
    self._cash_reserved.pop(contract.contract_key, None)
```

### Use market IV instead of ATR proxy:

```python
# In analyse_options(), replace:
iv = min(max(atr_pct * (252 ** 0.5), 0.10), 2.0)

# With:
iv_col = chain["calls"]["impliedVolatility"].median()
iv = float(iv_col) if pd.notna(iv_col) and iv_col > 0 else atr_pct * (252 ** 0.5)
iv = np.clip(iv, 0.05, 3.0)
```

### Add options liquidity filters:

```python
# Minimum open interest and volume before trading
MIN_OPTION_OI     = 100    # open interest
MIN_OPTION_VOLUME = 10     # daily volume
MAX_SPREAD_PCT    = 0.15   # max bid-ask spread as % of mid

liquid_calls = calls[
    (calls["openInterest"].fillna(0) >= MIN_OPTION_OI) &
    (calls["volume"].fillna(0) >= MIN_OPTION_VOLUME) &
    ((calls["ask"] - calls["bid"]) / ((calls["ask"] + calls["bid"]) / 2 + 1e-9) <= MAX_SPREAD_PCT)
]
```

### Track portfolio-level options Greeks:

```python
@property
def portfolio_delta(self) -> float:
    """Net delta exposure across all options (in share equivalents)."""
    return sum(
        c.greeks.get("delta", 0) * c.contracts * 100 *
        (1 if c.position == "long" else -1)
        for c in self.positions.values()
    )

@property
def portfolio_theta(self) -> float:
    """Daily theta decay in dollars."""
    return sum(
        c.greeks.get("theta", 0) * c.contracts * 100 *
        (1 if c.position == "long" else -1)
        for c in self.positions.values()
    )
```

---

## 9. Missing Validation and Observability Infrastructure

### Data freshness checks (add to broker startup):

```python
def validate_startup(portfolio, parquet_path, sentiment_path):
    errors = []

    # Check price data freshness
    df = pd.read_parquet(parquet_path)
    last_price_date = pd.to_datetime(df.index).max().date()
    days_stale = (datetime.today().date() - last_price_date).days
    if days_stale > 3:
        errors.append(f"PRICE DATA STALE: {days_stale} days old")

    # Check sentiment freshness
    sent = pd.read_csv(sentiment_path, usecols=["date"])
    last_sent_date = pd.to_datetime(sent["date"]).max().date()
    sent_stale = (datetime.today().date() - last_sent_date).days
    if sent_stale > 7:
        errors.append(f"SENTIMENT DATA STALE: {sent_stale} days old")

    # Check portfolio state consistency
    if portfolio.cash < 0:
        errors.append(f"NEGATIVE CASH: ${portfolio.cash:.2f}")

    total_position_value = sum(
        p["shares"] * p["last_price"] for p in portfolio.positions.values()
    )
    if total_position_value > portfolio.equity * 1.01:
        errors.append("POSITION VALUE EXCEEDS EQUITY (accounting error)")

    return errors
```

### Daily integrity report (add to journal):

```python
def daily_integrity_check(portfolio, spy_price_today, spy_price_start):
    spy_ret_today = (spy_price_today / spy_price_start) - 1
    portfolio_ret = portfolio.total_return

    report = {
        "date":           datetime.today().date().isoformat(),
        "equity":         portfolio.equity,
        "cash":           portfolio.cash,
        "n_positions":    len(portfolio.positions),
        "n_options":      len(portfolio.options.positions),
        "total_return":   portfolio_ret,
        "spy_return":     spy_ret_today,
        "alpha_vs_spy":   portfolio_ret - spy_ret_today,
        "beating_spy":    portfolio_ret > spy_ret_today,
        "cash_pct":       portfolio.cash / portfolio.equity,
    }
    return report
```

---

## 10. Paper Trading Readiness Checklist

Before running this system in paper trading, every item below must be complete:

- [ ] SPY benchmark fetched automatically and compared every cycle
- [ ] Agent-broker architecture mismatch resolved (unified execution engine)
  OR explicit acknowledgment that backtest numbers do not predict broker performance
- [ ] Options cash reservation fixed (short puts actually reserve cash)
- [ ] Options assignment fixed (actually updates stock positions and cash)
- [ ] Options exercise fixed (actually updates cash)
- [ ] Market IV used instead of ATR proxy for options valuation
- [ ] Portfolio-level daily loss limit implemented and tested
- [ ] Portfolio-level drawdown circuit breaker implemented and tested
- [ ] Startup validation checks (data freshness, portfolio consistency)
- [ ] Kill-switch for stale data (halt trading if price data > 3 days old)
- [ ] Kill-switch for stale sentiment (warn prominently if > 7 days old)
- [ ] Duplicate order prevention (idempotent cycle execution)
- [ ] Market hours enforcement (no trades outside 9:30am-4pm ET)
- [ ] Execution cost model (at minimum: spread estimate for penny stocks)
- [ ] Delisted ticker handling (close positions in tickers that stop trading)
- [ ] Basic logging of every decision with timestamp, price, and reason
- [ ] Equity curve saved with SPY comparison at every cycle
- [ ] At least 30 days of paper trading before any real money consideration

---

## 11. Real-Money Readiness Checklist

In addition to everything in the paper trading checklist:

- [ ] Minimum 90 days of paper trading with documented results
- [ ] Paper trading Sharpe > 1.0 annualised
- [ ] Paper trading beats SPY on risk-adjusted basis (alpha > 0)
- [ ] Paper trading max drawdown < 20%
- [ ] Downside capture ratio < 0.80 (lose less than SPY in down markets)
- [ ] Full broker-backtest parity (same execution logic in both)
- [ ] Sensitivity analysis: performance stable across ±20% parameter perturbation
- [ ] Regime analysis: performance documented in at least one bear market period
- [ ] Slippage model validated against actual paper fills
- [ ] Options accounting audited and verified correct
- [ ] Portfolio-level risk engine with all circuit breakers tested
- [ ] Disaster recovery tested (kill process, restart, verify state is correct)
- [ ] Position size limits appropriate for account size (no single position > 10%)
- [ ] Maximum total options exposure < 10% of equity enforced
- [ ] Penny stock exposure documented and risk understood
- [ ] Tax implications understood (short-term capital gains on frequent trading)
- [ ] Broker API integration (if using real broker) tested with paper account first
- [ ] Emergency manual override documented and tested

---

## 12. Prioritized Implementation Roadmap

### Tier 1 — Required Before Paper Trading

**1. Fix options accounting** (2-3 days)
- Cash reservation for short puts
- Assignment updates stock positions and cash
- Exercise updates cash
- Use market IV from chain
- Why it matters: current accounting will produce incorrect equity values
  and allow over-allocation. This is a money-losing bug.

**2. Add SPY benchmark to broker journal** (1 day)
- Fetch SPY daily via yfinance at each cycle
- Store alongside equity curve
- Compute beta, alpha, information ratio, upside/downside capture
- Print in `--status` output
- Why it matters: without this you cannot evaluate whether the system works.

**3. Portfolio-level daily loss limit and drawdown circuit breaker** (1 day)
- Halt new entries if portfolio down >3% today
- Go to cash if drawdown from peak exceeds 15%
- Why it matters: without this, a bad model can lose everything before you notice.

**4. Startup validation** (half day)
- Check data freshness before trading
- Check portfolio state consistency
- Refuse to start if critical checks fail
- Why it matters: prevents trading on stale data or corrupted state.

**5. Market hours enforcement** (half day)
- Do not execute trades outside 9:30am-4pm ET on trading days
- Why it matters: yfinance prices outside market hours are unreliable.

**6. Execution cost model for penny stocks** (1 day)
- Estimate bid-ask spread based on price tier
- Apply to all trades in the broker
- Why it matters: penny stock spreads of 2-5% make many apparent opportunities
  unprofitable after execution costs.

### Tier 2 — Required Before Real Money

**7. Unified execution engine** (1 week)
- Single `ExecutionEngine` class used by both backtest and broker
- Backtest simulates broker logic, not RL environment
- Why it matters: current backtest numbers are not predictive of broker performance.

**8. Full benchmark framework** (2-3 days)
- Rolling 3/6/12-month relative performance vs SPY
- Regime-tagged performance (bull/bear/sideways/crisis)
- Bootstrap confidence intervals on Sharpe
- Why it matters: need statistical evidence of outperformance, not just a number.

**9. Correlation and concentration risk** (1 day)
- Cap pairwise correlation between positions
- Track factor exposures (market beta, size, momentum)
- Why it matters: 10 tech stocks in a portfolio is not 10 independent bets.

**10. Delisted and halted ticker handling** (1 day)
- Detect when a held ticker stops trading
- Force-close position at last known price
- Why it matters: positions in dead stocks will never be closed otherwise.

**11. Sensitivity analysis** (2 days)
- Grid search over key parameters (min_score, stop_loss, take_profit)
- Verify performance is stable, not a knife-edge optimum
- Why it matters: overfit parameters will fail out-of-sample.

**12. 90-day paper trading period** (90 days)
- Document every trade, every cycle
- Compare weekly to SPY
- Why it matters: paper trading reveals execution issues that backtests miss.

### Tier 3 — Useful But Non-Critical

**13. Feature importance and signal attribution**
- Which of the 19 features actually drive decisions?
- Is sentiment actually adding value over pure technicals?

**14. Post-trade analysis**
- Automated review of losing trades
- Pattern detection in bad trades

**15. Options Greeks portfolio dashboard**
- Net delta, gamma, theta, vega across all options
- Alert when net delta exceeds threshold

**16. Monte Carlo resampling of backtest**
- Bootstrap confidence intervals on all metrics
- Permutation test for statistical significance

**17. Intraday data for options**
- Options pricing is meaningless with daily OHLCV
- Need at least hourly data for options positions

---

## 13. The 5 Highest-Value Changes to Build Next

**1. Fix options accounting** — this is a correctness bug that will cause
incorrect equity calculations and potentially allow the broker to commit
more capital than it has. Fix it before running a single cycle with options enabled.

**2. Add SPY to the broker journal** — one yfinance call per cycle, store
alongside equity. Without this you are flying blind. You cannot evaluate
the system at all.

**3. Daily loss limit circuit breaker** — one check at the start of each cycle:
`if today_loss > 0.03 * equity: skip_new_entries()`. This single guard
prevents a runaway loss scenario.

**4. Startup validation** — check data freshness and portfolio consistency
before every run. Refuse to trade on stale data. This prevents the most
common operational failure mode.

**5. Execution cost model** — add a `spread_cost` estimate to every trade,
especially penny stocks. A 2% spread on a penny stock means you need a 4%
move just to break even. Many of the screener's penny stock picks will be
unprofitable after this adjustment, which is important to know before
committing real capital.

---

*This audit reflects the state of the codebase as of the review date.
All items marked as critical should be resolved before paper trading begins.
No real money should be committed until the Tier 2 checklist is complete
and 90 days of documented paper trading results are available.*
