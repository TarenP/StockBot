# StockBot Trade Quality & Completeness Audit
Date: 2026-04-24

## Overall Assessment: 5.3/10

Solid foundation — universe discipline, risk controls, and ML pipeline are real.
But the decision engine has clear gaps that limit trade quality, and several
implemented features are disabled or underutilized.

---

## 1. ENTRY LOGIC — What's Good, What's Weak

### Good
- Multi-stage filtering: screener → RL ranking → sector caps → correlation blocking → earnings avoidance
- Conviction-based position sizing that scales with market regime
- Sentiment integration with configurable policies (informational / penalize / veto)
- RL Phase 1 ranking uses rank percentiles (stable across shortlist sizes)

### Weak

**A. Composite score is too simplistic (broker/analyst.py)**
The heuristic composite score is a fixed weighted average of 10 signals with no
interaction terms, no regime-specific weighting, and no adaptive thresholds.
A stock with strong momentum but terrible sentiment gets the same score as one
with weak momentum and great sentiment. The weights are hardcoded.

**B. No earnings surprise data**
The system avoids earnings windows (4 days) but never uses actual EPS beats/misses,
guidance changes, or earnings surprise scores. Post-earnings drift is one of the
most documented alpha sources in equities.

**C. No insider trading signals**
No Form 4 data, no unusual insider buying/selling patterns. Insider buying clusters
are a strong signal for small/mid-cap names.

**D. No options flow data**
No unusual call/put volume, skew, or open interest signals. Options flow often
leads price moves by 1-3 days.

**E. Macro data is minimal**
Only VIX level and SPY 20-day return. Missing: yield curve slope, credit spreads,
put/call ratio, advance/decline line, Fed funds rate expectations.

**F. No social sentiment**
Only analyst ratings CSV. No Reddit/Twitter/StockTwits integration. Retail
sentiment can be a useful contrarian or momentum signal.

---

## 2. EXIT LOGIC — The Biggest Trade Quality Gap

### Good
- ATR-adjusted stop losses (7-25% range, per-stock volatility)
- Trailing stops (12% default, regime-adjusted)
- Partial take-profit at +35% (sell 50%, let rest run)
- Signal deterioration tracking with grace period
- RL Phase 2 conviction-drop exits (implemented but disabled)

### Weak

**A. Trailing stop activation is too high (18%)**
Winners need to gain 18% before trailing kicks in. Many mean-reversion trades
never reach this. Should be 10-12% or regime-adjusted.

**B. No time-based exits**
No "exit if no progress in 10 days" or "max hold 60 days" rules. Dead-money
positions sit indefinitely, consuming capital.

**C. No profit-taking on weakness**
If a position is up 50% but signal deteriorates, it still holds until the
trailing stop triggers. Should have a "sell winners on weakness" rule that
exits profitable positions faster when signals turn negative.

**D. Signal exit grace period is rigid (2 cycles)**
Doesn't account for market regime. In choppy markets, weak signals are noise;
in trending markets, they're real. Should be regime-adjusted.

**E. RL Phase 2 exits are disabled**
`rl_phase = 1` in config. The conviction-drop exit logic is fully implemented
but never runs. This is the most impactful quick win.

**F. No volatility-based exits**
High-vol positions aren't exited earlier even if they're underwater. A stock
that doubles its ATR while losing money should be cut faster.

---

## 3. POSITION SIZING — Reasonable but Improvable

### Good
- Conviction scaling: size = max_position_pct × conviction
- Market regime adjustment (larger in bull, smaller in bear)
- Sector/theme/correlation caps prevent concentration

### Weak

**A. No volatility-based sizing**
High-ATR stocks get the same dollar allocation as low-ATR stocks. A 40% vol
biotech gets the same size as a 15% vol utility. Should scale inversely with
realized volatility (risk parity).

**B. No liquidity-based sizing**
Low-volume names get the same allocation as mega-cap liquid names. Should scale
down for names where daily volume < 10x the position size.

**C. Conviction formula is too aggressive for marginal signals**
A score of 0.60 (barely above min_score=0.58) gets conviction ≈ 0.5, sizing ≈ 9%
of equity. This is too large for a marginal signal. Should use a steeper
conviction curve that gives small sizes to marginal signals.

---

## 4. SCREENER — Good Foundation, Room to Improve

### Good
- Bidirectional GRU with attention pooling (good architecture for time-series)
- Regression on percentile rank (not binary classification)
- Heuristic blending with meta-model
- AUC ~0.66, lift ~2.2x (reasonable for stock prediction)

### Weak

**A. Lookback is only 40 days**
Too short for regime detection or longer-term trend signals. Should test 60-90 days.

**B. Forward window is only 20 days**
Misses longer-term trends. Should test 20/60/120 day windows and ensemble.

**C. Heuristic weights are hardcoded**
ret_20d=0.24, ret_5d=0.16, etc. Should be learned or regime-adjusted.

**D. No feature importance tracking**
Doesn't log which features drive the top picks. Makes it hard to debug bad
shortlists or understand regime shifts.

---

## 5. RL INTEGRATION — Conservative but Underutilized

### Good
- Ensemble inference (averages multiple fold checkpoints)
- Rank percentile output (stable across shortlist sizes)
- Padding mask for tickers with insufficient history
- Phase 1 (ranking only) is conservative and safe

### Weak

**A. Phase 2 exits are disabled** (rl_phase=1)
Conviction-drop logic is fully implemented but never runs. This is the single
highest-impact change available — enable it.

**B. Phase 3 (continuous weights) is a stub**
WeightAdapter is defined but never called. The RL model outputs continuous
portfolio weights, but the broker converts them to discrete buy/sell decisions
through a separate heuristic path. This loses information.

**C. No RL retraining schedule**
Model is trained once, never updated with live performance data. The weekly
finetune in the scheduler exists but doesn't incorporate live trade outcomes.

**D. No confidence intervals**
Just a point estimate (rank percentile). No uncertainty quantification. The
broker can't distinguish "high conviction, low uncertainty" from "high
conviction, high uncertainty."

---

## 6. OPTIONS SYSTEM — Exists but Not Production-Ready

### Current State
- Black-Scholes Greeks calculation: **working** (delta, gamma, theta, vega)
- 4 strategies: Long Call, Bull Call Spread, Long Put, Cash-Secured Put
- Accounting: **audit-compliant** (cash reservations, assignment handling)
- Options chain fetching from yfinance: **working**
- Integration with portfolio: **clean**
- **Disabled by default** (`no_options = true`)

### What's Missing for Production

**A. No IV rank / percentile**
Uses market IV from chain but doesn't compare to historical IV. This means
it buys expensive vol and sells cheap vol — the opposite of what you want.

**B. No skew or term structure analysis**
Doesn't exploit put/call skew or choose expiration based on theta decay vs
gamma exposure.

**C. Strategy selection is too simplistic**
Long Call if score ≥ 0.75 + positive sentiment. No consideration of IV rank,
expected move, or risk/reward ratio.

**D. No Greeks-based risk management**
No portfolio-level delta/gamma/vega limits. No hedging logic.

**E. No roll logic**
Doesn't roll expiring positions. Just lets them expire.

**F. No earnings-aware options**
Doesn't avoid earnings dates or use earnings vol expansion for straddles.

**G. No liquidity checks**
Doesn't verify bid-ask spreads or open interest before entering.

### Verdict
The options system is a real implementation (not a stub), but it needs IV rank,
skew analysis, and Greeks-based risk management before it should be enabled.

---

## 7. RISK MANAGEMENT — Too Tight in Some Areas, Too Loose in Others

### Current Settings
| Parameter | Value | Assessment |
|---|---|---|
| max_daily_loss | 2.5% | Too tight — blocks trading after one bad day |
| max_drawdown | 12% | Too tight — normal equity drawdowns hit this |
| max_gross_exposure | 99% | Too high — leaves no cash buffer |
| cash_floor | 1% | Too low — no dry powder for opportunities |
| target_volatility | 22% | Reasonable |
| max_position_pct | 18% | Aggressive for a 20-position portfolio |

### Missing
- No position-level max loss (e.g., "cut any position down 25%")
- No liquidity risk management (bid-ask spread checks)
- No correlation-based portfolio risk (only pairwise, not portfolio-level)
- No tail risk management (no VaR or CVaR limits)

---

## 8. SHADOW EVOLUTION — Good Idea, Limited Scope

### Good
- 1000-genome population with elites, mutation, crossover
- Fast-scoring + weekly full replay validation
- Promotion requires beating baseline by 0.05 Sharpe

### Weak
- Only 12 parameters evolve — missing entry/exit thresholds, feature weights
- Single-objective (Sharpe only) — should also consider drawdown, Calmar
- No constraint enforcement (e.g., take_profit > stop_loss)
- Validation is expensive (hours for 20 genomes)

---

## 9. PRIORITIZED FIX LIST

### Quick Wins (no retraining needed)
1. **Enable RL Phase 2 exits** — change `rl_phase = 2` in broker.config
2. **Lower trailing stop activation** — 18% → 12%
3. **Add time-based exits** — exit positions with no progress after 15 trading days
4. **Relax risk limits** — daily loss 5%, drawdown 20%, cash floor 3%
5. **Steeper conviction curve** — reduce sizing for marginal signals

### Medium Effort (code changes, no retraining)
6. **Volatility-based position sizing** — scale inversely with realized ATR
7. **Profit-taking on weakness** — sell profitable positions when signal deteriorates
8. **Regime-adjusted signal exit grace** — shorter grace in trending markets
9. **Liquidity-based sizing** — scale down for low-volume names
10. **Enable options with IV rank filter** — only trade when IV rank < 30 (selling) or > 70 (buying)

### Larger Effort (retraining or new data)
11. **Earnings surprise scoring** — integrate EPS beat/miss data
12. **Insider trading signals** — Form 4 data integration
13. **Macro features** — yield curve, credit spreads, put/call ratio
14. **Screener lookback extension** — 40 → 60-90 days
15. **RL Phase 3 (continuous weights)** — implement WeightAdapter for production

---

## 10. WHAT TO DO FIRST

If you want better trades starting Monday:
1. Set `rl_phase = 2` in broker.config (enables conviction-drop exits)
2. Set `trailing_activation = 0.120` (lower from 0.180)
3. Set `max_daily_loss = 0.050` (relax from 0.025)
4. Set `max_drawdown = 0.200` (relax from 0.120)
5. Set `cash_floor = 0.030` (raise from 0.010)

These are config-only changes — no code needed, no retraining.
