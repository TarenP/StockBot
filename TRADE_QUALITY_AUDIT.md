# StockBot Trade Quality Audit — Corrected Assessment
Date: 2026-04-24

## Overall: 6.9/10 → targeting 7.5+ after this pass

Good structure, real controls, still leaving obvious edge on the table.

---

## Changes Applied

### Config (broker.config)
| Parameter | Before | After | Why |
|---|---|---|---|
| rl_phase | 1 | 2 | Enable conviction-drop exits — biggest quick win |
| trailing_activation | 0.180 | 0.140 | Engage trailing sooner, fewer round-trips |
| max_daily_loss | 0.025 | 0.050 | Stop blocking trades after one bad day |
| max_drawdown | 0.120 | 0.200 | Normal equity drawdowns hit 12% too easily |
| cash_floor | 0.010 | 0.030 | Keep dry powder for opportunities |
| dead_money_days | (new) | 15 | Exit positions with no progress after 15 days |
| dead_money_min_return | (new) | 0.02 | "No progress" = return below 2% |

### Code (broker/brain.py)
- Dead-money exit logic added: positions held > `dead_money_days` with
  return < `dead_money_min_return` are sold automatically
- Wired through broker.py CLI args and replay kwargs

### What's Still Not Changed (and why)
- Options remain off (`no_options = true`) — stock-side edge needs to be
  stronger first
- Screener architecture unchanged — current AUC 0.66 / lift 2.2x is
  reasonable, improvements should come from exits and sizing first
- RL Phase 3 (continuous weights) still a stub — Phase 2 exits are the
  higher-leverage change
- No new data sources (earnings surprise, insider, options flow) — the
  current signal stack is substantial, the bottleneck is exit quality

---

## What the system already does well
- Broad tradable U.S. equity universe with explicit investability filters
- Explicit sentiment policies (informational / penalize / veto)
- Theme caps and low-price caps beyond just sector limits
- Paper execution costs with tiered spread model
- Replay determinism controls (snapshot freezing, manifests)
- Benchmark integrity handling (SPY as benchmark only)
- 29 features including market context, fundamentals, and regime detection
- Two-stage ML pipeline (screener → RL ranker)
- 1000-genome evolutionary parameter search

## What still needs attention (next pass)
1. Volatility-aware position sizing (scale inversely with realized ATR)
2. Liquidity-aware sizing (scale down for low-volume names)
3. Winner-on-weakness exits (sell profitable positions when signal turns)
4. Regime-adjusted signal exit grace (shorter in trending markets)
5. Steeper conviction curve for marginal signals
