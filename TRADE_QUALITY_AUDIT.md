# StockBot Trade Quality Audit — Corrected Assessment
Date: 2026-04-24

## Overall: 6.9/10 → targeting 7.5+ after this pass

Good structure, real controls, still leaving obvious edge on the table.

---

## Policy Selection Workflow Update (2026-05-04)

The policy-selection runbook is now executable through the replay matrix layer.
Use it before changing broker defaults for the two active policy families:

- `weak_sleeve`: `weak_sleeve=50%`, `weak_sleeve=25%`, `weak_sleeve=block`, `weak_sleeve=cooldown2`
- `low_price`: `low_price=late_cap`, `low_price=pre_penalty`, `low_price=exclude_high_rank`

### Operating rule

Test one family at a time. Run the weak-sleeve matrix first, freeze or hold its
outcome, then run the low-price matrix with that weak-sleeve outcome fixed. Do
not mix exit-stack changes, new signal families, sentiment changes, or
execution-cost changes into the same first-pass comparison.

### Standard replay setup

Use five rolling one-year windows stepped quarterly where runtime allows. The
minimum acceptable short run is three windows: early, middle, recent. Every
variant inside a family must use the exact same windows.

The matrix runner writes:

- per-window `sensitivity.csv`
- per-window `policy_review.csv`
- per-window `policy_review.json`
- `window_manifest.csv` and `window_manifest.json`
- `winner_stability.csv`
- `aggregate_sensitivity.csv`
- `aggregate_policy_review.csv`
- `aggregate_policy_review.json`
- `summary_table.csv`

### Promotion standard

A policy may replace the current default only if it clears all promotion gates:

- `family_rank == 1`
- minimum incumbent edge
- mechanism-score floor
- confidence gate with small-sample penalties applied
- repeated-window stability
- drawdown guardrail
- turnover guardrail

Allowed decision statuses are:

- `promote`
- `hold_for_more_evidence`
- `reject_mechanism`
- `reject_confidence`
- `reject_drawdown`
- `reject_turnover`
- `reject_insufficient_edge`

`hold_for_more_evidence` is a valid outcome. Do not lower thresholds simply to
force a winner.

### Review priority

Settle the weak-sleeve and low-price defaults with replay evidence before
adding major new feature families such as earnings reactions, macro regimes, or
insider adjustments.

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
