# Post Feature-Ablation Audit

## Purpose

This audit reviews the current StockBot feature-ablation results and defines the next testing standard before any uncertain feature can be promoted into live broker influence.

The current test is sufficient to block promotion. It is not sufficient to permanently reject every feature family.

The key rule remains:

> A feature is beneficial only if it beats the frozen baseline when isolated, across enough replay windows, with real decision-changing touches and acceptable downside behavior.

---

## Current audit result

### Final decision

```text
decision_status = hold_for_more_evidence
promoted_feature = null
```

No feature cleared all promotion gates.

### Practical interpretation

Do not promote any of the tested uncertain features yet.

Current action:

```text
earnings_reaction_enabled = false
macro_regime_enabled = false
insider_adjustment_enabled = false
llm_sidecar_broker_influence = false
event_sidecar_broker_influence = false
pattern_sidecar_broker_influence = false
```

Diagnostic and precompute layers may remain enabled:

```text
llm_sidecar_enabled = true
event_sidecar_enabled = true
pattern_sidecar_enabled = true
macro_shock_dashboard_enabled = true
enable_llm_sidecar_precompute = true
enable_event_sidecar_precompute = true
```

---

## What the current test proved

### 1. The baseline is usable

The frozen baseline produced a clean comparator across all five replay windows.

Baseline summary:

```text
avg_total_return = 0.2001
avg_sharpe = 0.8936
avg_max_drawdown = -0.2434
avg_turnover = 6.8413
avg_win_rate = 0.5505
avg_holding_days = 100.34
```

This is now the benchmark for uncertain-feature testing.

### 2. Macro was the only meaningfully exercised active feature

Macro had many touches and changed sizing/ranking enough to produce different results.

Macro summary:

```text
feature_touch_count = 6866
adjusted_entry_count = 568
adjusted_size_count = 568
wins = 4 / 5
winner_rate = 0.8
avg_total_return = 0.1925
avg_sharpe = 0.8730
avg_max_drawdown = -0.2281
avg_turnover = 7.0110
incumbent_edge = -0.0207
worst_window_return_delta = -0.0821
decision_status = reject_insufficient_edge
```

Macro reduced drawdown modestly but failed the main promotion test because it underperformed the baseline on average policy score and return, increased turnover, and had an unacceptable worst-window hit.

### 3. Earnings, event sidecar, LLM sidecar, and pattern sidecar were not really tested

These variants produced zero decision touches.

A zero-touch result means:

```text
hold_for_more_evidence
```

not:

```text
feature is bad
```

### 4. Insider touched many rows but did not improve outcomes

Insider produced many audit touches but no winning windows and no meaningful performance improvement.

This suggests either:

1. the feature is present but not decision-changing enough,
2. the current implementation is too weak or directionally noisy,
3. the touch audit is overcounting signal presence as real influence, or
4. historical insider coverage is too thin or stale to support replay inference.

---

## What the current test did not prove

### It did not prove macro is useless

It only proved the current macro implementation and current strength are not promotable.

Macro may still be useful as:

```text
risk-off-only adjustment
volatility-scaler-only adjustment
drawdown-guard-only adjustment
weaker sizing nudge
regime-specific exposure brake
```

### It did not prove earnings reaction is useless

Earnings had no decision touches. The next question is whether the data path is missing, the audit is not detecting touches, or the feature genuinely has no eligible opportunities in the tested windows.

### It did not prove LLM/event/pattern sidecars are useless

Those are still diagnostics/shadow systems. They need enough cached historical features and a real shadow-touch study before any influence test.

### It did not prove the test windows are independent

The five windows are useful, but they overlap heavily. This is good for a first screen and bad for final promotion confidence.

---

## Main issues found

## Issue 1: Live config may still enable unpromoted active features

The ablation runner correctly freezes uncertain features off for the baseline, but live broker config should not keep unpromoted active influence enabled unless the explicit intention is paper-only observation.

Recommended default:

```text
earnings_reaction_enabled = false
macro_regime_enabled = false
insider_adjustment_enabled = false
```

Reason:

Live paper trading should not quietly accumulate results from unpromoted features while the audit baseline assumes they are off.

## Issue 2: Touch audit needs stronger mechanism evidence

Current touch output is useful, but it is not yet strong enough for confident promotion.

Needed improvements:

```text
rank_before
rank_after
would_enter_baseline
would_enter_variant
actual_next_5d_return
actual_next_20d_return
true_score_delta
true_weight_delta
source_timestamp
replay_safe
```

The audit must distinguish:

```text
feature exists
feature changed score
feature changed rank
feature changed size
feature changed entry
feature changed exit
feature improved forward return
```

Presence is not the same as influence.

## Issue 3: Macro result is mixed and needs a dedicated sweep

Macro was active enough to deserve deeper testing, but the current version is not promotable.

The most likely problem is blunt sizing adjustment.

Next test should sweep smaller strengths and narrower conditions.

## Issue 4: Sidecar shadows need data coverage before replay influence

LLM/event/pattern sidecars showed no touches. That means the next step is not promotion. The next step is coverage and shadow plumbing.

---

## Required next fixes before deeper testing

### Fix 1: Add true decision-change accounting

Add fields that identify whether a feature changed the actual decision path.

Required booleans:

```text
feature_present
score_changed
rank_changed
size_changed
entry_changed
exit_changed
decision_changed
```

Promotion should require:

```text
decision_changed_count >= 30
```

not merely:

```text
feature_touch_count >= 30
```

### Fix 2: Add forward-return attribution

For every touched ticker/date, compute:

```text
actual_next_5d_return
actual_next_20d_return
actual_next_60d_return
```

Then report:

```text
avg_forward_return_when_feature_positive
avg_forward_return_when_feature_negative
feature_directional_hit_rate
feature_return_spread
```

### Fix 3: Add per-window feature mechanism summary

Each variant should write:

```text
window_feature_mechanism_summary.csv
```

Required columns:

```text
window_id
variant_label
feature_present_count
score_changed_count
rank_changed_count
size_changed_count
entry_changed_count
exit_changed_count
avg_score_delta
avg_weight_delta
avg_next_5d_return
avg_next_20d_return
hit_rate_5d
hit_rate_20d
```

### Fix 4: Normalize feature_touch_rate

Current values like `1373.2` are not interpretable as a rate.

Use:

```text
feature_touch_rate = feature_touch_count / evaluated_candidate_count
```

Also include:

```text
touches_per_window
```

as a separate field.

### Fix 5: Add source leakage checks

Every feature touch should include:

```text
as_of_date
source_timestamp
feature_available_at
replay_safe
leakage_reason
```

Promotion requires:

```text
all replay_safe = true
```

---

## Next testing plan

## Phase 1: Lock the safe baseline

Before deeper testing, make the live/paper default match the audit baseline unless intentionally running a labeled experiment.

Recommended config:

```text
earnings_reaction_enabled = false
macro_regime_enabled = false
insider_adjustment_enabled = false
llm_sidecar_broker_influence = false
event_sidecar_broker_influence = false
pattern_sidecar_broker_influence = false
```

If a feature is intentionally enabled in paper mode, the run must be labeled as an experiment and written to a separate artifact path.

## Phase 2: Run macro sensitivity sweep

Macro had enough touches to justify deeper testing.

Recommended variants:

```text
macro_weight_0.02
macro_weight_0.04
macro_weight_0.06
macro_weight_0.08
macro_risk_off_only
macro_no_bull_boost
macro_drawdown_guard_only
macro_volatility_scaler_only
```

Promotion standard:

```text
positive incumbent_edge
winner_rate >= 0.60
worst_window_return_delta >= -0.05
turnover_degradation <= 0.25
max_drawdown_degradation <= 0.15
decision_changed_count >= 30
```

Expected result to look for:

A weaker or narrower macro feature that keeps the drawdown benefit without dragging average return.

## Phase 3: Repair earnings touch path

Earnings had zero touches. Before rerunning, answer:

```text
Are earnings reaction scores being generated historically?
Are they attached to candidate reports during replay?
Are they visible in score audit rows?
Are they allowed to change rank/size?
Are they filtered out by missing source timestamps?
```

Only rerun earnings after at least one dry-run or toy test proves earnings can touch a decision.

## Phase 4: Reassess insider as diagnostic-only

Insider currently touched many rows but did not improve outcomes.

Next actions:

```text
inspect top positive insider touches
inspect top negative insider touches
compare actual next 20d returns
check source freshness
check whether old/stale Form 4s are being counted too long
```

Do not rerun insider promotion until source freshness and forward-return spread are available.

## Phase 5: Keep sidecars in shadow mode

LLM, event, and pattern sidecars should remain diagnostics-only until they show enough shadow touches.

Required before any influence test:

```text
quality gate passes
manual-review queue reviewed
shadow_touch_count >= 30
source timestamps replay-safe
forward-return attribution exists
no hard buy/sell authority
```

---

## Recommended commands

### First: rerun current audit only after touch fixes

```bash
python run_feature_ablation.py --n-windows 5 --window-years 1 --step-months 3
```

### Better: reduce overlap

```bash
python run_feature_ablation.py --n-windows 8 --window-years 1 --step-months 6
```

### Broader regime test

```bash
python run_feature_ablation.py --n-windows 10 --window-years 2 --step-months 6
```

### Macro-specific sweep, recommended new runner

```bash
python run_macro_ablation_sweep.py --n-windows 8 --window-years 1 --step-months 6
```

---

## Promotion standard going forward

A feature may be promoted only if it clears every gate below.

### Outcome gates

```text
incumbent_edge >= 0.05
winner_rate >= 0.60
winner_windows >= 3 out of 5
avg_policy_score > baseline_avg_policy_score
```

For larger tests:

```text
winner_windows >= 5 out of 8
```

or:

```text
winner_windows >= 6 out of 10
```

### Risk gates

```text
max_drawdown_degradation <= 0.15
turnover_degradation <= 0.25
worst_window_return_delta >= -0.05
```

### Mechanism gates

```text
decision_changed_count >= 30
feature_forward_return_spread > 0
feature_directional_hit_rate >= 0.52
```

### Replay-safety gates

```text
no future-dated features
no live cache leakage
no source timestamp after decision date
no raw LLM calls inside replay decision path
no post-outcome feature construction
```

---

## Current feature status table

| Feature | Current evidence | Decision | Next action |
|---|---|---|---|
| Earnings reaction | Zero touches | Hold | Fix data/touch plumbing |
| Macro regime | Many touches, 4/5 wins, negative edge, bad worst window | Reject current version | Run strength/condition sweep |
| Insider adjustment | Many touches, 0/5 wins, no improvement | Reject/diagnostic | Add freshness and forward-return study |
| Event sidecar | Zero touches | Hold | Improve cached event coverage and shadow audit |
| LLM sidecar | Zero touches | Hold | Review quality reports, require trusted parses |
| Pattern sidecar | Zero touches | Hold | Add pattern forward-return study |

---

## Final conclusion

The current feature-ablation run did its job: it prevented premature promotion.

The right next move is not to add more features. The right next move is to harden the audit so it can tell the difference between:

```text
feature exists
feature touches audit rows
feature changes decisions
feature improves future returns
feature survives risk gates
```

Until then:

```text
Diagnostics on.
Influence off.
Macro deserves a smaller, narrower retest.
Everything else needs better touch evidence before judgment.
```
