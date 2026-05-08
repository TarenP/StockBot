# Feature Ablation Audit

This repo treats uncertain feature layers as diagnostic until isolated replay
ablations prove they improve the frozen baseline.

Run:

```powershell
python run_feature_ablation.py
```

Quick artifact smoke test:

```powershell
python run_feature_ablation.py --dry-run --n-windows 3 --run-id smoke
```

Artifacts are written under:

```text
experiments/feature_ablation/<run_id>/
```

Required variants:

- `baseline`
- `earnings_only`
- `macro_only`
- `insider_only`
- `llm_sidecar_shadow`
- `event_sidecar_shadow`
- `pattern_sidecar_shadow`

The frozen baseline disables active unpromoted influence:

```text
earnings_reaction_enabled = false
macro_regime_enabled = false
insider_adjustment_enabled = false
llm_sidecar_broker_influence = false
event_sidecar_broker_influence = false
pattern_sidecar_broker_influence = false
```

Diagnostic/precompute layers may stay enabled when they do not alter ranking,
exits, sizing, or portfolio construction.

Promotion requires all gates to pass: edge, winner rate, drawdown, turnover,
mechanism touches, confidence, and replay-safety. If a feature is interesting
but thinly sampled, the correct decision is `hold_for_more_evidence`, not a
live config change.

