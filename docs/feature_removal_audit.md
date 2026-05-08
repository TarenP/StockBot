# Feature Removal Audit

Unpromoted feature families remain diagnostic-only until a future promotion decision proves isolated replay edge, enough decision-changing touches, acceptable risk, and replay safety.

Live and paper broker defaults:

```text
earnings_reaction_enabled = false
macro_regime_enabled = false
insider_adjustment_enabled = false
allow_unpromoted_feature_influence = false
llm_sidecar_broker_influence = false
event_sidecar_broker_influence = false
pattern_sidecar_broker_influence = false
```

Diagnostic systems may remain enabled:

```text
llm_sidecar_enabled = true
event_sidecar_enabled = true
pattern_sidecar_enabled = true
macro_shock_dashboard_enabled = true
enable_llm_sidecar_precompute = true
enable_event_sidecar_precompute = true
```

Required behavior:

- Unpromoted features may appear in diagnostics, journals, manifests, memos, and audit artifacts.
- Unpromoted features must not mutate rank score, candidate ordering, target weight, entries, exits, risk controls, or policy promotion.
- Feature-ablation tooling may opt into `allow_unpromoted_feature_influence = true` only for isolated replay experiments.
- A feature may re-enter broker influence only when a `promotion_decision.json` says `promote`.
