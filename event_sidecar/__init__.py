"""Cached market-event and crowd-sentiment diagnostics.

The broker hot path should only consume cached structured features from this
package. Network collection belongs in orchestrator/precompute workflows.
"""

