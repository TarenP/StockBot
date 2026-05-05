"""Free local-LLM sidecar for StockBot.

The broker hot path must only consume cached structured features from this
package. Network calls, raw document parsing, and Ollama calls belong in
precompute/orchestration flows.
"""

