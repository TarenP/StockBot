"""Minimal local Ollama client.

This module is for sidecar/precompute workflows only. Broker decision code must
not call it directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import re as _re
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nRespond with valid JSON only, no other text.\n\n{user_prompt}",
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw = (data.get("response") or "").strip()

        if not raw:
            raise ValueError("Model returned empty response")

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Extract first JSON object from response (handles markdown code blocks etc.)
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as exc:
                logger.warning("Ollama returned invalid JSON: %s", exc)
                raise

        raise ValueError(f"No JSON found in model response: {raw[:200]}")

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return list(response.json().get("embedding") or [])

