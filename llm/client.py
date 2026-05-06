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
        model: str = "mistral:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def _is_alive(self) -> bool:
        """Check if Ollama server is still responding."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import re as _re
        import time as _time

        # Wait for server if it just crashed
        for attempt in range(3):
            if self._is_alive():
                break
            logger.warning("Ollama not responding, waiting 10s (attempt %d/3)...", attempt + 1)
            _time.sleep(10)
        else:
            raise ConnectionError("Ollama server is not responding after 3 attempts")

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

        # Try to repair truncated JSON by closing open braces/brackets
        repaired = raw
        open_braces = raw.count("{") - raw.count("}")
        open_brackets = raw.count("[") - raw.count("]")
        if open_braces > 0 or open_brackets > 0:
            # Close any open strings first
            if raw.count('"') % 2 == 1:
                repaired += '"'
            repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Extract first JSON object from response (handles markdown code blocks etc.)
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                # Try repairing the extracted fragment too
                fragment = match.group()
                open_b = fragment.count("{") - fragment.count("}")
                if open_b > 0:
                    if fragment.count('"') % 2 == 1:
                        fragment += '"'
                    fragment += "}" * open_b
                    try:
                        return json.loads(fragment)
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"No valid JSON found in model response: {raw[:200]}")

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return list(response.json().get("embedding") or [])

