"""
Anthropic Claude Client for Norns LLM Agent.
Direct HTTP client using standard library (urllib.request) so it works out of the box
with or without the external 'anthropic' PyPI package installed.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


class AnthropicClient:
    """
    Lightweight, robust client for the Anthropic Messages API (Claude 3.5 Sonnet / Claude 3 Opus / Haiku).
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """
        Sends request to Anthropic Claude Messages API and returns the text response.
        """
        if not self.is_configured:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set or invalid. Please set the environment variable "
                "or pass api_key to the AnthropicClient."
            )

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        req = urllib.request.Request(self.API_URL, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content_blocks = result.get("content", [])
                text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                return "".join(text_parts).strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API Error (HTTP {e.code}): {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Anthropic API: {e}") from e
