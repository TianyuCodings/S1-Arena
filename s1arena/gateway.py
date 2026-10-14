"""HTTP access to the Vercel AI Gateway, which serves both Jev and the LLMs."""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

import httpx

from .env import load_env

BASE_URL = "https://ai-gateway.vercel.sh"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

_client: httpx.Client | None = None
_lock = threading.Lock()


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            load_env()
            key = os.environ.get("AI_GATEWAY_API_KEY")
            if not key:
                raise RuntimeError("AI_GATEWAY_API_KEY is not set; add it to .env")
            _client = httpx.Client(
                timeout=httpx.Timeout(120.0, connect=20.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
                headers={
                    "Authorization": f"Bearer {key}",
                    "ai-gateway-protocol-version": "0.0.1",
                    "ai-gateway-auth-method": "api-key",
                },
            )
        return _client


def post(path: str, body: dict[str, Any], headers: dict[str, str] | None = None, retries: int = 6) -> dict[str, Any]:
    """POST JSON with exponential backoff and jitter on rate limits and transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client().post(BASE_URL + path, json=body, headers=headers or {})
            if response.status_code in RETRY_STATUS and attempt < retries:
                time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
                continue
            if response.is_error:
                raise RuntimeError(f"gateway HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        except httpx.TransportError as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
                continue
    raise RuntimeError(f"gateway unreachable: {last_error}")
