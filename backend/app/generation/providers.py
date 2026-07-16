"""The ChatProvider interface — swappable seam #2.

Providers are thin REST wrappers (no vendor SDKs) selected by the
``LLM_PROVIDER`` env var. Adding a provider means adding one class and
one registry entry; nothing downstream changes.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

_TIMEOUT_SECONDS = 60.0
_MAX_ATTEMPTS = 5
_RETRY_STATUS = (429, 503)


def _post_with_retry(url: str, json: dict, headers: dict) -> httpx.Response:
    """POST with exponential backoff on rate-limit/overload responses."""
    response: httpx.Response | None = None
    for attempt in range(_MAX_ATTEMPTS):
        response = httpx.post(url, json=json, headers=headers, timeout=_TIMEOUT_SECONDS)
        if response.status_code not in _RETRY_STATUS:
            return response
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(15 * (attempt + 1))
    assert response is not None
    return response


class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ProviderError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ProviderError(
            f"{name} is not set. Add it to backend/.env (see .env.example)."
        )
    return value


class GeminiProvider:
    def __init__(self, model: str | None = None) -> None:
        self._api_key = _require_env("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def complete(self, system: str, user: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1},
        }
        response = _post_with_retry(
            url, json=body, headers={"x-goog-api-key": self._api_key}
        )
        if response.status_code != 200:
            raise ProviderError(
                f"Gemini API error {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected Gemini response shape: {data}") from exc
        return "".join(part.get("text", "") for part in parts).strip()


class _OpenAICompatibleProvider:
    """Shared implementation for OpenAI-compatible chat completion APIs."""

    base_url: str
    key_env: str
    default_model: str
    model_env: str

    def __init__(self, model: str | None = None) -> None:
        self._api_key = _require_env(self.key_env)
        self.model = model or os.getenv(self.model_env, self.default_model)

    def complete(self, system: str, user: str) -> str:
        response = _post_with_retry(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        if response.status_code != 200:
            raise ProviderError(
                f"{type(self).__name__} API error "
                f"{response.status_code}: {response.text[:500]}"
            )
        return response.json()["choices"][0]["message"]["content"].strip()


class GroqProvider(_OpenAICompatibleProvider):
    base_url = "https://api.groq.com/openai/v1"
    key_env = "GROQ_API_KEY"
    default_model = "llama-3.3-70b-versatile"
    model_env = "GROQ_MODEL"


class OpenAIProvider(_OpenAICompatibleProvider):
    base_url = "https://api.openai.com/v1"
    key_env = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"
    model_env = "OPENAI_MODEL"


_REGISTRY: dict[str, type] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
}


def get_provider(name: str | None = None) -> ChatProvider:
    name = (name or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if name not in _REGISTRY:
        raise ProviderError(
            f"Unknown LLM_PROVIDER {name!r}; choose from {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()
