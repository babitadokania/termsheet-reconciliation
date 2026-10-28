"""Groq client (free developer API, OpenAI-compatible): https://console.groq.com/keys

Why it is here: a second free LLM provider that needs no billing or cloud project. Same contract as
the Gemini client: JSON mode, temperature 0, retries with backoff on 429/5xx/timeouts.

Robustness specifics:
- Model choice: GROQ_MODEL if set; otherwise the first available model from PREFERRED_MODELS
  (checked against the live /models list, so a retired model never breaks the run).
- Free-tier tokens-per-minute limits: if a request is too large for a model (HTTP 413), the client
  moves on to the next candidate model automatically.
- If JSON mode is rejected (400), it retries once without it; the JSON is still parsed and validated.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import LLMClient, LLMError, LLMResponseFormatError, parse_json_response

log = logging.getLogger(__name__)

BASE_URL = "https://api.groq.com/openai/v1"
# Highest free-tier token-per-minute limits first (long term sheets need ~12k tokens per request).
PREFERRED_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]
RETRYABLE = {408, 429, 500, 502, 503, 504}


class _HTTPError(Exception):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:400]}")
        self.code = code
        self.body = body


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMResponseFormatError):
        return True
    if isinstance(exc, _HTTPError):
        return exc.code in RETRYABLE
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class GroqClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout_s: int = 120,
        min_interval_s: float = 2.0,
        base_url: str = BASE_URL,
        http: httpx.Client | None = None,
    ):
        self._http = http or httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s
        )
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.json_mode = True
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._candidates = [model] if model else self._available_models()
        self.model = self._candidates[0]
        self.name = f"groq:{self.model}"
        log.info("Groq model: %s (fallbacks: %s)", self.model, self._candidates[1:] or "none")

    def _available_models(self) -> list[str]:
        try:
            r = self._http.get("/models")
        except httpx.HTTPError as exc:
            log.warning("Could not list Groq models (%s); using defaults", exc)
            return list(PREFERRED_MODELS)
        if r.status_code in (401, 403):
            raise LLMError(f"Groq rejected the API key (HTTP {r.status_code}) - check GROQ_API_KEY")
        if r.status_code != 200:
            return list(PREFERRED_MODELS)
        live = {m.get("id") for m in r.json().get("data", [])}
        ordered = [m for m in PREFERRED_MODELS if m in live]
        if not ordered:
            raise LLMError(f"None of the preferred Groq models are available; set GROQ_MODEL to one of {sorted(live)}")
        return ordered

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _post(self, system: str, prompt: str) -> Any:
        self._throttle()
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = self._http.post("/chat/completions", json=payload)
        if r.status_code == 400 and self.json_mode and "json" in r.text.lower() and "response_format" in r.text:
            log.warning("Groq rejected JSON mode for %s; retrying without it", self.model)
            self.json_mode = False
            return self._post(system, prompt)
        if r.status_code == 413 or (r.status_code == 429 and "Request too large" in r.text):
            idx = self._candidates.index(self.model)
            if idx + 1 < len(self._candidates):
                self.model = self._candidates[idx + 1]
                log.warning("Request too large for the free-tier limit; switching to %s", self.model)
                return self._post(system, prompt)
        if r.status_code != 200:
            raise _HTTPError(r.status_code, r.text)
        content = r.json()["choices"][0]["message"].get("content")
        return parse_json_response(content)

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        started = time.monotonic()
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(multiplier=2, min=2, max=60),
                retry=retry_if_exception(_retryable),
                before_sleep=before_sleep_log(log, logging.WARNING),
                reraise=False,
            ):
                with attempt:
                    result = self._post(system, prompt)
        except RetryError as exc:
            raise LLMError(
                f"Groq call '{task}' failed after {self.max_retries} attempts: {exc.last_attempt.exception()}"
            ) from exc
        except _HTTPError as exc:
            hint = " - check GROQ_API_KEY" if exc.code in (401, 403) else ""
            raise LLMError(f"Groq call '{task}' failed: {exc}{hint}") from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Groq call '{task}' failed: {exc}") from exc
        log.info("Groq '%s' call ok in %.1fs (%s)", task, time.monotonic() - started, self.model)
        return result
