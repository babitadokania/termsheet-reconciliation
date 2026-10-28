"""Google Gemini client (free tier via Google AI Studio API key).

- Structured output: the JSON schema of the expected answer is passed as `response_json_schema`, so
  Gemini is constrained to the right shape. If the API rejects the schema (400), the client logs it,
  switches to plain JSON mode and carries on - Pydantic validation downstream is the real gate.
- Temperature 0 for deterministic output.
- Retries with exponential backoff + jitter on rate limits (429), server errors (5xx) and timeouts.
- Client-side throttle so the free-tier requests-per-minute limit is respected.
- Non-retryable errors (bad key, bad request) fail fast with a clear message.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from typing import Any

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

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _status(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _retired_model_replacement(exc: BaseException) -> str | None:
    """404 'model ... is no longer available ... use models/<new>' -> '<new>'."""
    if _status(exc) != 404:
        return None
    m = re.search(r"use models/([A-Za-z0-9.\-_]+)", str(exc))
    return m.group(1).rstrip(".") if m else None


def _no_quota(exc: BaseException) -> bool:
    """429 that will not clear by waiting: model not on the free tier (limit 0) or daily quota used up."""
    text = str(exc)
    return _status(exc) == 429 and ("limit: 0" in text or "PerDay" in text or "per day" in text.lower())


def _model_rank(name: str) -> tuple:
    """Preference order for fallback models: numbered stable Flash (newest first), then Flash-Lite,
    then '-latest' aliases, then everything else (omni / preview / experimental)."""
    m = re.match(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$", name)
    if m:
        return (1 if m.group(2) else 0, -float(m.group(1)), name)
    if name.endswith("-latest"):
        return (2, 0.0, name)
    return (3, 0.0, name)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMResponseFormatError):
        return True  # occasionally the model truncates JSON - a retry usually fixes it
    if _no_quota(exc):
        return False  # waiting won't help - switch model instead
    code = _status(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "servererror", "unavailable"))


class GeminiClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout_s: int = 120,
        min_interval_s: float = 4.0,
        structured_output: bool = True,
        vertexai: bool | None = None,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise LLMError("google-genai is not installed: pip install -r requirements.txt") from exc

        self._types = types
        # Default: the free Gemini Developer API (AI Studio keys - "AIza..." or the newer "AQ..." format).
        # Vertex AI is opt-in (GEMINI_USE_VERTEX=true) because it needs a billing-enabled GCP project.
        self.vertexai = bool(vertexai)
        self._client = genai.Client(
            api_key=api_key, vertexai=self.vertexai, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )
        log.info("Gemini endpoint: %s", "Vertex AI (express mode)" if self.vertexai else "Gemini Developer API")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.structured_output = structured_output
        self.name = f"gemini:{model}"
        self._model_list: list[str] | None = None
        self.max_model_switches = 6
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _config(self, system: str, schema: dict[str, Any] | None):
        kwargs: dict[str, Any] = dict(
            system_instruction=system, temperature=self.temperature, response_mime_type="application/json"
        )
        if schema is not None and self.structured_output:
            kwargs["response_json_schema"] = schema
        return self._types.GenerateContentConfig(**kwargs)

    def _call_once(self, system: str, prompt: str, schema: dict[str, Any] | None) -> Any:
        self._throttle()
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=self._config(system, schema)
            )
        except Exception as exc:
            replacement = _retired_model_replacement(exc)
            if replacement and replacement != self.model:
                log.warning("Gemini model %s is retired; switching to Google's suggested %s", self.model, replacement)
                self.model = replacement
                return self._call_once(system, prompt, schema)
            if schema is not None and self.structured_output and _status(exc) == 400:
                log.warning("Gemini rejected the response schema (%s); switching to plain JSON mode", exc)
                self.structured_output = False
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config(system, None)
                )
            else:
                raise
        text = getattr(response, "text", None)
        if not text:
            reason = None
            with contextlib.suppress(AttributeError, IndexError, TypeError):
                candidates = response.candidates or []
                reason = candidates[0].finish_reason if candidates else None
            raise LLMResponseFormatError(f"Gemini returned no text (finish_reason={reason})")
        return parse_json_response(text)

    def _flash_models(self) -> list[str]:
        """Text 'flash' models this key can use, from the live model list (cached)."""
        if self._model_list is None:
            names: list[str] = []
            try:
                for m in self._client.models.list():
                    name = str(getattr(m, "name", "")).removeprefix("models/")
                    actions = getattr(m, "supported_actions", None) or []
                    if actions and "generateContent" not in actions:
                        continue
                    if "flash" in name and not any(
                        x in name for x in ("image", "tts", "live", "audio", "embed", "thinking")
                    ):
                        names.append(name)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not list Gemini models (%s)", exc)
            self._model_list = sorted(set(names), key=_model_rank)
            log.info("Gemini fallback models available: %s", self._model_list or "none")
        return self._model_list

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        """Call the current model; if Google keeps reporting it overloaded (503/504/500), move to the
        next available Flash model instead of failing."""
        tried: set[str] = set()
        while True:
            try:
                return self._generate_once_with_retries(system, prompt, task, schema)
            except LLMError as exc:
                cause = exc.__cause__ or exc
                overloaded = _status(cause) in (500, 503, 504)
                if not (overloaded or _no_quota(cause) or _status(cause) == 404):
                    raise
                tried.add(self.model)
                nxt = next((m for m in self._flash_models() if m not in tried), None)
                if nxt is None or len(tried) >= self.max_model_switches + 1:
                    raise
                why = "is overloaded" if overloaded else "has no free-tier quota / is unavailable"
                log.warning("Gemini model %s %s; switching to %s", self.model, why, nxt)
                self.model = nxt

    def _generate_once_with_retries(self, system: str, prompt: str, task: str, schema: dict[str, Any] | None) -> Any:
        started = time.monotonic()
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(multiplier=2, min=2, max=60),
                retry=retry_if_exception(_is_retryable),
                before_sleep=before_sleep_log(log, logging.WARNING),
                reraise=False,
            ):
                with attempt:
                    result = self._call_once(system, prompt, schema)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            raise LLMError(f"Gemini call '{task}' failed after {self.max_retries} attempts: {last}") from last
        except Exception as exc:  # non-retryable (auth, invalid request, ...)
            msg = str(exc)
            if "SERVICE_DISABLED" in msg or "has not been used in project" in msg:
                hint = (
                    " - this key's Google Cloud project has the Vertex AI API disabled: enable it via the link above, "
                    "or use a free Gemini API key (AIza...) from https://aistudio.google.com/app/apikey"
                )
            elif _status(exc) in (400, 401, 403):
                hint = " - check GEMINI_API_KEY"
            else:
                hint = ""
            raise LLMError(f"Gemini call '{task}' failed: {exc}{hint}") from exc
        log.info("Gemini '%s' call ok in %.1fs (%s)", task, time.monotonic() - started, self.model)
        return result
