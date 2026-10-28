"""LLM provider layer: one interface, a Gemini implementation, an offline mock, replay and a disk cache.

Providers
- gemini : live Google Gemini API (needs GEMINI_API_KEY)
- groq   : live Groq API, open models (needs GROQ_API_KEY) - free, no billing / cloud project needed
- replay : recorded LLM responses from recordings/ - real LLM output, no key, no network
- mock   : deterministic offline rules (regex extraction + rule-based adjudication)
- auto   : gemini if its key is set, else groq if its key is set, else replay if recordings exist, else mock
"""

from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMClient, LLMError
from .cache import CachedLLMClient, ReplayClient

log = logging.getLogger(__name__)


def _has_recordings(settings: Settings) -> bool:
    return settings.recordings_dir.exists() and any(settings.recordings_dir.glob("*.json"))


def resolve_provider(settings: Settings) -> str:
    if settings.llm_provider != "auto":
        return settings.llm_provider
    if settings.gemini_api_key:
        return "gemini"
    if settings.groq_api_key:
        return "groq"
    return "replay" if _has_recordings(settings) else "mock"


def build_llm_client(settings: Settings) -> LLMClient:
    provider = resolve_provider(settings)
    client: LLMClient

    if provider == "replay":
        if not _has_recordings(settings):
            raise LLMError(
                f"--provider replay needs recorded responses in {settings.recordings_dir} "
                "(create them with: python run_pipeline.py --provider groq --record)"
            )
        replay = ReplayClient(settings.recordings_dir)
        log.info("Replaying recorded responses from %s (no API calls): %s", settings.recordings_dir, replay.name)
        return replay

    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set (free key: https://console.groq.com/keys).")
        from .groq_client import GroqClient

        client = GroqClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
        )
        if settings.record:
            log.info("Recording responses to %s", settings.recordings_dir)
            client = CachedLLMClient(client, settings.recordings_dir)
            log.info("LLM provider: %s", client.name)
            return client
    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set (see README > Configuration).")
        from .gemini_client import GeminiClient

        client = GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
            min_interval_s=settings.llm_min_interval_s,
            structured_output=settings.gemini_structured_output,
            vertexai=settings.gemini_use_vertex,
        )
        if settings.record:
            log.info("Recording Gemini responses to %s", settings.recordings_dir)
            client = CachedLLMClient(client, settings.recordings_dir)
            log.info("LLM provider: %s", client.name)
            return client
    else:
        if settings.llm_provider == "auto":
            log.warning("No GEMINI_API_KEY and no recordings - running with the OFFLINE MOCK provider.")
        from .mock_client import MockLLMClient

        client = MockLLMClient()

    if settings.use_cache:
        client = CachedLLMClient(client, settings.cache_dir)
    log.info("LLM provider: %s", client.name)
    return client


__all__ = ["LLMClient", "LLMError", "build_llm_client", "resolve_provider"]
