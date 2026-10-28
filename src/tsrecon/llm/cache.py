"""Disk cache / recordings for LLM responses.

Keyed by a hash of (provider name, system prompt, user prompt), so:
- re-running on unchanged documents makes no API calls (cheaper, reproducible, rate-limit friendly);
- a Gemini run can be *recorded* into `recordings/` (python run_pipeline.py --provider gemini --record)
  and committed, and anyone can *replay* it without a key (python run_pipeline.py --provider replay).

Delete output/llm_cache (or use --no-cache) to force fresh calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .base import LLMClient, LLMError

log = logging.getLogger(__name__)


def cache_key(provider_name: str, system: str, prompt: str) -> str:
    return hashlib.sha256(f"{provider_name}\n{system}\n{prompt}".encode()).hexdigest()[:32]


class CachedLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, cache_dir: Path, *, read_only: bool = False):
        self.inner = inner
        self.cache_dir = cache_dir
        self.read_only = read_only
        if not read_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        path = self.cache_dir / f"{task}_{cache_key(self.inner.name, system, prompt)}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                log.debug("LLM cache hit: %s", path.name)
                return payload["response"]
            except (json.JSONDecodeError, KeyError):
                log.warning("Corrupt cache entry %s - ignoring", path.name)
        result = self.inner.generate_json(system=system, prompt=prompt, task=task, context=context, schema=schema)
        self.misses += 1
        if not self.read_only:
            try:
                path.write_text(
                    json.dumps(
                        {"provider": self.inner.name, "task": task, "response": result},
                        indent=2,
                        default=str,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:  # caching is best-effort
                log.warning("Could not write LLM cache %s: %s", path, exc)
        return result


class ReplayMissError(LLMError):
    """No recorded response exists for this exact prompt."""


class RecordingNotFoundClient(LLMClient):
    """Inner client for replay mode: every cache miss is an error (no network calls ever)."""

    def __init__(self, name: str):
        self.name = name

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        raise ReplayMissError(
            f"No recorded {self.name} response for task '{task}' (inputs or prompts changed since recording). "
            "Re-record with: python run_pipeline.py --provider gemini --record"
        )


def recorded_providers(recordings_dir: Path) -> list[str]:
    names: set[str] = set()
    for f in recordings_dir.glob("*.json"):
        try:
            names.add(json.loads(f.read_text(encoding="utf-8"))["provider"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return sorted(names)


class ReplayClient(LLMClient):
    """Serves recorded responses only (any provider that was recorded); never calls the network."""

    def __init__(self, recordings_dir: Path):
        self.recordings_dir = recordings_dir
        self.providers = recorded_providers(recordings_dir)
        if not self.providers:
            raise LLMError(
                f"No recordings in {recordings_dir} (create them with --provider gemini --record or --provider groq --record)"
            )
        self.name = f"replay({', '.join(self.providers)})"
        self.hits = 0
        self.misses = 0

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        for provider in self.providers:
            path = self.recordings_dir / f"{task}_{cache_key(provider, system, prompt)}.json"
            if path.exists():
                self.hits += 1
                return json.loads(path.read_text(encoding="utf-8"))["response"]
        self.misses += 1
        raise ReplayMissError(
            f"No recorded response for task '{task}' (inputs or prompts changed since recording). "
            "Re-record with --provider gemini --record or --provider groq --record"
        )
