"""Provider-agnostic LLM interface."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """Unrecoverable LLM failure (after retries)."""


class LLMResponseFormatError(LLMError):
    """The model replied, but not with parseable JSON."""


class LLMClient(ABC):
    """All LLM calls go through `generate_json`.

    `task` names the call ("extract_fields" / "adjudicate_mismatches"); `context` carries the
    structured inputs used to build the prompt; `schema` is an optional JSON schema the provider may
    use to constrain its output (Gemini structured output). Real providers only use the prompts and
    schema; the offline mock uses `context` to produce deterministic answers.
    """

    name: str = "llm"

    @abstractmethod
    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any: ...


def parse_json_response(text: str) -> Any:
    """Parse model output as JSON, tolerating markdown fences or leading prose."""
    if text is None:
        raise LLMResponseFormatError("Empty response from model")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost JSON object/array in the text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMResponseFormatError(f"Model output is not valid JSON: {cleaned[:300]!r}")
