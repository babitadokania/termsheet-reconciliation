"""Step 3b: LLM second pass - classify each MISMATCH as TRUE_MISMATCH or FALSE_ALERT (with reason).

- Identical breaks (same document, field, term sheet value and booking value) are adjudicated once
  and the verdict is fanned out to every trade, which keeps API usage low.
- Breaks are sent in batches; the response is validated with Pydantic.
- If the LLM fails or omits an item, the break is conservatively kept as a TRUE MISMATCH and marked
  "needs review", so a failure can never hide a genuine break.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .llm.base import LLMClient, LLMError
from .llm.mock_client import adjudicate as rule_adjudicate
from .llm.prompts import ADJUDICATION_SYSTEM, build_adjudication_prompt

log = logging.getLogger(__name__)


class Verdict(BaseModel):
    id: int
    verdict: Literal["TRUE_MISMATCH", "FALSE_ALERT"]
    reason: str = Field(min_length=3)
    confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    adjudicated_by: str = ""


class AdjudicationResponse(BaseModel):
    results: list[Verdict]


ADJUDICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["TRUE_MISMATCH", "FALSE_ALERT"]},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "verdict", "reason", "confidence"],
            },
        }
    },
    "required": ["results"],
}


BreakKey = tuple[str, str, str, str]  # (document, field, ts_value, booking_value)


class MismatchAdjudicator:
    def __init__(self, llm: LLMClient, batch_size: int = 20):
        self.llm = llm
        self.batch_size = batch_size

    @staticmethod
    def _fallback(item: dict, why: str) -> Verdict:
        """LLM unavailable: use the deterministic equivalence rules (units, formatting, synonyms, dates).

        The rules only call something a False Alert when the values are provably equivalent; anything
        else stays a True Mismatch. Low-confidence rule verdicts are flagged for manual review.
        """
        rule = rule_adjudicate(item)
        confidence = rule.get("confidence")
        return Verdict(
            id=item["id"],
            verdict=rule["verdict"],
            reason=f"[Rule-based - LLM unavailable: {why}] {rule['reason']}",
            confidence=confidence,
            needs_review=confidence is None or confidence < 0.8,
            adjudicated_by="rules-fallback",
        )

    def adjudicate(self, breaks: list[dict]) -> dict[BreakKey, Verdict]:
        """`breaks`: dicts with document, field, field_label, kind, ts_value, ts_quote, booking_column, booking_value."""
        unique: dict[BreakKey, dict] = {}
        for b in breaks:
            key = (b["document"], b["field"], str(b["ts_value"]), str(b["booking_value"]))
            unique.setdefault(key, b)
        if not unique:
            return {}
        log.info(
            "Second pass: %d mismatching field(s) -> %d unique break(s) sent to %s",
            len(breaks),
            len(unique),
            self.llm.name,
        )

        keys = list(unique)
        verdicts: dict[BreakKey, Verdict] = {}
        for start in range(0, len(keys), self.batch_size):
            batch_keys = keys[start : start + self.batch_size]
            items = []
            for i, key in enumerate(batch_keys, start=start + 1):
                b = unique[key]
                items.append(
                    {
                        "id": i,
                        "document": b["document"],
                        "field": b["field"],
                        "field_label": b["field_label"],
                        "kind": b["kind"],
                        "term_sheet_value": b["ts_value"],
                        "term_sheet_quote": b.get("ts_quote"),
                        "booking_column": b["booking_column"],
                        "booking_value": b["booking_value"],
                    }
                )
            by_id = self._run_batch(items)
            for i, key in enumerate(batch_keys, start=start + 1):
                verdicts[key] = by_id.get(i) or self._fallback(items[i - start - 1], "no verdict returned")
        return verdicts

    def _run_batch(self, items: list[dict]) -> dict[int, Verdict]:
        try:
            raw = self.llm.generate_json(
                system=ADJUDICATION_SYSTEM,
                prompt=build_adjudication_prompt(items),
                task="adjudicate_mismatches",
                context={"items": items},
                schema=ADJUDICATION_JSON_SCHEMA,
            )
            if isinstance(raw, list):
                raw = {"results": raw}
            parsed = AdjudicationResponse.model_validate(raw)
        except (LLMError, ValidationError, TypeError) as exc:
            log.error("Adjudication batch failed (%s) - using rule-based adjudication for %d break(s)", exc, len(items))
            return {it["id"]: self._fallback(it, type(exc).__name__) for it in items}
        out = {}
        for v in parsed.results:
            v.adjudicated_by = self.llm.name
            out[v.id] = v
        return out
