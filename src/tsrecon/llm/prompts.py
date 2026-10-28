"""Prompt templates. Kept in one place so they can be versioned and reviewed like code."""

from __future__ import annotations

import json
from typing import Any

from ..schema import llm_json_contract

PROMPT_VERSION = "v1.2"

EXTRACTION_SYSTEM = """You are a meticulous capital-markets operations analyst.
You extract trade economics from bond term sheets for a reconciliation against booking systems.

Rules:
1. Use ONLY the document text provided. Never use outside knowledge about the issuer.
2. For every field return {"value", "quote", "confidence", "note"}.
   - "quote" MUST be copied verbatim (character for character) from the document: the shortest
     snippet that proves the value. Do not paraphrase, do not add words.
   - If a value is derived (e.g. a date computed from "on the anniversary of the Deemed Date of
     Allotment", or an amount converted from crores), quote the source text and explain the
     derivation in "note".
3. If the document does not state a field and it cannot be derived with confidence, return
   {"value": null, "quote": null, "confidence": 0, "note": "<why>"}. Do NOT guess.
4. Formats: dates as YYYY-MM-DD; percentages as plain numbers (10.75% -> 10.75); amounts as plain
   numbers in absolute units (Rs.1,500 Crores -> 15000000000; Rs.10,00,000 -> 1000000; no thousands
   separators); currency as ISO-4217 code; perpetual bonds have maturity "PERPETUAL".
5. Return a single JSON object with exactly the keys requested and nothing else."""

EXTRACTION_TEMPLATE = """Extract the following fields from the term sheet below.

Required JSON structure (one entry per field):
{contract}

=== TERM SHEET: {document} ===
{text}
=== END OF TERM SHEET ==="""


ADJUDICATION_SYSTEM = """You are a senior reconciliation analyst reviewing breaks between a bond term
sheet and a trade booking system. For each candidate break decide whether it is:

- "FALSE_ALERT": the two values mean the same thing economically and only differ in formatting,
  units, scale, case, whitespace, synonyms or level of detail (e.g. "10.75%" vs 0.1075,
  "NO 0010894330" vs "NO0010894330", "semi-annual" vs "Semi-annual", "repaid in full at maturity"
  vs "Bullet", 1,500 crores vs 15000000000).
- "TRUE_MISMATCH": the values are economically different (a different rate, amount, date, party or
  term), or the booking states something the term sheet contradicts. Even a 1-day date difference or
  a 0.01 rate difference is a TRUE_MISMATCH.

Use the verbatim term-sheet quote as the primary evidence. Be conservative: if you are not sure the
values are equivalent, answer TRUE_MISMATCH. Return JSON only."""

ADJUDICATION_TEMPLATE = """Review these {n} candidate breaks.

{items}

Return a JSON object: {{"results": [{{"id": <id>, "verdict": "TRUE_MISMATCH" | "FALSE_ALERT",
"reason": <one concise sentence a business user understands>, "confidence": <0-1>}}, ...]}}
with exactly one result per id."""


def build_extraction_prompt(document: str, text: str) -> str:
    return EXTRACTION_TEMPLATE.format(contract=llm_json_contract(), document=document, text=text)


def build_adjudication_prompt(items: list[dict[str, Any]]) -> str:
    rendered = "\n".join(json.dumps(i, ensure_ascii=False, default=str) for i in items)
    return ADJUDICATION_TEMPLATE.format(n=len(items), items=rendered)
