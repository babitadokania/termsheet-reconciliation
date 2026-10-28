"""Offline, deterministic stand-in for the LLM.

Used automatically when no GEMINI_API_KEY is configured, so reviewers and CI can run the complete
pipeline. It honours the same contract as the real model:

- extract_fields          -> regex heuristics over the document text (see heuristics.py)
- adjudicate_mismatches   -> rule-based equivalence checks (units, formatting, synonyms)

The output is clearly labelled "mock" in every report so it is never mistaken for an LLM result.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..heuristics import heuristic_extract
from .base import LLMClient, LLMError

SYNONYMS = {
    "frequency": [
        {"semi-annual", "semiannual", "semi annual", "half-yearly", "half yearly", "twice a year", "2"},
        {"annual", "annually", "yearly", "1"},
        {"quarterly", "4"},
        {"monthly", "12"},
    ],
    "daycount": [
        {"actual/actual", "act/act", "actual/actual (icma)", "act/act icma", "actual / actual"},
        {"30/360", "30e/360", "30/360 isda", "bond basis"},
        {"actual/360", "act/360"},
        {"actual/365", "act/365", "actual/365 fixed"},
    ],
    "amortization": [{"bullet", "repaid in full at maturity", "repaid in full at the maturity date", "at maturity"}],
}


def _norm(s: Any) -> str:
    return re.sub(r"[\s_\-.,]+", "", str(s)).lower()


def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _to_date(v: Any) -> date | None:
    try:
        return date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _same_synonym(a: str, b: str) -> bool:
    a_l, b_l = str(a).strip().lower(), str(b).strip().lower()
    return any(a_l in group and b_l in group for groups in SYNONYMS.values() for group in groups)


def adjudicate(item: dict[str, Any]) -> dict[str, Any]:
    ts, bk, kind = item.get("term_sheet_value"), item.get("booking_value"), item.get("kind")
    quote = (item.get("term_sheet_quote") or "").lower()
    res = {"id": item["id"], "confidence": 0.9}

    if kind in ("percent", "amount"):
        a, b = _to_float(ts), _to_float(bk)
        if a is not None and b is not None:
            for factor, label in (
                (100, "percent vs decimal"),
                (0.01, "decimal vs percent"),
                (1e7, "crores vs absolute units"),
                (1e5, "lakhs vs absolute units"),
                (1e3, "thousands vs units"),
                (1e6, "millions vs units"),
            ):
                if b and abs(a * factor - b) <= 1e-9 * max(abs(b), 1):
                    return {
                        **res,
                        "verdict": "FALSE_ALERT",
                        "reason": f"Same value expressed in different units/scale ({label}): {ts} vs {bk}.",
                    }
                if a and abs(b * factor - a) <= 1e-9 * max(abs(a), 1):
                    return {
                        **res,
                        "verdict": "FALSE_ALERT",
                        "reason": f"Same value expressed in different units/scale ({label}): {ts} vs {bk}.",
                    }
            return {
                **res,
                "verdict": "TRUE_MISMATCH",
                "reason": f"Economic difference of {b - a:+,.6g} (term sheet {a:,.6g} vs booking {b:,.6g}).",
            }

    if kind in ("date", "date_or_perpetual"):
        if str(ts).upper() == "PERPETUAL" and str(bk).upper() != "PERPETUAL":
            return {
                **res,
                "verdict": "TRUE_MISMATCH",
                "reason": f"Term sheet defines a perpetual instrument with no maturity, but booking carries a fixed maturity of {bk}.",
            }
        da, db = _to_date(ts), _to_date(bk)
        if da and db:
            if da == db:
                return {**res, "verdict": "FALSE_ALERT", "reason": "Same calendar date in a different format."}
            return {
                **res,
                "verdict": "TRUE_MISMATCH",
                "reason": f"Dates differ by {(db - da).days:+d} day(s): term sheet {da.isoformat()} vs booking {db.isoformat()}.",
            }

    if kind == "list":
        la = sorted(_norm(x) for x in re.split(r"[|,;]", str(ts)) if x.strip())
        lb = sorted(_norm(x) for x in re.split(r"[|,;]", str(bk)) if x.strip())
        if la == lb:
            return {**res, "verdict": "FALSE_ALERT", "reason": "Same set of values; only order/separator differs."}
        return {**res, "verdict": "TRUE_MISMATCH", "reason": f"Different sets of values: {ts} vs {bk}."}

    # strings
    if _norm(ts) == _norm(bk):
        return {
            **res,
            "verdict": "FALSE_ALERT",
            "reason": "Identical once whitespace, case and punctuation are ignored (formatting difference).",
        }
    if _same_synonym(str(ts), str(bk)) or _same_synonym(quote, str(bk)):
        return {
            **res,
            "verdict": "FALSE_ALERT",
            "reason": f"'{ts}' and '{bk}' are market-standard synonyms for the same term.",
        }
    if (
        item.get("field") == "business_day_convention"
        and "adjust" in str(bk).lower()
        and any(k in str(ts).lower() for k in ("following", "preceding"))
    ):
        return {
            **res,
            "verdict": "FALSE_ALERT",
            "confidence": 0.6,
            "reason": "Booking uses a generic 'business day adjustment' label; term sheet specifies adjusted "
            "(following/preceding) conventions - consistent, but booking is less specific.",
        }
    return {
        **res,
        "verdict": "TRUE_MISMATCH",
        "confidence": 0.8,
        "reason": f"Values are not equivalent: term sheet '{ts}' vs booking '{bk}'.",
    }


class MockLLMClient(LLMClient):
    name = "mock:offline-rules"

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        task: str,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        context = context or {}
        if task == "extract_fields":
            return heuristic_extract(context["text"])
        if task == "adjudicate_mismatches":
            return {"results": [adjudicate(i) for i in context["items"]]}
        raise LLMError(f"Mock client does not support task '{task}'")
