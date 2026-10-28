"""Extraction accuracy evaluation against hand-verified golden values (data/expected/<document>.json).

Golden file format (one entry per schema field):
    {"coupon_rate": {"expected": 9.25},
     "isin":        {"expected": null},                       # field is NOT in the document
     "issue_price": {"expected": null, "accept": [100]},      # alternatives that are also correct
     "business_day_convention": {"expected_contains": "following"}}

Outcomes: Correct | Wrong value | Missed (returned null) | Hallucinated (should be null) | Parsing error.
Used by the pipeline (an "Extraction Accuracy" sheet) and by scripts/evaluate_extraction.py, so the
quality of a prompt or model change can be measured instead of eyeballed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import FIELD_SPECS, FieldStatus, Kind, TermSheetExtraction

log = logging.getLogger(__name__)


def _norm_str(v: Any) -> str:
    return re.sub(r"[^a-z0-9/%]", "", str(v).lower())


def _equal(kind: Kind, got: Any, exp: Any) -> bool:
    if got is None or exp is None:
        return got is None and exp is None
    if kind in (Kind.PERCENT, Kind.AMOUNT):
        try:
            g, e = float(got), float(exp)
        except (TypeError, ValueError):
            return False
        return abs(g - e) <= 1e-6 * max(1.0, abs(e))
    if kind in (Kind.DATE, Kind.DATE_OR_PERPETUAL):
        got_s = got.isoformat() if isinstance(got, date) else str(got)
        return got_s.upper() == str(exp).upper()
    if kind == Kind.LIST:
        got_l = got if isinstance(got, list) else str(got).split("|")
        exp_l = exp if isinstance(exp, list) else str(exp).split("|")
        return sorted(_norm_str(x) for x in got_l) == sorted(_norm_str(x) for x in exp_l)
    return _norm_str(got) == _norm_str(exp)


def load_expected(expected_dir: Path, document: str) -> dict | None:
    path = expected_dir / f"{document}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("Ignoring invalid golden file %s: %s", path.name, exc)
        return None


def evaluate_extraction(extraction: TermSheetExtraction, expected: dict) -> pd.DataFrame:
    rows = []
    for spec in FIELD_SPECS:
        gold = expected.get(spec.name)
        if gold is None:
            continue
        f = extraction.fields[spec.name]
        got = f.value if f.status == FieldStatus.OK else None
        if "expected_contains" in gold:
            needle = str(gold["expected_contains"]).lower()
            ok = got is not None and needle in str(got).lower()
            exp_display = f"contains '{gold['expected_contains']}'"
            exp_is_null = False
        else:
            candidates = [gold.get("expected")] + list(gold.get("accept", []))
            ok = any(_equal(spec.kind, got, c) for c in candidates)
            exp_display = gold.get("expected")
            exp_is_null = gold.get("expected") is None

        if f.status == FieldStatus.PARSING_ERROR:
            outcome = "Parsing error"
            ok = False
        elif ok:
            outcome = "Correct"
        elif got is None:
            outcome = "Missed (returned null)"
        elif exp_is_null and not ok:
            outcome = "Hallucinated (should be null)"
        else:
            outcome = "Wrong value"
        rows.append(
            {
                "document": extraction.document,
                "field": spec.label,
                "schema_field": spec.name,
                "expected": json.dumps(exp_display, default=str) if isinstance(exp_display, list) else exp_display,
                "extracted": f.canonical()
                if f.status == FieldStatus.OK
                else (f.raw_value if f.raw_value is not None else None),
                "outcome": outcome,
                "correct": ok,
                "quote_verified": f.quote_verified,
                "extraction_method": extraction.extraction_method,
            }
        )
    return pd.DataFrame(rows)


def evaluate_all(extractions: list[TermSheetExtraction], expected_dir: Path) -> pd.DataFrame:
    frames = []
    for ex in extractions:
        gold = load_expected(expected_dir, ex.document)
        if gold is None:
            log.info("No golden values for %s - accuracy not evaluated", ex.document)
            continue
        frames.append(evaluate_extraction(ex, gold))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for doc, g in df.groupby("document"):
        log.info(
            "Extraction accuracy [%s]: %d/%d fields correct (%.0f%%)",
            doc,
            g["correct"].sum(),
            len(g),
            100 * g["correct"].mean(),
        )
    return df


def accuracy_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a (no golden values)"
    parts = [f"{doc}: {int(g['correct'].sum())}/{len(g)}" for doc, g in df.groupby("document")]
    return f"{int(df['correct'].sum())}/{len(df)} fields correct ({100 * df['correct'].mean():.0f}%) - " + ", ".join(
        parts
    )
