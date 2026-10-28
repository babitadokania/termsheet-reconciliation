"""Rule-based (regex) term sheet extractor.

Two roles:
1. Powers the offline MOCK provider so the full workflow runs without an API key (CI, reviewers).
2. Automatic fallback when the LLM is unavailable after retries (EXTRACTION_FALLBACK=true).

It returns exactly the same JSON shape the LLM is asked for ({value, quote, confidence, note}),
so it flows through the same strict schema validation. It is deliberately simple: label-driven
patterns that cover common term sheet wording. The LLM is the primary extractor.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_PATTERNS = [
    (rf"\b(\d{{1,2}})\s+({MONTHS})\s*,?\s+(\d{{4}})\b", "%d %B %Y"),
    (rf"\b({MONTHS})\s+(\d{{1,2}})\s*,\s*(\d{{4}})\b", "%B %d %Y"),
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y %m %d"),
]
CRORE, LAKH = 10_000_000, 100_000


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def find_date(snippet: str) -> tuple[date | None, str | None]:
    best = None
    for pattern, fmt in DATE_PATTERNS:
        m = re.search(pattern, snippet, re.IGNORECASE)
        if m and (best is None or m.start() < best[0].start()):
            best = (m, fmt)
    if not best:
        return None, None
    m, fmt = best
    try:
        return datetime.strptime(" ".join(m.groups()), fmt).date(), m.group(0)
    except ValueError:
        return None, None


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _ev(value: Any, quote: str | None, confidence: float = 0.8, note: str | None = None) -> dict:
    return {"value": value, "quote": quote.strip() if quote else None, "confidence": confidence, "note": note}


def _missing(note: str) -> dict:
    return {"value": None, "quote": None, "confidence": 0.0, "note": note}


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + years, day=28)


class HeuristicExtractor:
    def __init__(self, text: str):
        self.raw = text
        self.flat = _flat(text)

    # -- helpers ------------------------------------------------------------------------------
    def _search(self, pattern: str, flags: int = re.IGNORECASE) -> re.Match | None:
        return re.search(pattern, self.flat, flags)

    def _label_value(self, label: str) -> tuple[str, str] | None:
        """Value text that follows a label at the start of a line ('Issuer: X' / '2. Issuer X')."""
        m = re.search(rf"^(?:\d+\.\s*)?{label}:?[ \t]+(.+)$", self.raw, re.IGNORECASE | re.MULTILINE)
        if not m:
            return None
        return m.group(1).strip(), m.group(0).strip()

    # -- fields -------------------------------------------------------------------------------
    def isin(self):
        m = self._search(r"ISIN:?\s*([A-Z]{2}\s?[A-Z0-9]{9,10})\b", 0)
        return _ev(m.group(1), m.group(0)) if m else _missing("No ISIN printed in the term sheet")

    def _entity(self, label: str):
        found = self._label_value(label)
        if not found:
            return _missing(f"No '{label}' line found")
        value, _line = found
        name = re.split(r"\s+\(|,\s+incorporated|\s+and\s+LEI", value)[0].strip().rstrip(".")
        return _ev(name, name)

    def issuer(self):
        return self._entity("Issuer")

    def parent(self):
        return self._entity("Parent")

    def currency(self):
        found = self._label_value("Currency")
        if found:
            m = re.search(r"\b([A-Z]{3})\b", found[0])
            if m:
                return _ev(m.group(1), found[1], 0.95)
        m = self._search(r"Rs\.\s?[\d,]+[^.]{0,40}|Rupees[^.)]{0,40}")
        if m:
            return _ev("INR", m.group(0), 0.85, "Amounts denominated in Rupees (Rs.)")
        m = self._search(r"\b(USD|EUR|GBP|NOK|SEK|JPY|CHF)\s?[\d,]{4,}", 0)
        return _ev(m.group(1), m.group(0), 0.7) if m else _missing("Currency not stated")

    def coupon_rate(self):
        m = self._search(r"(?:Coupon Rate|Interest Rate|Coupon):?\s*(\d+(?:\.\d+)?)\s*%(?:\s*p\.a\.)?")
        return _ev(_num(m.group(1)), m.group(0), 0.95) if m else _missing("Coupon rate not found")

    def maturity_date(self):
        m = self._search(r"Maturity Date:?\s*(.{0,80})")
        if m:
            d, dq = find_date(m.group(1))
            if d and dq:
                quote = m.group(0)[: m.group(0).find(dq) + len(dq)]
                return _ev(d.isoformat(), quote, 0.9)
        m = self._search(r"Tenor:?\s+Perpetual|perpetual bonds")
        if m:
            return _ev("PERPETUAL", m.group(0), 0.9, "Instrument is perpetual - no maturity date")
        return _missing("Maturity not found")

    def issue_date(self):
        m = self._search(r"Issue Date:?\s*(.{0,60})")
        if m:
            d, dq = find_date(m.group(1))
            if d and dq:
                return _ev(d.isoformat(), m.group(0)[: m.group(0).find(dq) + len(dq)], 0.9)
        m = self._search(rf"Deemed\s+Date\s+of\s+(?:Allotment\s+)?(({MONTHS})\s+\d{{1,2}}\s*,\s*\d{{4}})")
        if m:
            d, _ = find_date(m.group(1))
            if d:
                return _ev(d.isoformat(), m.group(0), 0.85, "Deemed Date of Allotment used as issue date")
        return _missing("Issue date not found")

    def settlement_date(self):
        issue = self.issue_date()
        if issue["value"]:
            return _ev(
                issue["value"],
                issue["quote"],
                0.7,
                "No separate settlement date; primary settlement assumed on issue/allotment date",
            )
        return _missing("Settlement date not found")

    def first_interest_payment_date(self):
        m = self._search(r"First Interest Payment\s*(?:Date:?)?\s*(.{0,80})")
        if m:
            d, dq = find_date(m.group(1))
            if d and dq:
                return _ev(d.isoformat(), m.group(0)[: m.group(0).find(dq) + len(dq)], 0.9)
        m = self._search(r"On the anniversary of the Deemed Date of Allotment")
        issue = self.issue_date()
        if m and issue["value"]:
            first = _add_years(date.fromisoformat(issue["value"]), 1)
            return _ev(
                first.isoformat(),
                m.group(0),
                0.75,
                "Annual coupon on the anniversary of the Deemed Date of Allotment -> allotment + 1 year",
            )
        return _missing("First interest payment date not found")

    def issue_amount(self):
        m = self._search(r"Issue Amount:?\s*([A-Z]{3})?\s*([\d,]+(?:\.\d+)?)")
        if m:
            return _ev(_num(m.group(2)), m.group(0), 0.95)
        m = self._search(r"Issue Size[^R]{0,60}(Rs\.?\s?([\d,]+(?:\.\d+)?)\s*(Crores?|Lakhs?))", re.IGNORECASE)
        if m:
            mult = CRORE if m.group(3).lower().startswith("crore") else LAKH
            return _ev(
                _num(m.group(2)) * mult,
                m.group(1),
                0.85,
                f"{m.group(2)} {m.group(3)} converted to absolute units (base issue, excl. green-shoe)",
            )
        return _missing("Issue amount not found")

    def issue_price(self):
        m = self._search(r"Issue Price:?\s*(\d+(?:\.\d+)?)\s*%")
        if m:
            return _ev(_num(m.group(1)), m.group(0), 0.95)
        m = self._search(r"(?:issued|Issue Price)\s+at\s+par")
        if m:
            return _ev(100.0, m.group(0), 0.8, "Issued at par")
        return _missing("Issue price not stated in the term sheet")

    def nominal_amount_per_bond(self):
        m = self._search(r"nominal amount of\s+([A-Z]{3})\s?([\d,]+)")
        if m:
            return _ev(_num(m.group(2)), m.group(0), 0.95)
        m = self._search(r"Face Value:?\s*(Rs\.?\s?([\d,]+))")
        if m:
            return _ev(_num(m.group(2)), m.group(1), 0.9, "Indian digit grouping (10,00,000 = 1,000,000)")
        return _missing("Nominal amount not found")

    def minimum_subscription(self):
        m = self._search(r"Minimum subscription[^.]{0,40}?amount shall be\s+([A-Z]{3})\s?([\d,]+)")
        if m:
            return _ev(_num(m.group(2)), m.group(0), 0.95)
        m = self._search(r"Minimum Application[^0-9]{0,80}?(\d+)\s+Bonds")
        nominal = self.nominal_amount_per_bond()
        if m and nominal["value"]:
            n = int(m.group(1))
            return _ev(
                n * nominal["value"], f"{m.group(1)} Bonds", 0.8, f"{n} bonds x face value {nominal['value']:,.0f}"
            )
        return _missing("Minimum subscription not found")

    def interest_payment_frequency(self):
        for pattern, value in (
            (r"semi-?\s?annual(?:ly)?[^.]{0,40}", "Semi-annual"),
            (r"payable\s+quarterly|quarterly interest", "Quarterly"),
            (r"payable\s+annually(?:\s+in\s+arrears?)?", "Annual"),
        ):
            m = self._search(pattern)
            if m:
                return _ev(value, m.group(0), 0.85)
        return _missing("Coupon frequency not found")

    def day_count_fraction(self):
        m = self._search(r"Day-?\s?count fraction for coupon is\s*[“\"]?([^”\"]+)[”\"]?")
        if m:
            return _ev(m.group(1).strip(), m.group(0), 0.95)
        m = self._search(r"Computation of Interest:?\s*(Actual\s*/\s*Actual|30/360|Actual\s*/\s*36[05])")
        if m:
            return _ev(re.sub(r"\s*/\s*", "/", m.group(1)), m.group(0), 0.9)
        return _missing("Day count not found")

    def business_day_convention(self):
        m = self._search(r"business day convention is\s*[“\"]?([A-Za-z ]+?)[”\"]")
        if m:
            return _ev(m.group(1).strip().capitalize(), m.group(0), 0.95)
        m = self._search(r"Business Day Convention:?\s*(.{0,600})")
        if m:
            body = m.group(1).lower()
            if "immediately next business day" in body and "immediately preceding" in body:
                q = self._search(r"Should the Coupon Payment Date[^.]{0,200}immediately next business day")
                return _ev(
                    "Following (coupon dates); Preceding (other dates)",
                    q.group(0) if q else "Business Day Convention",
                    0.75,
                )
        return _missing("Business day convention not found")

    def business_day_locations(self):
        m = self._search(r"business day is\s*((?:[“\"][^”\"]+[”\"](?:\s*(?:and|,)\s*)?)+)")
        if m:
            cities = re.findall(r"[“\"]([^”\"]+)[”\"]", m.group(1))
            return _ev(cities, m.group(0).strip(), 0.9)
        return _missing("Business day calendars not stated")

    def amortization_type(self):
        m = self._search(r"Amortization:?\s*(The Bonds shall be repaid in full at the Maturity Date[^.]{0,60})")
        if m:
            return _ev("Bullet", m.group(1), 0.85, "Repaid in full at maturity = bullet")
        m = self._search(r"Tenor:?\s+Perpetual")
        if m:
            return _ev("Perpetual", m.group(0), 0.85)
        return _missing("Amortization not stated")

    def extract(self) -> dict[str, dict]:
        from .schema import FIELD_SPECS

        out = {}
        for spec in FIELD_SPECS:
            fn: Callable[[], dict] = getattr(self, spec.name)
            try:
                out[spec.name] = fn()
            except Exception as exc:  # noqa: BLE001 - one bad rule must not break the others
                out[spec.name] = _missing(f"heuristic error: {exc}")
        return out


def heuristic_extract(text: str) -> dict[str, dict]:
    return HeuristicExtractor(text).extract()
