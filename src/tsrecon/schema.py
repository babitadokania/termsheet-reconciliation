"""Strict extraction schema.

`TermSheetFields` is the contract between the LLM and the rest of the pipeline. Every field is an
`Evidence[T]` object: the typed value plus the verbatim quote from the term sheet it came from.

Each field is validated *independently* (see `validate_extraction`) so that one bad value, e.g. the
string "9.25%" where a float is expected, is flagged as a PARSING_ERROR for that field only. A parsing
error is never compared with the booking, so it can never be reported as a mismatch.

`FIELD_SPECS` maps every schema field to its booking-system column and comparison kind, so adding a
field means one line here plus one attribute on the model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, TypeAdapter, ValidationError

T = TypeVar("T")


# --------------------------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------------------------
Percent = Annotated[StrictFloat | StrictInt, Field(ge=0, le=100, description="Percent number, e.g. 9.25 for 9.25%")]
Amount = Annotated[StrictFloat | StrictInt, Field(ge=0, description="Absolute amount in base currency units")]
IsoCurrency = Annotated[StrictStr, Field(pattern=r"^[A-Z]{3}$", description="ISO-4217 code, e.g. USD")]
NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
MaturityValue = date | Literal["PERPETUAL"]


class Evidence(BaseModel, Generic[T]):
    """A typed value plus the exact source snippet used to derive it."""

    model_config = ConfigDict(extra="ignore")

    value: T | None = None
    quote: str | None = Field(default=None, description="Verbatim snippet from the term sheet")
    confidence: float | None = Field(default=None, ge=0, le=1)
    note: str | None = Field(default=None, description="How the value was derived, if not literal")


class TermSheetFields(BaseModel):
    """Strict schema for the fields extracted from one term sheet."""

    model_config = ConfigDict(extra="forbid")

    isin: Evidence[NonEmptyStr]
    issuer: Evidence[NonEmptyStr]
    parent: Evidence[NonEmptyStr]
    currency: Evidence[IsoCurrency]
    coupon_rate: Evidence[Percent]
    maturity_date: Evidence[MaturityValue]
    issue_date: Evidence[date]
    settlement_date: Evidence[date]
    first_interest_payment_date: Evidence[date]
    issue_amount: Evidence[Amount]
    issue_price: Evidence[Percent]
    nominal_amount_per_bond: Evidence[Amount]
    minimum_subscription: Evidence[Amount]
    interest_payment_frequency: Evidence[NonEmptyStr]
    day_count_fraction: Evidence[NonEmptyStr]
    business_day_convention: Evidence[NonEmptyStr]
    business_day_locations: Evidence[list[NonEmptyStr]]
    amortization_type: Evidence[NonEmptyStr]


class Kind(str, Enum):
    STRING = "string"
    DATE = "date"
    DATE_OR_PERPETUAL = "date_or_perpetual"
    PERCENT = "percent"
    AMOUNT = "amount"
    LIST = "list"


@dataclass(frozen=True)
class FieldSpec:
    name: str  # attribute on TermSheetFields
    booking_column: str  # column in the booking extract
    kind: Kind
    label: str  # business-friendly label for reports
    guidance: str  # extraction hint given to the LLM


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "isin", "ISIN", Kind.STRING, "ISIN", "International Securities Identification Number exactly as printed."
    ),
    FieldSpec(
        "issuer", "Issuer", Kind.STRING, "Issuer", "Legal name of the issuing entity only (no incorporation details)."
    ),
    FieldSpec(
        "parent", "Parent", Kind.STRING, "Parent / Guarantor", "Parent company of the issuer if stated, otherwise null."
    ),
    FieldSpec(
        "currency", "Currency", Kind.STRING, "Currency", "ISO-4217 code (e.g. 'Rs.'/'Rupees' -> INR, '$' -> USD)."
    ),
    FieldSpec(
        "coupon_rate", "Coupon", Kind.PERCENT, "Coupon (%)", "Annual coupon as a percent number: 10.75% -> 10.75."
    ),
    FieldSpec(
        "maturity_date",
        "Maturity",
        Kind.DATE_OR_PERPETUAL,
        "Maturity",
        "ISO date, or the string PERPETUAL if the bond has no maturity.",
    ),
    FieldSpec("issue_date", "IssueDate", Kind.DATE, "Issue Date", "Issue date / deemed date of allotment, ISO date."),
    FieldSpec(
        "settlement_date",
        "SettlementDate",
        Kind.DATE,
        "Settlement Date",
        "Primary settlement / pay-in completion date, ISO date. Usually equals the issue or deemed allotment date.",
    ),
    FieldSpec(
        "first_interest_payment_date",
        "InterestPaymentDate",
        Kind.DATE,
        "First Interest Payment Date",
        "First coupon payment date, ISO date. Derive from the schedule if only described in words.",
    ),
    FieldSpec(
        "issue_amount",
        "IssueAmount",
        Kind.AMOUNT,
        "Issue Amount",
        "Base issue size in absolute currency units (1 crore = 10,000,000; 1 lakh = 100,000). Exclude green-shoe/over-subscription options.",
    ),
    FieldSpec(
        "issue_price",
        "IssuePrice",
        Kind.PERCENT,
        "Issue Price (% of nominal)",
        "Issue price as percent of nominal: 97% -> 97.",
    ),
    FieldSpec(
        "nominal_amount_per_bond",
        "NominalAmountPerBond",
        Kind.AMOUNT,
        "Nominal / Face Value per Bond",
        "Face value of one bond in absolute currency units.",
    ),
    FieldSpec(
        "minimum_subscription",
        "MinimumSubscription",
        Kind.AMOUNT,
        "Minimum Subscription",
        "Minimum subscription/application in absolute currency units (convert a number of bonds using the face value).",
    ),
    FieldSpec(
        "interest_payment_frequency",
        "InterestPaymentFrequency",
        Kind.STRING,
        "Coupon Frequency",
        "One of Annual, Semi-annual, Quarterly, Monthly.",
    ),
    FieldSpec(
        "day_count_fraction",
        "DayCountFraction",
        Kind.STRING,
        "Day Count",
        "Day-count convention, e.g. 30/360, Actual/Actual.",
    ),
    FieldSpec(
        "business_day_convention",
        "BusinessDayConvention",
        Kind.STRING,
        "Business Day Convention",
        "Business day convention, e.g. Unadjusted, Following, Modified Following, Preceding.",
    ),
    FieldSpec(
        "business_day_locations",
        "BusinessDayLocation",
        Kind.LIST,
        "Business Day Calendars",
        "List of business-day centres/calendars, e.g. ['Oslo', 'New York'].",
    ),
    FieldSpec(
        "amortization_type", "AmortizationType", Kind.STRING, "Amortization", "Bullet, Amortizing, Perpetual, etc."
    ),
)

SPEC_BY_NAME = {s.name: s for s in FIELD_SPECS}
SPEC_BY_COLUMN = {s.booking_column: s for s in FIELD_SPECS}

assert set(SPEC_BY_NAME) == set(TermSheetFields.model_fields), "FIELD_SPECS and TermSheetFields are out of sync"


# --------------------------------------------------------------------------------------------
# Validated output
# --------------------------------------------------------------------------------------------
class FieldStatus(str, Enum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    PARSING_ERROR = "PARSING_ERROR"


class ExtractedField(BaseModel):
    name: str
    label: str
    booking_column: str
    kind: Kind
    status: FieldStatus
    value: Any = None  # typed value (date, float, list, ...) when status == OK
    raw_value: Any = None  # exactly what the LLM returned
    quote: str | None = None
    quote_verified: bool | None = None
    confidence: float | None = None
    note: str | None = None
    error: str | None = None

    def canonical(self) -> str | None:
        """String form used for Spark comparison (dates ISO, lists pipe-joined)."""
        if self.status != FieldStatus.OK or self.value is None:
            return None
        if isinstance(self.value, date):
            return self.value.isoformat()
        if isinstance(self.value, list):
            return "|".join(str(v) for v in self.value)
        if isinstance(self.value, float) and self.value.is_integer():
            return str(int(self.value))
        return str(self.value)


class TermSheetExtraction(BaseModel):
    document: str
    source_path: str
    extraction_method: str  # e.g. "gemini:gemini-3.8-flash" or "heuristic"
    fields: dict[str, ExtractedField]
    warnings: list[str] = Field(default_factory=list)

    def get(self, name: str) -> ExtractedField:
        return self.fields[name]

    @property
    def parsing_errors(self) -> list[str]:
        return [n for n, f in self.fields.items() if f.status == FieldStatus.PARSING_ERROR]


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------
_ADAPTERS: dict[str, TypeAdapter] = {
    name: TypeAdapter(info.annotation) for name, info in TermSheetFields.model_fields.items()
}


def normalise_text(text: str) -> str:
    """Normalise quotes/dashes/whitespace so evidence can be matched against the source."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‘’‚‛′‟„“”«»\"'`]", "", text)
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"[^\w%./,:;()\-&$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_source(quote: str | None, source_norm: str) -> bool | None:
    if not quote:
        return None
    q = normalise_text(quote)
    if not q:
        return None
    if q in source_norm:
        return True
    # Tolerate an ellipsis-joined quote: every fragment must be present
    parts = [normalise_text(p) for p in re.split(r"\.\.\.|…", quote) if p.strip()]
    return bool(parts) and all(p in source_norm for p in parts)


def _format_errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors())


def validate_extraction(
    raw: dict[str, Any],
    *,
    document: str,
    source_path: str,
    source_text: str,
    method: str,
) -> TermSheetExtraction:
    """Validate the raw LLM JSON field-by-field against `TermSheetFields`."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        raise TypeError(f"LLM output must be a JSON object, got {type(raw).__name__}")
    unknown = set(raw) - set(SPEC_BY_NAME)
    if unknown:
        warnings.append(f"Ignored unexpected fields from LLM: {sorted(unknown)}")

    source_norm = normalise_text(source_text)
    fields: dict[str, ExtractedField] = {}
    for spec in FIELD_SPECS:
        item = raw.get(spec.name)
        base = dict(name=spec.name, label=spec.label, booking_column=spec.booking_column, kind=spec.kind)

        if item is None:
            fields[spec.name] = ExtractedField(
                **base, status=FieldStatus.NOT_FOUND, error="Field missing from LLM output"
            )
            continue
        if not isinstance(item, dict):  # LLM returned a bare value instead of {value, quote}
            item = {"value": item, "quote": None}

        raw_value = item.get("value")
        quote = item.get("quote")
        if isinstance(raw_value, str) and raw_value.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "not found",
            "not applicable",
        }:
            raw_value = None
        try:
            ev = _ADAPTERS[spec.name].validate_python({**item, "value": raw_value})
        except ValidationError as exc:
            fields[spec.name] = ExtractedField(
                **base,
                status=FieldStatus.PARSING_ERROR,
                raw_value=raw_value,
                quote=quote if isinstance(quote, str) else None,
                quote_verified=quote_in_source(quote, source_norm) if isinstance(quote, str) else None,
                error=_format_errors(exc),
            )
            continue

        if ev.value is None:
            fields[spec.name] = ExtractedField(
                **base, status=FieldStatus.NOT_FOUND, raw_value=raw_value, quote=ev.quote, note=ev.note
            )
            continue

        value = ev.value
        if spec.kind in (Kind.PERCENT, Kind.AMOUNT):
            value = float(value)
        if spec.kind == Kind.DATE_OR_PERPETUAL and isinstance(value, str):
            value = value.upper()

        verified = quote_in_source(ev.quote, source_norm)
        if not ev.quote:
            warnings.append(f"{spec.name}: no evidence quote returned")
        elif verified is False:
            warnings.append(f"{spec.name}: evidence quote not found verbatim in source document")

        fields[spec.name] = ExtractedField(
            **base,
            status=FieldStatus.OK,
            value=value,
            raw_value=raw_value,
            quote=ev.quote,
            quote_verified=verified,
            confidence=ev.confidence,
            note=ev.note,
        )

    return TermSheetExtraction(
        document=document, source_path=source_path, extraction_method=method, fields=fields, warnings=warnings
    )


def llm_json_contract() -> str:
    """Compact description of the expected JSON, embedded in the extraction prompt."""
    lines = []
    for spec in FIELD_SPECS:
        type_hint = {
            Kind.STRING: "string" if spec.name != "currency" else "string (ISO-4217)",
            Kind.DATE: "string YYYY-MM-DD",
            Kind.DATE_OR_PERPETUAL: 'string YYYY-MM-DD or "PERPETUAL"',
            Kind.PERCENT: "number",
            Kind.AMOUNT: "number",
            Kind.LIST: "array of strings",
        }[spec.kind]
        lines.append(
            f'  "{spec.name}": {{"value": <{type_hint} | null>, "quote": <string | null>, '
            f'"confidence": <0-1>, "note": <string | null>}}   // {spec.guidance}'
        )
    return "{\n" + ",\n".join(lines) + "\n}"


def extraction_json_schema() -> dict[str, Any]:
    """JSON schema handed to Gemini as `response_json_schema` (structured output).

    Hand-written (no $refs) to stay inside the schema subset Gemini supports. It constrains shape
    and primitive types; Pydantic still performs the authoritative, stricter validation afterwards.
    """
    value_types = {
        Kind.STRING: {"type": "string"},
        Kind.DATE: {"type": "string", "description": "YYYY-MM-DD"},
        Kind.DATE_OR_PERPETUAL: {"type": "string", "description": "YYYY-MM-DD or PERPETUAL"},
        Kind.PERCENT: {"type": "number"},
        Kind.AMOUNT: {"type": "number"},
        Kind.LIST: {"type": "array", "items": {"type": "string"}},
    }
    nullable_str = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    props = {}
    for spec in FIELD_SPECS:
        props[spec.name] = {
            "type": "object",
            "description": spec.guidance,
            "properties": {
                "value": {"anyOf": [value_types[spec.kind], {"type": "null"}]},
                "quote": nullable_str,
                "confidence": {"type": "number"},
                "note": nullable_str,
            },
            "required": ["value", "quote", "confidence", "note"],
        }
    return {"type": "object", "properties": props, "required": [s.name for s in FIELD_SPECS]}
