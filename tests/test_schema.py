from datetime import date

from tsrecon.schema import (
    FIELD_SPECS,
    FieldStatus,
    TermSheetFields,
    normalise_text,
    quote_in_source,
    validate_extraction,
)

SOURCE = "Coupon Rate: 9.25% p.a., semi-annual interest payments. Maturity Date: 14 October 2025. Tenor Perpetual"


def _validate(raw):
    return validate_extraction(raw, document="doc", source_path="doc.pdf", source_text=SOURCE, method="test")


def test_specs_cover_schema():
    assert {s.name for s in FIELD_SPECS} == set(TermSheetFields.model_fields)


def test_valid_values_are_typed():
    ex = _validate(
        {
            "coupon_rate": {"value": 9.25, "quote": "Coupon Rate: 9.25% p.a."},
            "maturity_date": {"value": "2025-10-14", "quote": "Maturity Date: 14 October 2025"},
        }
    )
    assert ex.get("coupon_rate").status == FieldStatus.OK
    assert ex.get("coupon_rate").value == 9.25
    assert ex.get("maturity_date").value == date(2025, 10, 14)
    assert ex.get("coupon_rate").quote_verified is True


def test_string_where_float_expected_is_parsing_error():
    ex = _validate({"coupon_rate": {"value": "9.25%", "quote": "Coupon Rate: 9.25% p.a."}})
    f = ex.get("coupon_rate")
    assert f.status == FieldStatus.PARSING_ERROR
    assert f.raw_value == "9.25%"
    assert "coupon_rate" in ex.parsing_errors


def test_bad_date_and_currency_are_parsing_errors():
    ex = _validate(
        {
            "maturity_date": {"value": "14 October 2025", "quote": "x"},
            "currency": {"value": "Rs.", "quote": "x"},
            "issue_amount": {"value": "1,500 Crores", "quote": "x"},
        }
    )
    assert ex.get("maturity_date").status == FieldStatus.PARSING_ERROR
    assert ex.get("currency").status == FieldStatus.PARSING_ERROR
    assert ex.get("issue_amount").status == FieldStatus.PARSING_ERROR


def test_perpetual_and_missing_fields():
    ex = _validate({"maturity_date": {"value": "perpetual", "quote": "Tenor Perpetual"}, "isin": {"value": None}})
    assert ex.get("maturity_date").status == FieldStatus.PARSING_ERROR  # case-sensitive literal
    ex = _validate({"maturity_date": {"value": "PERPETUAL", "quote": "Tenor Perpetual"}})
    assert ex.get("maturity_date").value == "PERPETUAL"
    assert ex.get("isin").status == FieldStatus.NOT_FOUND


def test_bare_value_and_null_strings_tolerated():
    ex = _validate({"coupon_rate": 9.25, "parent": {"value": "N/A", "quote": None}})
    assert ex.get("coupon_rate").status == FieldStatus.OK
    assert ex.get("parent").status == FieldStatus.NOT_FOUND


def test_fabricated_quote_is_flagged():
    ex = _validate({"coupon_rate": {"value": 9.25, "quote": "The coupon shall be nine and a quarter percent"}})
    assert ex.get("coupon_rate").quote_verified is False
    assert any("not found verbatim" in w for w in ex.warnings)


def test_quote_matching_ignores_quotes_and_whitespace():
    src = normalise_text("business day convention is “unadjusted” and\n business day is “Oslo”")
    assert quote_in_source('business day convention is "unadjusted"', src)
    assert quote_in_source("convention is unadjusted ... business day is Oslo", src)
