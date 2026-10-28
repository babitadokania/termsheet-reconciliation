"""PySpark ingestion + reconciliation tests (need Java)."""

import json

import pytest

from tsrecon.booking_ingest import load_bookings
from tsrecon.reconcile import compare
from tsrecon.schema import validate_extraction

TEXT = "Issuer: Acme Bond Co\nCoupon Rate: 5.00% p.a.\nThe Bonds will have a nominal amount of USD 1,000 each."


def _extraction(**fields):
    base = {
        "isin": {"value": "XS0000000001", "quote": None},
        "issuer": {"value": "Acme Bond Co", "quote": "Acme Bond Co"},
        "coupon_rate": {"value": 5.0, "quote": "Coupon Rate: 5.00% p.a."},
        "nominal_amount_per_bond": {"value": 1000, "quote": "nominal amount of USD 1,000"},
        "maturity_date": {"value": "2030-01-15", "quote": None},
    }
    base.update(fields)
    return validate_extraction(base, document="acme", source_path="acme.pdf", source_text=TEXT, method="test")


@pytest.fixture
def booking_dir(tmp_path):
    (tmp_path / "a.csv").write_text(
        "TradeID,ISIN,Issuer,Maturity,Notional,Coupon\n"
        '1,XS0000000001,Acme Bond Co,1/15/2030,"5,000",5.00\n'
        "2,XS0000000001,Acme Bond Co,2030-01-16,5500,5.0\n"
        "3,XS0000000001,Acme Bond Co,not-a-date,5000,abc\n"
        "4,XS9999999999,Other Issuer,2030-01-15,5000,5\n"
    )
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "TradeID": 1,
                        "ISIN": "XS0000000001",
                        "Issuer": "Acme Bond Co",
                        "Maturity": "2030-01-15",
                        "Notional": 5000,
                        "Coupon": 5.0,
                    }
                ]
            }
        )
    )
    (tmp_path / "broken.json").write_text("{not json")
    return tmp_path


def test_ingestion_normalises_dedups_and_reports_dq(spark, booking_dir):
    res = load_bookings(spark, booking_dir)
    rows = {r["TradeID"]: r for r in res.bookings.collect()}
    assert rows["1"]["Maturity"] == "2030-01-15"  # US M/d/yyyy normalised
    assert rows["1"]["Notional"] == 5000.0  # thousands separator removed
    assert rows["1"]["source_files"] == "a.csv, a.json"  # CSV + JSON copies de-duplicated
    issues = " ".join(i["issue"] for i in res.dq_issues)
    assert "Unparseable date" in issues and "Non-numeric value" in issues
    assert "File skipped" in issues  # broken.json did not stop the run
    assert any(i["column"] == "Parent" for i in res.dq_issues)  # missing column reported


def test_compare_results(spark, booking_dir):
    res = load_bookings(spark, booking_dir)
    df = compare(spark, res.bookings, [_extraction()]).toPandas()
    get = lambda ref, field: df[(df.booking_ref == ref) & (df.field == field)].initial_result.iloc[0]  # noqa: E731
    assert get("XS0000000001-1", "maturity_date") == "MATCH"
    assert get("XS0000000001-1", "coupon_rate") == "MATCH"
    assert get("XS0000000001-2", "maturity_date") == "MISMATCH"
    assert get("XS0000000001-2", "notional_lot_check") == "RULE_BREACH"  # 5500 not a multiple of 1000
    assert get("XS0000000001-3", "coupon_rate") == "MISSING_IN_BOOKING"  # 'abc' became null
    assert get("XS0000000001-1", "issue_price") == "NOT_COMPARED"
    assert get("XS9999999999-4", "term_sheet") == "NO_TERM_SHEET"


def test_parsing_error_is_never_a_mismatch(spark, booking_dir):
    res = load_bookings(spark, booking_dir)
    ex = _extraction(coupon_rate={"value": "5%", "quote": "Coupon Rate: 5.00% p.a."})
    df = compare(spark, res.bookings, [ex]).toPandas()
    assert set(df[df.field == "coupon_rate"].initial_result) == {"PARSING_ERROR"}


def test_link_by_issuer_when_no_isin(spark, booking_dir):
    res = load_bookings(spark, booking_dir)
    ex = _extraction(isin={"value": None})
    df = compare(spark, res.bookings, [ex]).toPandas()
    linked = df[df.document == "acme"]
    assert set(linked.link_method) == {"Issuer name"}
    assert "XS9999999999-4" not in set(linked.booking_ref)
