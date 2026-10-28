import pytest

from tsrecon.document_parser import DocumentParseError, parse_document
from tsrecon.heuristics import heuristic_extract
from tsrecon.schema import FieldStatus, validate_extraction


def _extract(path):
    doc = parse_document(path)
    return validate_extraction(
        heuristic_extract(doc.text), document=doc.name, source_path=str(path), source_text=doc.text, method="heuristic"
    )


def test_pdf_genel(root):
    ex = _extract(root / "data" / "term_sheets" / "Term_Sheet_Genel_Energy.pdf")
    assert ex.get("isin").value == "NO 0010894330"
    assert ex.get("coupon_rate").value == 9.25
    assert ex.get("maturity_date").value.isoformat() == "2025-10-14"
    assert ex.get("business_day_locations").value == ["Oslo", "New York"]
    assert all(f.quote_verified for f in ex.fields.values() if f.status == FieldStatus.OK)


def test_pdf_idbi_units_and_perpetual(root):
    ex = _extract(root / "data" / "term_sheets" / "Term_Sheet_IDBI.pdf")
    assert ex.get("issue_amount").value == 15_000_000_000  # Rs.1,500 Crores
    assert ex.get("nominal_amount_per_bond").value == 1_000_000  # Rs.10,00,000
    assert ex.get("minimum_subscription").value == 5_000_000  # 5 bonds
    assert ex.get("maturity_date").value == "PERPETUAL"
    assert ex.get("currency").value == "INR"
    assert ex.get("isin").status == FieldStatus.NOT_FOUND


def test_word_term_sheet(root):
    path = root / "samples" / "word" / "Term_Sheet_Genel_Energy_KeyTerms.docx"
    ex = _extract(path)
    assert parse_document(path).format == "docx"
    assert ex.get("coupon_rate").value == 9.25
    assert ex.get("issue_amount").value == 300_000_000
    assert ex.get("currency").value == "USD"


def test_unsupported_and_empty_files(tmp_path):
    bad = tmp_path / "x.xls"
    bad.write_text("data")
    with pytest.raises(DocumentParseError):
        parse_document(bad)
    empty = tmp_path / "empty.txt"
    empty.write_text("  ")
    with pytest.raises(DocumentParseError):
        parse_document(empty)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(DocumentParseError):
        parse_document(corrupt)
