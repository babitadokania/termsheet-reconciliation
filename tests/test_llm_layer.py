import pytest

from tsrecon.adjudicator import MismatchAdjudicator
from tsrecon.extractor import ExtractionService
from tsrecon.llm.base import LLMClient, LLMError, LLMResponseFormatError, parse_json_response
from tsrecon.llm.cache import CachedLLMClient
from tsrecon.llm.mock_client import adjudicate


class FailingLLM(LLMClient):
    name = "failing"

    def generate_json(self, **kwargs):
        raise LLMError("simulated outage")


class CountingLLM(LLMClient):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        return {"ok": True}


def test_parse_json_response_variants():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Here you go: {"a": 1} thanks') == {"a": 1}
    with pytest.raises(LLMResponseFormatError):
        parse_json_response("no json here")


@pytest.mark.parametrize(
    "item,verdict",
    [
        ({"kind": "percent", "term_sheet_value": "10.75", "booking_value": "0.1075"}, "FALSE_ALERT"),
        ({"kind": "percent", "term_sheet_value": "9.25", "booking_value": "9.5"}, "TRUE_MISMATCH"),
        ({"kind": "amount", "term_sheet_value": "1500", "booking_value": "15000000000"}, "FALSE_ALERT"),
        ({"kind": "date", "term_sheet_value": "2025-10-14", "booking_value": "2025-10-15"}, "TRUE_MISMATCH"),
        (
            {"kind": "date_or_perpetual", "term_sheet_value": "PERPETUAL", "booking_value": "2025-10-17"},
            "TRUE_MISMATCH",
        ),
        ({"kind": "string", "term_sheet_value": "NO 0010894330", "booking_value": "NO0010894330"}, "FALSE_ALERT"),
        ({"kind": "string", "term_sheet_value": "Semi-annual", "booking_value": "half-yearly"}, "FALSE_ALERT"),
        ({"kind": "string", "term_sheet_value": "Act/Act", "booking_value": "Actual/Actual"}, "FALSE_ALERT"),
        ({"kind": "string", "term_sheet_value": "Genel Energy plc", "booking_value": "Other Corp"}, "TRUE_MISMATCH"),
        ({"kind": "list", "term_sheet_value": "Oslo|New York", "booking_value": "New York|Oslo"}, "FALSE_ALERT"),
    ],
)
def test_mock_adjudication_rules(item, verdict):
    assert adjudicate({"id": 1, "field": "f", **item})["verdict"] == verdict


def test_adjudicator_falls_back_to_rules_on_llm_failure():
    breaks = [
        {
            "document": "d",
            "field": "coupon_rate",
            "field_label": "Coupon",
            "kind": "percent",
            "ts_value": "10.75",
            "ts_quote": "q",
            "booking_column": "Coupon",
            "booking_value": "0.1075",
        }
    ]
    real_break = dict(breaks[0], ts_value="10.75", booking_value="11")
    verdicts = MismatchAdjudicator(FailingLLM()).adjudicate([*breaks, real_break])
    by_value = {k[3]: v for k, v in verdicts.items()}
    assert by_value["0.1075"].verdict == "FALSE_ALERT"  # provable unit difference
    assert by_value["11"].verdict == "TRUE_MISMATCH"
    assert all(v.adjudicated_by == "rules-fallback" and "LLM unavailable" in v.reason for v in verdicts.values())


def test_adjudicator_deduplicates_identical_breaks():
    class Echo(LLMClient):
        name = "echo"
        calls = 0

        def generate_json(self, *, context, **kw):
            Echo.calls += 1
            return {"results": [{"id": i["id"], "verdict": "FALSE_ALERT", "reason": "same"} for i in context["items"]]}

    b = {
        "document": "d",
        "field": "isin",
        "field_label": "ISIN",
        "kind": "string",
        "ts_value": "NO 1",
        "ts_quote": "q",
        "booking_column": "ISIN",
        "booking_value": "NO1",
    }
    verdicts = MismatchAdjudicator(Echo()).adjudicate([b, dict(b), dict(b)])
    assert len(verdicts) == 1 and Echo.calls == 1


def test_extraction_falls_back_to_heuristics(root):
    from tsrecon.document_parser import parse_document

    doc = parse_document(root / "data" / "term_sheets" / "Term_Sheet_Genel_Energy.pdf")
    ex = ExtractionService(FailingLLM(), fallback_to_heuristics=True).extract_document(doc)
    assert ex.extraction_method == "heuristic-fallback"
    assert ex.get("coupon_rate").value == 9.25
    with pytest.raises(LLMError):
        ExtractionService(FailingLLM(), fallback_to_heuristics=False).extract_document(doc)


def test_cache_avoids_repeat_calls(tmp_path):
    inner = CountingLLM()
    c = CachedLLMClient(inner, tmp_path)
    for _ in range(3):
        assert c.generate_json(system="s", prompt="p", task="t") == {"ok": True}
    assert inner.calls == 1 and c.hits == 2
