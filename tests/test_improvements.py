"""Tests for accuracy evaluation, replay/recording, Gemini structured output fallback and peer checks."""

import json
from types import SimpleNamespace

import pytest

from tsrecon.config import load_settings
from tsrecon.evaluation import evaluate_extraction
from tsrecon.llm import LLMError, build_llm_client, resolve_provider
from tsrecon.llm.cache import CachedLLMClient, RecordingNotFoundClient, ReplayMissError
from tsrecon.schema import extraction_json_schema, validate_extraction

TEXT = "Coupon Rate: 9.25% p.a. ISIN: NO 0010894330 Tenor Perpetual"


def _ex(raw):
    return validate_extraction(raw, document="d", source_path="d.pdf", source_text=TEXT, method="test")


# ---------------------------------------------------------------- evaluation
def test_evaluation_outcomes():
    ex = _ex(
        {
            "coupon_rate": {"value": 9.25, "quote": "Coupon Rate: 9.25% p.a."},
            "isin": {"value": "NO 0010894330", "quote": "ISIN: NO 0010894330"},
            "issue_price": {"value": 100, "quote": None},
            "currency": {"value": "Rs.", "quote": None},
            "parent": {"value": "Acme plc", "quote": None},
        }
    )
    gold = {
        "coupon_rate": {"expected": 9.25},
        "isin": {"expected": "NO0010894330"},
        "issue_price": {"expected": 97},
        "currency": {"expected": "INR"},
        "parent": {"expected": None},
        "maturity_date": {"expected": "PERPETUAL"},
    }
    out = evaluate_extraction(ex, gold).set_index("schema_field")["outcome"].to_dict()
    assert out == {
        "isin": "Correct",
        "parent": "Hallucinated (should be null)",
        "currency": "Parsing error",
        "coupon_rate": "Correct",
        "maturity_date": "Missed (returned null)",
        "issue_price": "Wrong value",
    }


def test_golden_files_cover_every_field(root):
    from tsrecon.schema import FIELD_SPECS

    for path in (root / "data" / "expected").glob("*.json"):
        gold = json.loads(path.read_text())
        assert {s.name for s in FIELD_SPECS} <= set(gold), path.name


# ---------------------------------------------------------------- replay / record
def test_provider_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = load_settings(recordings_dir=tmp_path / "rec")
    assert resolve_provider(s) == "mock"
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "x.json").write_text("{}")
    assert resolve_provider(s) == "replay"
    s = load_settings(recordings_dir=tmp_path / "rec", gemini_api_key="k")
    assert resolve_provider(s) == "gemini"


def test_replay_requires_recordings(tmp_path):
    with pytest.raises(LLMError):
        build_llm_client(load_settings(llm_provider="replay", recordings_dir=tmp_path / "none"))


def test_record_then_replay_roundtrip(tmp_path):
    class FakeGemini:
        name = "gemini:gemini-2.5-flash"
        calls = 0

        def generate_json(self, **kw):
            FakeGemini.calls += 1
            return {"answer": 42}

    recorder = CachedLLMClient(FakeGemini(), tmp_path)  # what --record does
    assert recorder.generate_json(system="s", prompt="p", task="extract_fields") == {"answer": 42}

    replay = CachedLLMClient(RecordingNotFoundClient("gemini:gemini-2.5-flash"), tmp_path, read_only=True)
    assert replay.generate_json(system="s", prompt="p", task="extract_fields") == {"answer": 42}
    with pytest.raises(ReplayMissError):  # changed prompt -> no recording, never a network call
        replay.generate_json(system="s", prompt="other", task="extract_fields")
    assert FakeGemini.calls == 1


# ---------------------------------------------------------------- Gemini structured output
class _APIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


def _fake_gemini(responses):
    from tsrecon.llm.gemini_client import GeminiClient

    client = GeminiClient(api_key="fake", model="m", max_retries=3, min_interval_s=0)
    seen = []

    def generate_content(model, contents, config):
        seen.append(config)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=r)

    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, seen


def test_gemini_sends_schema():
    client, seen = _fake_gemini(['{"ok": 1}'])
    assert client.generate_json(system="s", prompt="p", task="t", schema={"type": "object"}) == {"ok": 1}
    assert seen[0].response_json_schema == {"type": "object"}
    assert seen[0].temperature == 0.0


def test_gemini_falls_back_to_plain_json_when_schema_rejected():
    client, seen = _fake_gemini([_APIError(400), '{"ok": 1}'])
    assert client.generate_json(system="s", prompt="p", task="t", schema={"type": "object"}) == {"ok": 1}
    assert seen[1].response_json_schema is None and client.structured_output is False


def test_gemini_retries_rate_limit_and_bad_json(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    client, _ = _fake_gemini([_APIError(429), "not json", '{"ok": 1}'])
    assert client.generate_json(system="s", prompt="p", task="t") == {"ok": 1}


def test_gemini_auth_error_fails_fast():
    client, seen = _fake_gemini([_APIError(401)])
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        client.generate_json(system="s", prompt="p", task="t")
    assert len(seen) == 1


def test_extraction_schema_is_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(extraction_json_schema())


# ---------------------------------------------------------------- peer notional check
def test_peer_outlier_flagged(spark, tmp_path):
    from tsrecon.booking_ingest import load_bookings
    from tsrecon.reconcile import compare

    rows = "\n".join(f"{i},XS1,Acme,2030-01-15,{n},5" for i, n in enumerate([1000, 1000, 1000, 3000], 1))
    (tmp_path / "b.csv").write_text("TradeID,ISIN,Issuer,Maturity,Notional,Coupon\n" + rows + "\n")
    ex = validate_extraction(
        {"isin": {"value": "XS1"}, "issuer": {"value": "Acme"}},
        document="acme",
        source_path="a",
        source_text="x" * 60,
        method="t",
    )
    df = compare(spark, load_bookings(spark, tmp_path).bookings, [ex]).toPandas()
    peer = df[df.field == "notional_peer_check"].set_index("booking_ref")["initial_result"].to_dict()
    assert peer == {"XS1-1": "MATCH", "XS1-2": "MATCH", "XS1-3": "MATCH", "XS1-4": "PEER_OUTLIER"}


def test_gemini_endpoint_selection():
    from tsrecon.llm.gemini_client import GeminiClient

    assert GeminiClient(api_key="AQ.fake", model="m").vertexai is False  # AI Studio keys -> Developer API
    assert GeminiClient(api_key="AIzaFake", model="m").vertexai is False
    assert GeminiClient(api_key="AQ.fake", model="m", vertexai=True).vertexai is True


def test_gemini_follows_retired_model_replacement():
    msg = "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available. Please use models/gemini-3.8-flash for"
    err = _APIError(404)
    err.args = (msg,)
    client2, _ = _fake_gemini([err, '{"ok": 1}'])
    assert client2.generate_json(system="s", prompt="p", task="t") == {"ok": 1}
    assert client2.model == "gemini-3.8-flash"


# ---------------------------------------------------------------- Groq
def _groq(handler, model=None):
    import httpx

    from tsrecon.llm.groq_client import GroqClient

    http = httpx.Client(base_url="https://api.groq.com/openai/v1", transport=httpx.MockTransport(handler))
    return GroqClient(api_key="gsk_fake", model=model, http=http, min_interval_s=0, max_retries=3)


def _chat(content):
    import httpx

    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_groq_picks_available_model_and_returns_json():
    import httpx

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "other"}]})
        assert json.loads(req.content)["response_format"] == {"type": "json_object"}
        return _chat('{"ok": 1}')

    c = _groq(handler)
    assert c.model == "llama-3.3-70b-versatile" and c.name == "groq:llama-3.3-70b-versatile"
    assert c.generate_json(system="s", prompt="p", task="t") == {"ok": 1}


def test_groq_switches_model_when_request_too_large():
    import httpx

    used = []

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": m} for m in ("llama-3.3-70b-versatile", "qwen/qwen3-32b")]}
            )
        model = json.loads(req.content)["model"]
        used.append(model)
        if model == "llama-3.3-70b-versatile":
            return httpx.Response(413, text="Request too large for model on tokens per minute (TPM)")
        return _chat('{"ok": 2}')

    c = _groq(handler)
    assert c.generate_json(system="s", prompt="p", task="t") == {"ok": 2}
    assert used == ["llama-3.3-70b-versatile", "qwen/qwen3-32b"]


def test_groq_bad_key_fails_clearly():
    import httpx

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        _groq(lambda req: httpx.Response(401, text="invalid api key"))


def test_replay_serves_any_recorded_provider(tmp_path):
    from tsrecon.llm.cache import ReplayClient, cache_key

    (tmp_path / f"t_{cache_key('groq:m', 's', 'p')}.json").write_text(
        json.dumps({"provider": "groq:m", "response": {"a": 1}})
    )
    r = ReplayClient(tmp_path)
    assert r.generate_json(system="s", prompt="p", task="t") == {"a": 1}
    with pytest.raises(ReplayMissError):
        r.generate_json(system="s", prompt="x", task="t")


def test_gemini_switches_model_when_overloaded(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    from tsrecon.llm.gemini_client import GeminiClient

    client = GeminiClient(api_key="fake", model="gemini-3.8-flash", max_retries=2, min_interval_s=0)
    used = []

    def generate_content(model, contents, config):
        used.append(model)
        if model == "gemini-3.8-flash":
            raise _APIError(503)
        return SimpleNamespace(text='{"ok": 1}')

    models = [
        SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
        for n in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.8-flash-image", "gemini-pro")
    ]
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content, list=lambda: models))
    assert client.generate_json(system="s", prompt="p", task="t") == {"ok": 1}
    assert used == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash"]
    assert client.name == "gemini:gemini-3.8-flash"  # stable name keeps recordings/replay consistent


def test_gemini_skips_models_without_free_quota(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    from tsrecon.llm.gemini_client import GeminiClient

    client = GeminiClient(api_key="fake", model="gemini-3.8-flash", max_retries=3, min_interval_s=0)
    used = []

    def generate_content(model, contents, config):
        used.append(model)
        if model == "gemini-3.8-flash":
            raise _APIError(503)
        if model == "gemini-3.7-flash":
            err = _APIError(429)
            err.args = ("429 RESOURCE_EXHAUSTED quota exceeded, limit: 0, model: gemini-3.7-flash",)
            raise err
        return SimpleNamespace(text='{"ok": 1}')

    models = [
        SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
        for n in ("gemini-omni-1.1-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")
    ]
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content, list=lambda: models))
    assert client.generate_json(system="s", prompt="p", task="t") == {"ok": 1}
    # 3 tries on the overloaded model, ONE try on the zero-quota model (no pointless retries), then success
    assert used == ["gemini-3.8-flash"] * 3 + ["gemini-3.7-flash", "gemini-3.6-flash"]
