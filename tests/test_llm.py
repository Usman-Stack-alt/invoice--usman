import json
import multiprocessing as mp
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from zoneinfo import ZoneInfo

import pytest

from app.config import get_settings
from app.pipeline import extract, llm
from app.pipeline.llm import LLMInvoice, LLMItem, LLMParty, LLMSkipped
from app.pipeline.merge import merge
from app.pipeline.ratelimit import QuotaLimiter

PT = ZoneInfo("America/Los_Angeles")


# ------------------------------------------------------------------ quota: the hard guarantee
class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_rpm_is_a_rolling_window(tmp_path):
    clock = Clock()
    q = QuotaLimiter(tmp_path, rpm=10, rpd=20, clock=clock)
    assert [q.try_acquire() for _ in range(12)] == [True] * 10 + [False] * 2
    clock.t += 30  # still inside the same 60 s window
    assert q.try_acquire() is False
    clock.t += 31
    assert q.try_acquire() is True


def test_rpd_cap_and_pacific_midnight_reset(tmp_path):
    clock = Clock(datetime(2026, 10, 1, 9, 0, tzinfo=PT).timestamp())
    q = QuotaLimiter(tmp_path, rpm=10, rpd=20, clock=clock)
    granted = 0
    for _ in range(30):
        clock.t += 61  # never the minute limit
        granted += q.try_acquire()
    assert granted == 20
    assert q.status()["remaining_today"] == 0
    clock.t = datetime(2026, 10, 2, 0, 1, tzinfo=PT).timestamp()  # just after midnight Pacific
    assert q.try_acquire() is True
    assert q.status()["remaining_today"] == 19


def test_state_survives_restart_and_is_shared(tmp_path):
    clock = Clock()
    for _ in range(10):
        assert QuotaLimiter(tmp_path, 10, 20, clock).try_acquire()
    assert QuotaLimiter(tmp_path, 10, 20, clock).try_acquire() is False  # a "new process" sees the same counter


def _hammer(path: str, n: int, out) -> None:
    q = QuotaLimiter(path, rpm=10, rpd=20)
    out.put(sum(q.try_acquire() for _ in range(n)))


def test_concurrent_processes_cannot_exceed_the_quota(tmp_path):
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    procs = [ctx.Process(target=_hammer, args=(str(tmp_path), 15, out)) for _ in range(6)]
    [p.start() for p in procs]
    [p.join() for p in procs]
    assert sum(out.get() for _ in procs) == 10  # 90 attempts across 6 processes, exactly rpm granted


# ------------------------------------------------------------------ merge policy
def _rules_result(template_a, **warn):
    return extract(template_a, "off")


def test_merge_fills_gaps_but_keeps_trusted_rules_values(template_a):
    base = extract(template_a, "off")
    assert base.meta.status == "ok"
    answer = LLMInvoice(invoice_number="WRONG", currency="usd", seller=LLMParty(name="Andrews, Kirby and Valdez", email="a@b.co"))
    merged, fields = merge(base, answer, trusted=True)
    assert merged.invoice.invoice_number == "51109338"  # trusted rules value wins a conflict
    assert merged.invoice.currency == "USD" and merged.seller.email == "a@b.co"  # gaps are filled
    assert set(fields) == {"seller.email"} or "invoice.currency" not in fields  # currency was already USD


def test_merge_untrusted_rules_lose_conflicts_and_arithmetic_picks_items(template_a):
    base = extract(template_a, "off")
    base.items[1].net_amount = 999.0  # a misread amount: breaks the arithmetic
    base.invoice.invoice_number = "ber"
    good = [
        LLMItem(
            description=i.description,
            quantity=i.quantity,
            unit_price=i.unit_price,
            net_amount=i.net_amount,
            gross_amount=i.gross_amount,
            vat_percent=i.vat_percent,
        )
        for i in extract(template_a, "off").items
    ]
    answer = LLMInvoice(invoice_number="51109338", items=good, subtotal=5640.17, tax=564.02, total=6204.19)
    merged, fields = merge(base, answer, trusted=False)
    assert merged.invoice.invoice_number == "51109338"
    assert merged.items[1].net_amount != 999.0 and "items" in fields


def test_merge_never_replaces_good_items_with_arithmetically_worse_ones(template_a):
    base = extract(template_a, "off")
    bad = [LLMItem(description="x", quantity=2, unit_price=10, net_amount=999)]
    merged, fields = merge(base, LLMInvoice(items=bad), trusted=False)
    assert len(merged.items) == 7 and "items" not in fields


def test_merge_ignores_malformed_llm_dates(template_a):
    base = extract(template_a, "off")
    base.invoice.date = None
    merged, _ = merge(base, LLMInvoice(date="13th Sept"), trusted=False)
    assert merged.invoice.date is None


# ------------------------------------------------------------------ pipeline hook (LLM mocked)
@pytest.fixture
def llm_env(monkeypatch, tmp_path):
    st = get_settings()
    monkeypatch.setattr(st, "gemini_api_key", "test-key")
    monkeypatch.setattr(st, "llm_state_dir", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        llm,
        "extract_with_llm",
        lambda data: calls.append(1) or LLMInvoice(invoice_number="LLM-1", seller=LLMParty(name="LLM Seller")),
    )
    return calls


def test_auto_mode_skips_the_llm_for_clean_invoices(template_a, llm_env):
    r = extract(template_a)
    assert llm_env == [] and not r.meta.llm_used and r.meta.status == "ok"


def _needs_review_precondition(receipt):
    if extract(receipt, "off").meta.status != "needs_review":
        pytest.skip("receipt is now extracted cleanly by the rules; pick another unreliable fixture")


def test_auto_mode_calls_the_llm_when_rules_are_unreliable(receipt, llm_env):
    _needs_review_precondition(receipt)
    r = extract(receipt)
    assert len(llm_env) == 1 and r.meta.llm_used
    assert r.invoice.invoice_number == "LLM-1" and "invoice.invoice_number" in r.meta.llm_fields
    assert r.meta.engine.endswith(get_settings().llm_model)


def test_off_mode_never_calls(template_b, llm_env):
    assert extract(template_b, "off").meta.llm_used is False and llm_env == []


def test_always_mode_calls_but_trusted_rules_win_conflicts(template_a, llm_env):
    r = extract(template_a, "always")
    assert len(llm_env) == 1
    assert r.invoice.invoice_number == "51109338"  # LLM said LLM-1; trusted rules kept


def test_quota_exhausted_returns_rules_result_with_warning(receipt, monkeypatch, llm_env):
    _needs_review_precondition(receipt)

    def boom(data):
        raise LLMSkipped("llm_quota_exhausted")

    monkeypatch.setattr(llm, "extract_with_llm", boom)
    r = extract(receipt)
    assert "llm_quota_exhausted" in r.meta.warnings and not r.meta.llm_used
    assert r.invoice.invoice_number == "TI-00218734"  # the rules result is still returned


def test_should_call_rules():
    from app.schemas import ExtractionMeta, InvoiceData, InvoiceMeta, LineItem, Party, Summary

    def doc(status="ok", number="1", date="2026-01-01", total=10.0, items=1, seller="S"):
        return InvoiceData(
            invoice=InvoiceMeta(invoice_number=number, date=date), seller=Party(name=seller),
            items=[LineItem(line=i) for i in range(items)], summary=Summary(total=total),
            meta=ExtractionMeta(status=status, confidence=1, engine="", processing_ms=0),
        )  # fmt: skip

    assert llm.should_call("auto", doc()) is False  # complete and consistent
    assert llm.should_call("auto", doc(status="needs_review")) is True
    assert llm.should_call("auto", doc(number=None)) is True
    assert llm.should_call("auto", doc(total=None)) is True
    assert llm.should_call("auto", doc(items=0)) is True
    assert llm.should_call("auto", doc(seller=None)) is True  # no party at all
    assert llm.should_call("off", doc(status="needs_review")) is False
    assert llm.should_call("always", doc()) is True


def test_unconfigured_llm_is_silent_in_auto_mode(template_b, monkeypatch):
    monkeypatch.setattr(get_settings(), "gemini_api_key", "")
    r = extract(template_b)
    assert not any(w.startswith("llm_") for w in r.meta.warnings)
    assert "llm_not_configured" in extract(template_b, "always").meta.warnings


# ------------------------------------------------------------------ real SDK against a fake Gemini server
class _Fake(BaseHTTPRequestHandler):
    hits: ClassVar[list[dict]] = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.hits.append({"path": self.path, "body": body})
        if _Fake.status != 200:
            payload = json.dumps({"error": {"code": _Fake.status, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        else:
            answer = LLMInvoice(
                invoice_number="INV-9",
                total=12.5,
                items=[LLMItem(description="Thing", quantity=1, unit_price=12.5, net_amount=12.5)],
            )
            payload = json.dumps(
                {
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": answer.model_dump_json()}]}, "finishReason": "STOP"}
                    ]
                }
            )
        self.send_response(_Fake.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_gemini(monkeypatch, tmp_path):
    _Fake.hits, _Fake.status = [], 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    st = get_settings()
    monkeypatch.setattr(st, "gemini_api_key", "test-key")
    monkeypatch.setattr(st, "gemini_base_url", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(st, "llm_state_dir", str(tmp_path))
    yield _Fake
    server.shutdown()


def test_sdk_request_shape_response_parsing_and_cache(fake_gemini, template_a):
    out = llm.extract_with_llm(template_a)
    assert out.invoice_number == "INV-9" and out.items[0].net_amount == 12.5

    hit = fake_gemini.hits[0]
    assert hit["path"].endswith(f"/models/{get_settings().llm_model}:generateContent")
    parts = hit["body"]["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mime_type"] == "image/jpeg" and parts[0]["inlineData"]["data"]
    assert "Never guess" in parts[1]["text"]
    cfg = hit["body"]["generationConfig"]
    assert cfg["temperature"] == 0 and cfg["responseMimeType"] == "application/json" and cfg["responseSchema"]

    llm.extract_with_llm(template_a)  # identical document: served from cache
    assert len(fake_gemini.hits) == 1
    assert llm.status()["remaining_today"] == 19  # only one request was charged


def test_http_429_is_not_retried_and_still_counts_against_quota(fake_gemini, template_a, caplog):
    fake_gemini.status = 429
    with pytest.raises(LLMSkipped) as e:
        llm.extract_with_llm(template_a)
    assert e.value.code == "llm_failed"
    assert len(fake_gemini.hits) == 1  # no retry: a retry would spend more quota
    rec = next(r for r in caplog.records if r.getMessage() == "gemini call failed")
    assert rec.http_status == "429" and rec.google_status == "RESOURCE_EXHAUSTED" and "quota" in rec.google_message
    assert llm.status()["remaining_today"] == 19


def test_requests_stop_when_quota_is_spent(fake_gemini, template_a, template_b, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_rpm", 1)
    llm.extract_with_llm(template_a)
    with pytest.raises(LLMSkipped) as e:
        llm.extract_with_llm(template_b)  # different document: not cached, quota used up
    assert e.value.code == "llm_quota_exhausted" and len(fake_gemini.hits) == 1


def test_pdf_is_sent_as_pdf(fake_gemini, scanned_pdf):
    llm.extract_with_llm(scanned_pdf)
    assert fake_gemini.hits[0]["body"]["contents"][0]["parts"][0]["inlineData"]["mime_type"] == "application/pdf"


def test_llm_calls_are_logged_with_latency_tokens_and_quota(fake_gemini, template_a, caplog):
    import logging

    caplog.set_level(logging.INFO)
    llm.extract_with_llm(template_a)
    call = next(r for r in caplog.records if r.getMessage() == "llm call")
    assert call.model == get_settings().llm_model and call.mime == "image/jpeg" and call.ms >= 0
    assert call.remaining_today == 19 and call.remaining_this_minute == 9

    caplog.clear()
    llm.extract_with_llm(template_a)
    hit = next(r for r in caplog.records if r.getMessage() == "llm cache hit")
    assert hit.quota_charged is False


def test_quota_exhaustion_is_logged_as_a_warning(fake_gemini, template_a, template_b, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(get_settings(), "llm_rpm", 1)
    llm.extract_with_llm(template_a)
    caplog.clear()
    with pytest.raises(LLMSkipped):
        llm.extract_with_llm(template_b)
    rec = next(r for r in caplog.records if "quota exhausted" in r.getMessage())
    assert rec.levelno == logging.WARNING and rec.remaining_this_minute == 0
