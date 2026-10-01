import pytest_asyncio
from httpx import ASGITransport, AsyncClient

H = {"X-API-Key": "test-key"}


@pytest_asyncio.fixture
async def client(monkeypatch):
    from app.api import extract as api
    from app.main import create_app

    # The processing limiter is a process-wide asyncio.Semaphore (one event loop per worker in production).
    # Each test runs on its own loop, so give every test a fresh one.
    monkeypatch.setattr(api, "_sem", None)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://t") as c:
        yield c


async def test_health_endpoints_need_no_auth(client):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    body = (await client.get("/readyz")).json()
    assert body["status"] == "ready" and body["tesseract"].startswith("5")


async def test_auth_required(client, template_a):
    r = await client.post("/v1/extract", files={"file": ("a.jpg", template_a)})
    assert r.status_code == 401
    r = await client.post("/v1/extract", files={"file": ("a.jpg", template_a)}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


async def test_extract_image(client, template_a):
    r = await client.post("/v1/extract", files={"file": ("a.jpg", template_a, "image/jpeg")}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["invoice"]["invoice_number"] == "51109338"
    assert body["summary"]["total"] == 6204.19
    assert body["meta"]["status"] == "ok"
    assert "x-request-id" in r.headers


async def test_extract_pdf_and_docx(client, scanned_pdf, docx_invoice):
    r = await client.post("/v1/extract", files={"file": ("a.pdf", scanned_pdf, "application/pdf")}, headers=H)
    assert r.status_code == 200 and r.json()["summary"]["total"] == 6204.19
    r = await client.post("/v1/extract", files={"file": ("a.docx", docx_invoice)}, headers=H)
    assert r.status_code == 200 and r.json()["invoice"]["invoice_number"] == "INV-1001"


async def test_rejects_bad_files(client):
    r = await client.post("/v1/extract", files={"file": ("a.jpg", b"<html>nope</html>", "image/jpeg")}, headers=H)
    assert r.status_code == 415
    r = await client.post("/v1/extract", files={"file": ("a.jpg", b"", "image/jpeg")}, headers=H)
    assert r.status_code == 400
    r = await client.post("/v1/extract", files={"file": ("a.jpg", b"\xff\xd8\xff" + b"junk" * 100, "image/jpeg")}, headers=H)
    assert r.status_code == 422 and r.json()["detail"]


async def test_size_limit(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    r = await client.post("/v1/extract", files={"file": ("a.jpg", b"\xff\xd8\xff" + b"0" * 2_000_000)}, headers=H)
    assert r.status_code == 413


async def test_batch_isolates_failures(client, template_a, template_b):
    files = [("files", ("a.jpg", template_a)), ("files", ("b.jpg", template_b)), ("files", ("bad.txt", b"hello"))]
    r = await client.post("/v1/extract/batch", files=files, headers=H)
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["ok"] for x in res] == [True, True, False]
    assert res[0]["data"]["invoice"]["invoice_number"] == "51109338"
    assert res[1]["data"]["invoice"]["invoice_number"] == "9362"
    assert "Unsupported" in res[2]["error"]


async def test_batch_limit(client, monkeypatch, template_a):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_batch_files", 2)
    r = await client.post("/v1/extract/batch", files=[("files", (f"{i}.jpg", template_a)) for i in range(3)], headers=H)
    assert r.status_code == 413


async def test_readyz_reports_llm_status(client):
    from app.config import get_settings

    body = (await client.get("/readyz")).json()
    assert body["llm"]["enabled"] is False and body["llm"]["model"] == get_settings().llm_model


async def test_llm_query_parameter_is_validated_and_passed_through(client, template_a, monkeypatch):
    r = await client.post("/v1/extract?llm=bogus", files={"file": ("a.jpg", template_a)}, headers=H)
    assert r.status_code == 422
    r = await client.post("/v1/extract?llm=always", files={"file": ("a.jpg", template_a)}, headers=H)
    assert r.status_code == 200
    assert "llm_not_configured" in r.json()["meta"]["warnings"]  # no key in tests: skipped, rules result returned
    r = await client.post("/v1/extract?llm=off", files={"file": ("a.jpg", template_a)}, headers=H)
    assert not any(w.startswith("llm_") for w in r.json()["meta"]["warnings"])


async def test_request_id_is_shared_by_api_and_pipeline_logs_and_sanitised(client, template_a, caplog):
    import logging

    caplog.set_level(logging.INFO)
    r = await client.post("/v1/extract?llm=off", files={"file": ("a.jpg", template_a)}, headers={**H, "X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    ids = {rec.getMessage(): rec.request_id for rec in caplog.records if rec.name in ("api", "pipeline")}
    assert ids["extracted"] == ids["request"] == "abc-123"  # the worker thread logs under the same id

    r = await client.post(
        "/v1/extract?llm=off", files={"file": ("a.jpg", template_a)}, headers={**H, "X-Request-ID": "bad id\nINJECT"}
    )
    assert r.headers["x-request-id"] != "bad id\nINJECT" and "\n" not in r.headers["x-request-id"]


async def test_rejections_are_logged_with_the_reason(client, caplog):
    import logging

    caplog.set_level(logging.INFO)
    await client.post("/v1/extract", files={"file": ("a.txt", b"hello")}, headers=H)
    rec = next(r for r in caplog.records if r.getMessage() == "rejected")
    assert rec.status == 415 and "Unsupported" in rec.reason


async def test_batch_holds_at_most_concurrency_files_in_memory(client, template_a, monkeypatch):
    """10 uploads must not all be read into memory at once: reads happen only while a processing slot is held."""

    from app.api import extract as api
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "concurrency", 2)
    monkeypatch.setattr(api, "_sem", None)  # rebuild the semaphore with the patched size
    state = {"now": 0, "peak": 0}
    real_read = api.read_upload

    async def tracked(f):
        data = await real_read(f)
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        return data

    real_extract = api.extract

    def slow_extract(data, llm=None):
        try:
            return real_extract(data, "off")
        finally:
            state["now"] -= 1

    monkeypatch.setattr(api, "read_upload", tracked)
    monkeypatch.setattr(api, "extract", slow_extract)
    files = [("files", (f"{i}.jpg", template_a)) for i in range(10)]
    r = await client.post("/v1/extract/batch", files=files, headers=H)
    assert r.status_code == 200 and all(x["ok"] for x in r.json()["results"])
    assert state["peak"] <= 2


async def test_batch_of_ten_respects_the_llm_quota_and_never_fails(client, template_a, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.pipeline import llm
    from app.pipeline.llm import LLMInvoice, LLMSkipped
    from app.pipeline.ratelimit import QuotaLimiter

    st = get_settings()
    monkeypatch.setattr(st, "gemini_api_key", "k")
    q = QuotaLimiter(tmp_path, rpm=4, rpd=20)
    calls = []

    def fake(data):
        if not q.try_acquire():
            raise LLMSkipped("llm_quota_exhausted")
        calls.append(1)
        return LLMInvoice()

    monkeypatch.setattr(llm, "extract_with_llm", fake)
    files = [("files", (f"{i}.jpg", template_a)) for i in range(10)]
    r = await client.post("/v1/extract/batch?llm=always", files=files, headers=H)
    res = r.json()["results"]
    assert r.status_code == 200 and all(x["ok"] for x in res)  # quota exhaustion never fails a file
    assert len(calls) == 4  # exactly the quota
    assert sum("llm_quota_exhausted" in x["data"]["meta"]["warnings"] for x in res) == 6
