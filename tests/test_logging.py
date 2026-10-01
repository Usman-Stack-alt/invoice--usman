import json
import logging

from app.logging_setup import JsonFormatter, TextFormatter, request_id_var
from app.pipeline import extract


def _rec(caplog, logger: str, msg: str) -> logging.LogRecord:
    found = [r for r in caplog.records if r.name == logger and r.getMessage() == msg]
    assert found, f"no '{msg}' record on '{logger}'; got {[(r.name, r.getMessage()) for r in caplog.records]}"
    return found[-1]


def test_extraction_logs_one_summary_line_with_stage_timings(template_a, caplog):
    caplog.set_level(logging.INFO)
    extract(template_a, "off")
    r = _rec(caplog, "pipeline", "extracted")
    assert (r.kind, r.status, r.items, r.llm, r.pages) == ("jpeg", "ok", 7, "off", 1)
    assert set(r.stages_ms) >= {"load", "ocr", "parse"} and r.ms >= sum(r.stages_ms.values()) * 0.5
    assert r.engine.startswith("tesseract")


def test_digital_pdf_logs_no_ocr_stage(docx_invoice, caplog):
    from app.pipeline.convert import docx_to_pdf, libreoffice_available

    if not libreoffice_available():
        return
    caplog.set_level(logging.INFO)
    extract(docx_to_pdf(docx_invoice), "off")
    assert "ocr" not in _rec(caplog, "pipeline", "extracted").stages_ms


def test_logs_never_contain_invoice_content(template_a, caplog):
    caplog.set_level(logging.DEBUG)
    extract(template_a, "off")
    text = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    for secret in ("Becker", "Andrews", "51109338", "GB75MCRL06841367619257", "945-82-2137", "6204.19"):
        assert secret not in text


def test_json_formatter_emits_valid_json_with_extras_and_request_id():
    token = request_id_var.set("rid-1")
    try:
        rec = logging.getLogger("t").makeRecord("t", logging.INFO, "f", 1, "hello", (), None, extra={"status": "ok"})
        line = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert line["msg"] == "hello" and line["status"] == "ok" and line["level"] == "INFO"


def test_text_formatter_is_one_readable_line():
    rec = logging.getLogger("pipeline").makeRecord(
        "pipeline", logging.INFO, "f", 1, "extracted", (), None, extra={"status": "ok", "ms": 5}
    )
    out = TextFormatter().format(rec)
    assert "INFO" in out and "extracted" in out and "status=ok" in out and "\n" not in out
