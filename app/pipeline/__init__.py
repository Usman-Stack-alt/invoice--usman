"""Invoice extraction pipeline: bytes -> clean InvoiceData.

load (image / PDF / DOCX) -> word boxes (PDF text layer, else Tesseract 5 on a
deskewed raster) -> layout parser -> arithmetic validation -> confidence.
"""

import logging
import time
from contextlib import contextmanager

from app.config import get_settings
from app.pipeline import llm as llm_stage
from app.pipeline.errors import ExtractionError
from app.pipeline.merge import merge
from app.pipeline.ocr import read_words, reread_number, tesseract_version
from app.pipeline.parser import PageParse, parse_page
from app.pipeline.preprocess import load_pages, normalize_source, prepare, sniff
from app.pipeline.validate import validate
from app.schemas import (
    Charge,
    ExtractionMeta,
    InvoiceData,
    InvoiceMeta,
    LineItem,
    Party,
    Payment,
    Summary,
)

log = logging.getLogger("pipeline")
__all__ = ["ExtractionError", "extract"]


def _merge(pages: list[PageParse]) -> PageParse:
    """Multi-page: header/parties from the first page that has them, items concatenated,
    summary from the last page that has a total."""
    m = PageParse()
    for p in pages:
        m.number = m.number or p.number
        m.date = m.date or p.date
        m.due_date = m.due_date or p.due_date
        m.currency = m.currency or p.currency
        m.seller = m.seller or p.seller
        m.client = m.client or p.client
        m.payment = m.payment or p.payment
        m.warnings += p.warnings
        m.table_found |= p.table_found
        for it in p.items:
            m.items.append({**it, "line": len(m.items) + 1})
        if p.summary.get("total") is not None or (p.summary and not m.summary):
            m.summary = p.summary
    return m


def _engine_name(used_ocr: bool, used_text: bool) -> str:
    parts = (["tesseract-" + tesseract_version()] if used_ocr else []) + (["pdf-text-layer"] if used_text else [])
    return "+".join(parts)


@contextmanager
def _timed(stages: dict[str, int], name: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        stages[name] = stages.get(name, 0) + int((time.perf_counter() - t) * 1000)


LLM_BASE_CONFIDENCE = 0.9  # the LLM reads the image itself, so OCR quality no longer bounds the score


def _score(result: InvoiceData, base_conf: float, parse_warnings: list[str], extra: list[str], started: float) -> None:
    """Validate, then set confidence / status / warnings / timing on result.meta."""
    s = get_settings()
    bad, missing = validate(result)
    if not (result.seller.name or result.client.name):
        missing.append("missing:parties")
    confidence = base_conf * (0.85 ** len(bad)) * (0.95 ** len(missing))
    result.meta.confidence = round(min(confidence, 1.0), 3)
    result.meta.warnings = list(dict.fromkeys(parse_warnings + extra + bad + missing))
    result.meta.status = "needs_review" if bad or result.meta.confidence < s.review_threshold else "ok"
    result.meta.processing_ms = int((time.perf_counter() - started) * 1000)


def extract(data: bytes, llm_mode: str | None = None) -> InvoiceData:
    """llm_mode: "auto" | "always" | "off"; None uses the LLM_MODE setting."""
    t0 = time.perf_counter()
    s = get_settings()
    stages: dict[str, int] = {}
    kind, size_kb = sniff(data), len(data) // 1024
    with _timed(stages, "load"):
        data = normalize_source(data)  # DOCX -> PDF once; the same bytes feed OCR/text layer and the LLM
        pages = load_pages(data)
    parsed: list[PageParse] = []
    confs: list[float] = []
    used_ocr = used_text = False
    for page in pages:
        if page.words is not None:
            words, width, reread = page.words, page.width, None
            used_text = True
        else:
            with _timed(stages, "ocr"):
                gray = prepare(page.image)
                words, width = read_words(gray), gray.shape[1]
            reread = lambda w, g=gray: reread_number(g, w)  # noqa: E731
            used_ocr = True
        if not words:
            parsed.append(PageParse(warnings=["blank_page"]))
            continue
        confs.append(sum(w.conf for w in words) / len(words) / 100)
        with _timed(stages, "parse"):
            parsed.append(parse_page(words, width, s.date_order, reread))
    if not confs:
        log.info("no text found", extra={"kind": kind, "size_kb": size_kb, "pages": len(pages), "stages_ms": stages})
        raise ExtractionError("No text could be read from the document")

    p = _merge(parsed)
    sm = dict(p.summary)
    sm["other_charges"] = [Charge(**c) for c in sm.get("other_charges", [])]
    result = InvoiceData(
        invoice=InvoiceMeta(invoice_number=p.number, date=p.date, due_date=p.due_date, currency=p.currency),
        seller=Party(**p.seller),
        client=Party(**p.client),
        items=[LineItem(**i) for i in p.items],
        summary=Summary(**sm),
        payment=Payment(**p.payment),
        meta=ExtractionMeta(
            status="ok", confidence=0, pages=len(parsed), engine=_engine_name(used_ocr, used_text), processing_ms=0
        ),
    )
    ocr_conf = sum(confs) / len(confs)
    _score(result, ocr_conf, p.warnings, [], t0)
    rules_status = result.meta.status

    mode = llm_mode or s.llm_mode
    llm_outcome = "off" if mode == "off" else "not_needed"
    if llm_stage.should_call(mode, result):
        try:
            with _timed(stages, "llm"):
                answer = llm_stage.extract_with_llm(data)
        except llm_stage.LLMSkipped as e:
            llm_outcome = e.code.removeprefix("llm_")  # not_configured | quota_exhausted | failed
            if mode == "always" or e.code != "llm_not_configured":  # in auto mode an unconfigured LLM is not news
                result.meta.warnings.append(e.code)
        except ExtractionError:
            llm_outcome = "failed"
            result.meta.warnings.append("llm_failed")
        else:
            trusted = result.meta.status == "ok"
            merged, fields = merge(result, answer, trusted)
            llm_outcome = "used" if fields else "no_change"
            if fields:
                merged.meta = result.meta.model_copy(update={"llm_used": True, "llm_fields": fields})
                merged.meta.engine = f"{result.meta.engine}+{s.llm_model}"
                _score(merged, LLM_BASE_CONFIDENCE, p.warnings, ["llm_filled"], t0)
                result = merged

    log.info(
        "extracted",
        extra={
            "kind": kind,
            "size_kb": size_kb,
            "pages": result.meta.pages,
            "engine": result.meta.engine,
            "status": result.meta.status,
            "rules_status": rules_status,
            "confidence": result.meta.confidence,
            "items": len(result.items),
            "warnings": result.meta.warnings,
            "llm": llm_outcome,
            "llm_fields": result.meta.llm_fields,
            "stages_ms": stages,
            "ms": result.meta.processing_ms,
        },
    )
    return result
