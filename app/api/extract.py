import asyncio
import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.api.deps import read_upload
from app.auth import require_api_key
from app.config import get_settings
from app.pipeline import ExtractionError, extract
from app.pipeline import llm as llm_stage
from app.pipeline.convert import libreoffice_available
from app.pipeline.ocr import tesseract_version
from app.schemas import BatchItem, BatchResponse, ErrorResponse, InvoiceData

log = logging.getLogger("api")
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)], tags=["extraction"])

_sem: asyncio.Semaphore | None = None
LlmMode = Literal["auto", "always", "off"]
_LLM_DOC = (
    "LLM fallback: `auto` (default) calls the LLM only when the rules result looks unreliable, `always` calls it for every "
    "document, `off` never does. Calls are limited to the configured requests per minute/day; when the quota is spent the "
    "rules result is returned with the warning `llm_quota_exhausted`."
)
_ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty file"},
    401: {"model": ErrorResponse, "description": "Missing / invalid X-API-Key"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Not a JPEG/PNG/TIFF/WEBP/PDF/DOCX"},
    422: {"model": ErrorResponse, "description": "File is a supported type but unreadable / contains no text"},
    504: {"model": ErrorResponse, "description": "OCR or DOCX conversion timed out"},
}


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().concurrency)
    return _sem


async def _run(data: bytes, llm: str | None = None) -> InvoiceData:
    """CPU-bound pipeline off the event loop, with bounded parallelism."""
    async with _semaphore():
        return await asyncio.to_thread(extract, data, llm)


def _safe_name(name: str | None) -> str:
    return re.sub(r"[^\w.\- ]", "_", (name or "upload").split("/")[-1].split("\\")[-1])[:200]


@router.post("/extract", response_model=InvoiceData, responses=_ERRORS, summary="Extract one invoice")
async def extract_one(
    file: UploadFile = File(..., description="Invoice as JPEG, PNG, TIFF, WEBP, PDF or DOCX"),
    llm: LlmMode | None = Query(None, description=_LLM_DOC),
) -> InvoiceData:
    data = await read_upload(file)
    try:
        return await _run(data, llm)
    except ExtractionError as e:
        log.info("rejected", extra={"status": 422, "reason": str(e)})
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    except TimeoutError as e:
        log.warning("timed out", extra={"status": 504, "reason": str(e)})
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(e) or "Processing timed out") from e


@router.post(
    "/extract/batch",
    response_model=BatchResponse,
    responses={k: v for k, v in _ERRORS.items() if k in (401, 413, 415)},
    summary="Extract several invoices in one request",
)
async def extract_batch(
    files: list[UploadFile] = File(..., description="Up to MAX_BATCH_FILES files"),
    llm: LlmMode | None = Query(None, description=_LLM_DOC),
) -> BatchResponse:
    """One bad file never fails the batch: each result carries `ok` and either `data` or `error`."""
    limit = get_settings().max_batch_files
    if len(files) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"At most {limit} files per batch")

    async def one(f: UploadFile) -> BatchItem:
        name = _safe_name(f.filename)
        try:
            # Read the upload only once a processing slot is free: at most CONCURRENCY files are in memory at a time,
            # not all of them (10 x MAX_UPLOAD_MB) at once.
            async with _semaphore():
                data = await read_upload(f)
                result = await asyncio.to_thread(extract, data, llm)
            return BatchItem(filename=name, ok=True, data=result)
        except HTTPException as e:
            return BatchItem(filename=name, ok=False, error=str(e.detail))
        except ExtractionError as e:
            return BatchItem(filename=name, ok=False, error=str(e))
        except TimeoutError as e:
            return BatchItem(filename=name, ok=False, error=str(e) or "Processing timed out")
        except Exception:
            log.exception("batch item failed", extra={"file": name})
            return BatchItem(filename=name, ok=False, error="Internal error")

    return BatchResponse(results=await asyncio.gather(*(one(f) for f in files)))


health = APIRouter(tags=["health"])


@health.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"status": "ok"}


@health.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    try:
        tess = tesseract_version()
    except Exception:
        raise HTTPException(503, detail="tesseract unavailable") from None
    return {"status": "ready", "tesseract": tess, "docx_support": libreoffice_available(), "llm": llm_stage.status()}
