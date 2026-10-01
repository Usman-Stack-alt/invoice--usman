"""Gemini fallback: reads the document itself and returns the same fields the rules extract.

Used only when the rules result looks unreliable (or LLM_MODE=always). It never raises into the pipeline:
every problem becomes an LLMSkipped with a code that ends up in meta.warnings, and the rules result is returned.
"""

import hashlib
import io
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.pipeline.errors import ExtractionError
from app.pipeline.preprocess import sniff
from app.pipeline.ratelimit import QuotaLimiter
from app.schemas import InvoiceData

log = logging.getLogger("llm")
PROMPT_VERSION = "1"  # bump to invalidate cached answers when the prompt or schema changes
MAX_IMAGE_SIDE = 2048


class LLMSkipped(Exception):
    """The LLM was not used. `code` is surfaced in meta.warnings."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# ---- schema Gemini must follow (flat and nullable: "absent" must be expressible) ------------------------
class LLMParty(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None
    email: str | None = None


class LLMItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    vat_percent: float | None = None
    net_amount: float | None = None
    gross_amount: float | None = None


class LLMPayment(BaseModel):
    beneficiary: str | None = None
    bank: str | None = None
    iban: str | None = None
    bic: str | None = None
    account_number: str | None = None
    reference: str | None = None


class LLMInvoice(BaseModel):
    invoice_number: str | None = None
    date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    seller: LLMParty | None = None
    client: LLMParty | None = None
    items: list[LLMItem] = []
    subtotal: float | None = None
    tax: float | None = None
    tax_rate: float | None = None
    discount: float | None = None
    total: float | None = None
    amount_paid: float | None = None
    amount_due: float | None = None
    payment: LLMPayment | None = None


PROMPT = """Extract the data of this invoice into the JSON schema.
Rules:
- Copy text exactly as printed. Use null for anything the document does not show. Never guess and never calculate.
- Dates as YYYY-MM-DD. If day and month are ambiguous, assume month first.
- Numbers as plain numbers: no currency symbols, no thousands separators (1 394,67 -> 1394.67).
- currency: a 3-letter ISO code if stated or clearly implied by the symbol and country, otherwise null.
- seller is the issuer (letterhead, 'From'). client is the billed party ('Bill to', 'To').
- items: one entry per line item, in order. A description may span several lines. net_amount is the line amount before tax;
  gross_amount only if the document shows an amount including tax.
- subtotal is the total before tax, tax the total tax, total the grand total of the invoice (not what is still owed).
  amount_paid and amount_due only if printed.
- payment holds bank transfer details if present.
- Ignore handwriting, stamps and footers that are not data."""


# ---- availability / quota -------------------------------------------------------------------------------
def _limiter() -> QuotaLimiter:
    s = get_settings()
    return QuotaLimiter(s.llm_state_dir, s.llm_rpm, s.llm_rpd)


def enabled() -> bool:
    s = get_settings()
    return bool(s.gemini_api_key) and s.llm_mode != "off"


def status() -> dict:
    s = get_settings()
    out: dict = {"mode": s.llm_mode, "enabled": enabled(), "model": s.llm_model}
    if s.gemini_api_key:
        try:
            out |= _limiter().status()
        except OSError:
            out["quota"] = "unavailable"
    return out


def should_call(mode: str, result: InvoiceData) -> bool:
    """auto: only when the rules result is unreliable. always: every document. off: never."""
    if mode == "off":
        return False
    if mode == "always":
        return True
    m, s = result.invoice, result.summary
    key_missing = not (m.invoice_number and m.date and s.total is not None and result.items)
    no_parties = not (result.seller.name or result.client.name)
    return result.meta.status == "needs_review" or key_missing or no_parties


# ---- payload / cache ------------------------------------------------------------------------------------
def _payload(data: bytes) -> tuple[bytes, str]:
    """PDF goes as is. Images are re-encoded: upright, RGB, at most 2048 px, JPEG (TIFF is unsupported by Gemini)."""
    kind = sniff(data)
    if kind == "pdf":
        return data, "application/pdf"
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (OSError, ValueError) as e:
        raise ExtractionError(f"Could not decode image: {e}") from e
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _cache_file(payload: bytes):
    s = get_settings()
    key = hashlib.sha256(payload + f"|{s.llm_model}|{PROMPT_VERSION}".encode()).hexdigest()
    return Path(s.llm_state_dir) / "cache" / f"{key}.json"


@lru_cache(maxsize=1)
def _client(api_key: str, base_url: str, timeout_s: int) -> genai.Client:
    opts = types.HttpOptions(
        timeout=timeout_s * 1000,
        retry_options=types.HttpRetryOptions(attempts=1),  # a retry would spend quota
        **({"base_url": base_url} if base_url else {}),
    )
    return genai.Client(api_key=api_key, http_options=opts)


def extract_with_llm(data: bytes) -> LLMInvoice:
    """Raises LLMSkipped(code) for anything that prevents a trustworthy answer."""
    s = get_settings()
    if not s.gemini_api_key:
        raise LLMSkipped("llm_not_configured")
    payload, mime = _payload(data)

    cache = _cache_file(payload)
    if cache.exists():  # identical document seen before: no request, no quota
        try:
            out = LLMInvoice.model_validate_json(cache.read_text())
            log.info("llm cache hit", extra={"model": s.llm_model, "quota_charged": False})
            return out
        except (OSError, ValidationError):
            cache.unlink(missing_ok=True)

    try:
        allowed = _limiter().try_acquire()
    except OSError:
        log.exception("quota state unavailable; skipping the LLM to stay within quota")
        raise LLMSkipped("llm_failed") from None
    if not allowed:
        log.warning("llm quota exhausted, request not sent", extra={**_limiter().status(), "model": s.llm_model})
        raise LLMSkipped("llm_quota_exhausted")

    t0 = time.perf_counter()
    try:
        resp = _client(s.gemini_api_key, s.gemini_base_url, s.llm_timeout_s).models.generate_content(
            model=s.llm_model,
            contents=[types.Part.from_bytes(data=payload, mime_type=mime), PROMPT],
            config=types.GenerateContentConfig(
                temperature=s.llm_temperature,
                response_mime_type="application/json",
                response_schema=LLMInvoice,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        out = LLMInvoice.model_validate(json.loads(resp.text or ""))
    except Exception as e:
        detail = {"error": type(e).__name__, "ms": int((time.perf_counter() - t0) * 1000), "model": s.llm_model}
        # Google's own error (HTTP code, status, message) is what makes a failure diagnosable; it holds no key or document data
        for attr, key in (("code", "http_status"), ("status", "google_status"), ("message", "google_message")):
            if (val := getattr(e, attr, None)) is not None:
                detail[key] = str(val)[:300]
        log.warning("gemini call failed", extra=detail)
        raise LLMSkipped("llm_failed") from None

    usage = getattr(resp, "usage_metadata", None)
    log.info(
        "llm call",
        extra={
            "model": s.llm_model,
            "mime": mime,
            "payload_kb": len(payload) // 1024,
            "ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            **_limiter().status(),
        },
    )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out.model_dump_json())
    except OSError:
        pass  # a missing cache only costs quota later
    return out
