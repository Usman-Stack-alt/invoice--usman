import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import extract
from app.config import get_settings
from app.logging_setup import request_id_var, setup_logging
from app.pipeline import llm as llm_stage

log = logging.getLogger("api")
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    s = get_settings()
    log.info(
        "started",
        extra={
            "env": s.env,
            "auth": bool(s.api_key_set),
            "cors_origins": len(s.cors_list),
            "max_upload_mb": s.max_upload_mb,
            "concurrency": s.concurrency,
            "llm": llm_stage.status(),
        },
    )
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="Invoice Extraction API",
        version="1.0.0",
        description=(
            "Upload an invoice image, PDF or DOCX, get clean structured JSON "
            "(Tesseract 5 + layout parser + arithmetic validation)."
        ),
        lifespan=lifespan,
        docs_url="/docs" if s.env != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if s.env != "production" else None,
    )
    if s.cors_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=s.cors_list,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        if not _SAFE_ID.fullmatch(rid):  # never log a client-controlled string verbatim
            rid = uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled", extra={"path": request.url.path})
            return JSONResponse(
                {"detail": "Internal server error", "request_id": rid}, status_code=500, headers={"X-Request-ID": rid}
            )
        response.headers["X-Request-ID"] = rid
        if request.url.path not in ("/healthz", "/readyz"):
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": int((time.perf_counter() - t0) * 1000),
                },
            )
        return response

    app.include_router(extract.health)
    app.include_router(extract.router)
    return app


app = create_app()
