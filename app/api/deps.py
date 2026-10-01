import logging

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings
from app.pipeline.errors import ExtractionError
from app.pipeline.preprocess import sniff

log = logging.getLogger("api")


async def read_upload(file: UploadFile) -> bytes:
    """Read with a hard size cap, then verify the real type from magic bytes (not the client's header)."""
    limit = get_settings().max_upload_mb * 1024 * 1024
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            log.info("rejected", extra={"status": 413, "reason": "file too large", "limit_mb": get_settings().max_upload_mb})
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {get_settings().max_upload_mb} MB")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        log.info("rejected", extra={"status": 400, "reason": "empty file"})
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    try:
        sniff(data)
    except ExtractionError as e:
        log.info("rejected", extra={"status": 415, "reason": str(e), "size_bytes": len(data)})
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)) from e
    return data
