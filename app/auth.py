import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Depends(_header), settings: Settings = Depends(get_settings)) -> None:
    allowed = settings.api_key_set
    if not allowed:  # auth disabled (local dev)
        return
    if not key or not any(hmac.compare_digest(key, k) for k in allowed):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})
