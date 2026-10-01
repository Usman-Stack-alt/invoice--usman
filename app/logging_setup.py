"""Structured logging. Every record carries the request id of the request that caused it, including records
written from the worker thread that runs the pipeline. Invoice content is never logged: only counts, codes and timings."""

import json
import logging
import sys
from contextvars import ContextVar

from app.config import get_settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_base_factory = logging.getLogRecordFactory()


def _factory(*args, **kwargs) -> logging.LogRecord:
    record = _base_factory(*args, **kwargs)
    if not hasattr(record, "request_id"):
        record.request_id = request_id_var.get()
    return record


logging.setLogRecordFactory(_factory)  # applies to every logger, in every thread


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STD}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    """Readable lines for local development: 12:01:02 INFO  pipeline  extracted  request_id=ab12 status=ok ..."""

    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        line = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} {record.name:<8} {record.getMessage()}  {extras}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging() -> None:
    s = get_settings()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(TextFormatter() if s.log_format == "text" else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(s.log_level)
