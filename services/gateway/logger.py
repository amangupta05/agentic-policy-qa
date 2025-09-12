# services/gateway/logger.py
from __future__ import annotations
import logging
from typing import Optional

_RID_KEY = "rid"

class _RidFilter(logging.Filter):
    """Ensure every LogRecord has a 'rid' attribute to satisfy format strings."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, _RID_KEY):
            setattr(record, _RID_KEY, "-")  # default when not provided
        return True

def configure_logging(level: int | str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s [rid=%(rid)s]"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(_RidFilter())  # <- add filter to inject rid
    root.addHandler(handler)

    # quiet noisy libs if desired
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

def get_logger(name: Optional[str] = None, rid: Optional[str] = None) -> logging.LoggerAdapter:
    base = logging.getLogger(name or "gateway")
    extra = {_RID_KEY: rid or "-"}
    return logging.LoggerAdapter(base, extra)
