"""Content-safe structured logging and in-process operational counters."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware


class CloudJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            payload.update(event_data)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(CloudJsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


@dataclass
class ServiceMetrics:
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    status_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.request_count += 1
            self.error_count += int(status_code >= 500)
            self.total_latency_ms += latency_ms
            self.status_counts[str(status_code)] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "mean_latency_ms": (
                    self.total_latency_ms / self.request_count if self.request_count else 0.0
                ),
                "status_counts": dict(self.status_counts),
            }


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: ServiceMetrics):
        super().__init__(app)
        self.metrics = metrics
        self.logger = logging.getLogger("acmeflow_rag.http")

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.metrics.observe(status_code, latency_ms)
            self.logger.info(
                "http_request",
                extra={"event_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 3),
                }},
            )
