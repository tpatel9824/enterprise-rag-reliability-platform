"""Metadata-only audit events. Document text and query content are intentionally excluded."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True)
class AuditEvent:
    timestamp: str
    subject: str
    action: str
    decision: str
    reason_code: str
    document_ids: tuple[str, ...]
    as_of: str


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class LoggingAuditSink:
    def __init__(self):
        self.logger = logging.getLogger("acmeflow_rag.audit")

    def record(self, event: AuditEvent) -> None:
        self.logger.info(json.dumps(asdict(event), sort_keys=True))


class MemoryAuditSink:
    def __init__(self):
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def make_audit_event(subject, decision, reason_code, document_ids, as_of):
    return AuditEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        subject=subject,
        action="rag.ask",
        decision=decision,
        reason_code=reason_code,
        document_ids=tuple(document_ids),
        as_of=as_of.isoformat(),
    )
