from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    department: str
    version: str
    effective_date: str
    access_roles: tuple[str, ...]
    tags: tuple[str, ...]
    path: str
    text: str
    effective_from: str = ""
    effective_until: str | None = None
    supersedes: tuple[str, ...] = ()
    status: str = "active"
    policy_family: str = ""
    policy_scope: str = "standard"
    applies_to_customer: str | None = None
    signed: bool = False
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    department: str
    version: str
    effective_date: str
    access_roles: tuple[str, ...]
    section: str
    tags: tuple[str, ...]
    text: str
    heading_path: tuple[str, ...] = ()
    source_path: str = ""
    start_char: int = 0
    end_char: int = 0
    effective_from: str = ""
    effective_until: str | None = None
    supersedes: tuple[str, ...] = ()
    status: str = "active"
    policy_family: str = ""
    policy_scope: str = "standard"
    applies_to_customer: str | None = None
    signed: bool = False
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    def to_dict(self) -> dict[str, Any]: return {"chunk": self.chunk.to_dict(), "score": self.score}
