"""Structured grounded-answer types and citation validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import RetrievalResult


@dataclass(frozen=True)
class GroundedClaim:
    claim_id: str
    text: str
    cited_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    claims: tuple[GroundedClaim, ...]


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    invalid_chunk_ids: tuple[str, ...]
    claim_count: int
    cited_claim_count: int

    @property
    def claim_evidence_coverage(self) -> float:
        return self.cited_claim_count / self.claim_count if self.claim_count else 1.0

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "invalid_chunk_ids": list(self.invalid_chunk_ids),
            "claim_count": self.claim_count,
            "cited_claim_count": self.cited_claim_count,
            "claim_evidence_coverage": self.claim_evidence_coverage,
            "limitation": "Citation existence does not establish factual correctness or entailment.",
        }


def validate_citations(
    generated: GeneratedAnswer, authorized_context: list[RetrievalResult]
) -> CitationValidation:
    allowed = {result.chunk.chunk_id for result in authorized_context}
    cited_claims = 0
    invalid: set[str] = set()
    for claim in generated.claims:
        if claim.cited_chunk_ids:
            cited_claims += 1
        invalid.update(chunk_id for chunk_id in claim.cited_chunk_ids if chunk_id not in allowed)
    return CitationValidation(
        valid=not invalid and cited_claims == len(generated.claims),
        invalid_chunk_ids=tuple(sorted(invalid)),
        claim_count=len(generated.claims),
        cited_claim_count=cited_claims,
    )
