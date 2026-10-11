"""Strategy-aware evidence assessment.

Scores in this module are retrieval relevance signals, never confidence probabilities.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .models import RetrievalResult
from .pipeline import normalize_strategy


WORD = re.compile(r"[a-z0-9]+")
HISTORICAL = re.compile(r"\b(before|previous|historical|prior|old|formerly)\b", re.I)
CURRENT = re.compile(r"\b(current|currently|latest|now|today|2026)\b", re.I)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "can", "do", "does",
    "for", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "which", "with",
}


@dataclass(frozen=True)
class EvidencePolicy:
    strategy: str
    minimum_top_score: float
    minimum_query_term_coverage: float = 0.20
    conflict_score_window: float = 0.01

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


DEFAULT_POLICIES = {
    "lexical": EvidencePolicy(
        "lexical", minimum_top_score=0.10, minimum_query_term_coverage=0.50
    ),
    "dense": EvidencePolicy(
        "dense", minimum_top_score=0.31, minimum_query_term_coverage=0.30
    ),
    "hybrid": EvidencePolicy(
        "hybrid", minimum_top_score=0.015, minimum_query_term_coverage=0.50
    ),
    "hybrid_rerank": EvidencePolicy(
        "hybrid_rerank", minimum_top_score=-2.0, minimum_query_term_coverage=0.20,
        conflict_score_window=1.0,
    ),
}


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason_code: str
    strategy: str
    relevance_signals: dict[str, float | int | bool]
    policy: EvidencePolicy

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "reason_code": self.reason_code,
            "strategy": self.strategy,
            "relevance_signals": self.relevance_signals,
            "policy": self.policy.to_dict(),
            "score_semantics": "retrieval relevance signals; not calibrated probabilities",
        }


def query_term_coverage(question: str, results: Iterable[RetrievalResult]) -> float:
    query_terms = {word for word in WORD.findall(question.lower()) if word not in STOP_WORDS}
    if not query_terms:
        return 0.0
    evidence_terms: set[str] = set()
    for result in results:
        evidence_terms.update(WORD.findall(result.chunk.text.lower()))
        evidence_terms.update(WORD.findall(result.chunk.title.lower()))
        evidence_terms.update(WORD.findall(result.chunk.section.lower()))
    return len(query_terms.intersection(evidence_terms)) / len(query_terms)


def _has_version_conflict(question: str, results: list[RetrievalResult], window: float) -> bool:
    if HISTORICAL.search(question) or not CURRENT.search(question):
        return False
    by_title: dict[str, list[RetrievalResult]] = {}
    for result in results:
        key = re.sub(r"\b20\d{2}\b", "", result.chunk.title.lower()).strip()
        by_title.setdefault(key, []).append(result)
    for same_title in by_title.values():
        versions = {item.chunk.version for item in same_title}
        if len(versions) > 1 and max(item.score for item in same_title) - min(
            item.score for item in same_title
        ) <= window + 1e-12:
            return True
    return False


class EvidenceAssessor:
    def __init__(self, policies: dict[str, EvidencePolicy] | None = None):
        self.policies = policies or DEFAULT_POLICIES

    def assess(
        self, question: str, results: list[RetrievalResult], strategy: str
    ) -> EvidenceAssessment:
        strategy = normalize_strategy(strategy)
        policy = self.policies[strategy]
        top_score = results[0].score if results else 0.0
        coverage = query_term_coverage(question, results[:3])
        signals: dict[str, float | int | bool] = {
            "result_count": len(results),
            "top_relevance_score": float(top_score),
            "query_term_coverage": coverage,
            "potential_version_conflict": False,
        }
        if not results:
            return EvidenceAssessment(False, "NO_RETRIEVAL_RESULTS", strategy, signals, policy)
        if top_score < policy.minimum_top_score:
            return EvidenceAssessment(False, "LOW_RELEVANCE_SCORE", strategy, signals, policy)
        if coverage < policy.minimum_query_term_coverage:
            return EvidenceAssessment(False, "LOW_QUERY_EVIDENCE_OVERLAP", strategy, signals, policy)
        conflict = _has_version_conflict(question, results[:5], policy.conflict_score_window)
        signals["potential_version_conflict"] = conflict
        if conflict:
            return EvidenceAssessment(False, "CONTRADICTORY_VERSION_EVIDENCE", strategy, signals, policy)
        return EvidenceAssessment(True, "EVIDENCE_SUFFICIENT", strategy, signals, policy)
