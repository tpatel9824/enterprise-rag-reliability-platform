"""Grounded generators returning claims with chunk-level citations."""

from __future__ import annotations

import json
import re
from typing import Protocol

from .grounding import GeneratedAnswer, GroundedClaim
from .models import RetrievalResult


WORDS = re.compile(r"[a-z0-9]+")
STOP_WORDS = {"a", "an", "and", "are", "do", "does", "for", "how", "is", "may", "the", "to", "what", "when", "which", "who"}


class GroundedGenerator(Protocol):
    def generate(self, question: str, retrieved: list[RetrievalResult]) -> GeneratedAnswer: ...


class OfflineGenerator:
    """Deterministic extractive mode for tests and credential-free demos."""

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer("", ())
        query_terms = {word for word in WORDS.findall(question.lower()) if word not in STOP_WORDS}
        claims: list[GroundedClaim] = []
        for result in retrieved:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", result.chunk.text.strip()) if s]
            if not sentences:
                continue
            ranked = sorted(
                ((len(query_terms.intersection(WORDS.findall(sentence.lower()))), sentence) for sentence in sentences),
                key=lambda item: (-item[0], sentences.index(item[1])),
            )
            if not ranked or ranked[0][0] == 0:
                continue
            claims.append(
                GroundedClaim(f"claim-{len(claims) + 1}", ranked[0][1], (result.chunk.chunk_id,))
            )
            if len(claims) == 3:
                break
        answer = " ".join(
            f"{claim.text} [{','.join(claim.cited_chunk_ids)}]" for claim in claims
        )
        return GeneratedAnswer(answer, tuple(claims))


class GeminiGenerator:
    def __init__(self, model_name: str, project: str, location: str, client=None):
        from google import genai
        from google.genai.types import HttpOptions

        if not model_name:
            raise ValueError("GENAI_MODEL must be set when Gemini is enabled")
        if not project or not location:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are required for Vertex AI"
            )
        self.client = client or genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        self.model_name = model_name

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer("", ())
        context = "\n\n".join(
            f"CHUNK_ID={r.chunk.chunk_id}\nDOCUMENT_ID={r.chunk.document_id}\n{r.chunk.text}"
            for r in retrieved
        )
        prompt = (
            "Answer only from the supplied authorized evidence. Return JSON with keys answer and "
            "claims. Each claim must have claim_id, text, and cited_chunk_ids. Every factual claim "
            "must cite one or more exact CHUNK_ID values. Do not cite document IDs. If the evidence "
            "does not support an answer, return an empty answer and claims list.\n\n"
            f"Evidence:\n{context}\n\nQuestion:\n{question}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": 0.0, "max_output_tokens": 600, "response_mime_type": "application/json"},
        )
        payload = json.loads(response.text)
        claims = tuple(
            GroundedClaim(
                str(item["claim_id"]),
                str(item["text"]),
                tuple(str(value) for value in item["cited_chunk_ids"]),
            )
            for item in payload.get("claims", [])
        )
        return GeneratedAnswer(str(payload.get("answer", "")), claims)


def build_generator(
    use_gemini: bool, model_name: str, project: str = "", location: str = "global"
) -> GroundedGenerator:
    return GeminiGenerator(model_name, project, location) if use_gemini else OfflineGenerator()
