from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .models import Chunk, Document


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)")


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start:self.end]


def _split_sections(text: str) -> list[tuple[tuple[str, ...], int, int]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        stripped_start = len(text) - len(text.lstrip())
        stripped_end = len(text.rstrip())
        return [(('General',), stripped_start, stripped_end)] if stripped_start < stripped_end else []

    sections: list[tuple[tuple[str, ...], int, int]] = []
    heading_stack: list[str] = []
    intro_end = matches[0].start()
    if text[:intro_end].strip():
        start = len(text[:intro_end]) - len(text[:intro_end].lstrip())
        end = len(text[:intro_end].rstrip())
        sections.append((('General',), start, end))

    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading_stack = heading_stack[:level - 1]
        heading_stack.append(match.group(2).strip())
        raw_start = match.end()
        raw_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[raw_start:raw_end]
        start = raw_start + len(body) - len(body.lstrip())
        end = raw_start + len(body.rstrip())
        if start < end:
            sections.append((tuple(heading_stack), start, end))
    return sections


def _sentence_spans(text: str, start: int, end: int, max_words: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for line_match in re.finditer(r"[^\n]+", text[start:end]):
        line_start = start + line_match.start()
        line = line_match.group(0)
        for sentence in SENTENCE_RE.finditer(line):
            raw_start = line_start + sentence.start()
            raw_end = line_start + sentence.end()
            piece = text[raw_start:raw_end]
            left = len(piece) - len(piece.lstrip())
            right = len(piece.rstrip())
            sentence_start, sentence_end = raw_start + left, raw_start + right
            if sentence_start >= sentence_end:
                continue
            words = list(re.finditer(r"\S+", text[sentence_start:sentence_end]))
            if len(words) <= max_words:
                spans.append(TextSpan(sentence_start, sentence_end))
                continue
            for offset in range(0, len(words), max_words):
                group = words[offset:offset + max_words]
                spans.append(TextSpan(sentence_start + group[0].start(), sentence_start + group[-1].end()))
    return spans


def _pack_spans(text: str, spans: list[TextSpan], max_words: int, overlap_words: int) -> list[TextSpan]:
    packed: list[TextSpan] = []
    start = 0
    word_counts = [len(span.text(text).split()) for span in spans]
    while start < len(spans):
        end = start
        words = 0
        while end < len(spans) and words + word_counts[end] <= max_words:
            words += word_counts[end]
            end += 1
        if end == start:  # Defensive: long units are normally split by _sentence_spans.
            end += 1
        packed.append(TextSpan(spans[start].start, spans[end - 1].end))
        if end == len(spans):
            break
        next_start = end
        overlap = 0
        for index in range(end - 1, start - 1, -1):
            if overlap + word_counts[index] > overlap_words:
                break
            overlap += word_counts[index]
            next_start = index
        start = next_start if next_start > start else end
    return packed


def _stable_chunk_id(doc: Document, heading_path: tuple[str, ...], text: str) -> str:
    identity = "\n".join((doc.document_id, doc.version, " > ".join(heading_path), text))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{doc.document_id}::{digest}"


def chunk_document(doc: Document, max_words: int = 90, overlap_words: int = 20) -> list[Chunk]:
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be non-negative and smaller than max_words")

    chunks: list[Chunk] = []
    for heading_path, section_start, section_end in _split_sections(doc.text):
        sentences = _sentence_spans(doc.text, section_start, section_end, max_words)
        for span in _pack_spans(doc.text, sentences, max_words, overlap_words):
            chunk_text = span.text(doc.text)
            chunks.append(Chunk(
                chunk_id=_stable_chunk_id(doc, heading_path, chunk_text),
                document_id=doc.document_id,
                title=doc.title,
                department=doc.department,
                version=doc.version,
                effective_date=doc.effective_date,
                access_roles=doc.access_roles,
                section=heading_path[-1],
                tags=doc.tags,
                text=chunk_text,
                heading_path=heading_path,
                source_path=doc.path,
                start_char=span.start,
                end_char=span.end,
                effective_from=doc.effective_from,
                effective_until=doc.effective_until,
                supersedes=doc.supersedes,
                status=doc.status,
                policy_family=doc.policy_family,
                policy_scope=doc.policy_scope,
                applies_to_customer=doc.applies_to_customer,
                signed=doc.signed,
            ))
    return chunks


def chunk_corpus(
    docs: list[Document], max_words: int = 90, overlap_words: int = 20
) -> list[Chunk]:
    return [
        chunk
        for document in docs
        for chunk in chunk_document(document, max_words=max_words, overlap_words=overlap_words)
    ]
