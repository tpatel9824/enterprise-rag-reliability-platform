from dataclasses import replace

from acmeflow_rag.chunker import chunk_document
from acmeflow_rag.models import Document


def make_document(text: str) -> Document:
    return Document(
        document_id="test_document_2026",
        title="Test Document",
        department="Engineering",
        version="1.0",
        effective_date="2026-01-01",
        access_roles=("engineering",),
        tags=("test",),
        path="/portable/source.md",
        text=text,
    )


def test_short_sections_preserve_heading_paths_metadata_and_offsets():
    document = make_document(
        "# Parent\n\nIntro sentence.\n\n## Child\n\nChild sentence one. Child sentence two."
    )

    chunks = chunk_document(document, max_words=20, overlap_words=4)

    assert [chunk.heading_path for chunk in chunks] == [("Parent",), ("Parent", "Child")]
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.version == document.version for chunk in chunks)
    assert all(chunk.source_path == document.path for chunk in chunks)
    assert all(document.text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)


def test_long_section_respects_word_limit_and_overlaps_complete_sentence():
    document = make_document(
        "# Policy\n\n## Rules\n\n"
        "Alpha beta gamma delta. Echo foxtrot golf hotel. India juliet kilo lima. "
        "Mike november oscar papa."
    )

    chunks = chunk_document(document, max_words=8, overlap_words=4)

    assert len(chunks) == 3
    assert all(len(chunk.text.split()) <= 8 for chunk in chunks)
    assert chunks[0].text.endswith("Echo foxtrot golf hotel.")
    assert chunks[1].text.startswith("Echo foxtrot golf hotel.")
    assert chunks[1].text.endswith("India juliet kilo lima.")
    assert chunks[2].text.startswith("India juliet kilo lima.")


def test_oversized_sentence_is_split_only_when_required_and_content_is_covered():
    words = [f"word{index}" for index in range(23)]
    document = make_document("# Long\n\n" + " ".join(words) + ".")

    chunks = chunk_document(document, max_words=10, overlap_words=2)
    covered = " ".join(chunk.text for chunk in chunks)

    assert [len(chunk.text.split()) for chunk in chunks] == [10, 10, 3]
    assert all(word in covered for word in words)


def test_chunk_ids_are_stable_and_independent_of_absolute_source_path():
    document = make_document("# Policy\n\n## Rule\n\nStable policy sentence.")
    moved = replace(document, path="D:/another/location/source.md")

    original_ids = [chunk.chunk_id for chunk in chunk_document(document)]
    repeated_ids = [chunk.chunk_id for chunk in chunk_document(document)]
    moved_ids = [chunk.chunk_id for chunk in chunk_document(moved)]

    assert original_ids == repeated_ids == moved_ids


def test_chunk_configuration_is_validated():
    document = make_document("Short sentence.")

    for max_words, overlap_words in [(0, 0), (10, -1), (10, 10)]:
        try:
            chunk_document(document, max_words=max_words, overlap_words=overlap_words)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk configuration was accepted")

