
import pytest

from src.services.document_chunking_service import (
    DocumentChunkingService,
)


def test_empty_text_returns_no_chunks():
    service = DocumentChunkingService()

    assert service.get_chunks_with_overlap("") == []
    assert service.get_chunks_with_overlap("   \n ") == []


def test_short_text_returns_one_chunk():
    service = DocumentChunkingService(chunk_size=5, chunk_overlap_fraction=0.1)
    chunks = service.get_chunks_with_overlap("One two three")
    assert chunks == ["One two three"]


def test_long_text_is_split_into_chunks():
    service = DocumentChunkingService(chunk_size=5, chunk_overlap_fraction=0.1)
    text = "one two three four five six seven eight nine"
    chunks = service.get_chunks_with_overlap(text)
    assert len(chunks) == 2
    assert chunks[0] == "one two three four five"
    assert chunks[1] == "five six seven eight nine"


def test_chunks_preserve_overlap():
    service = DocumentChunkingService(chunk_size=4, chunk_overlap_fraction=0.2)
    chunks = service.get_chunks_with_overlap("one two three four five six seven eight")
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-2:] == second_words[:2]


def test_chunk_size_is_respected():
    service = DocumentChunkingService(chunk_size=4, chunk_overlap_fraction=0.1)
    chunks = service.get_chunks_with_overlap("one two three four five six seven eight nine")

    assert all(len(chunk.split()) <= 4 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap_fraction"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_configuration_is_rejected(chunk_size, chunk_overlap_fraction):
    with pytest.raises(ValueError):
        DocumentChunkingService(
            chunk_size=chunk_size,
            chunk_overlap_fraction=chunk_overlap_fraction,
        )
