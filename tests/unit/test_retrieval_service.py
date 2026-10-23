
import numpy as np
import pytest

from src.services.retrieval_service import RetrievalService


class FakeEmbeddingService:
    """Provides deterministic embeddings without downloading a model."""
    vectors = {
        "account recovery question": [1.0, 0.0, 0.0],
        "recover account password": [0.9, 0.1, 0.0],
        "contact account support": [0.7, 0.7, 0.0],
        "dashboard color settings": [0.0, 0.0, 1.0],
    }

    def embed_text(self, text):
        return self.vectors[text]

    def embed_texts(self, texts):
        return [self.vectors[text] for text in texts]


@pytest.fixture
def service():
    return RetrievalService(FakeEmbeddingService())


def test_results_are_ranked_by_similarity(service):
    chunks = [
        "dashboard color settings",
        "contact account support",
        "recover account password",
    ]

    results = service.retrieve(
        "account recovery question",
        chunks,
        top_k=3,
    )

    assert results[0]["text"] == "recover account password"
    assert results[1]["text"] == "contact account support"
    assert results[2]["text"] == "dashboard color settings"
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]


def test_top_k_limits_results(service):
    chunks = [
        "dashboard color settings",
        "contact account support",
        "recover account password",
    ]

    results = service.retrieve(
        "account recovery question",
        chunks,
        top_k=2,
    )

    assert len(results) == 2


def test_results_include_rank_index_text_and_score(service):
    results = service.retrieve(
        "account recovery question",
        ["dashboard color settings", "recover account password"],
        top_k=1,
    )

    assert results[0]["rank"] == 1
    assert results[0]["chunk_index"] == 1
    assert results[0]["text"] == "recover account password"
    assert isinstance(results[0]["score"], float)


def test_empty_chunks_return_empty_results(service):
    assert service.retrieve("account recovery question", []) == []


@pytest.mark.parametrize("query", ["", "  ", "\n"])
def test_empty_query_is_rejected(service, query):
    with pytest.raises(ValueError, match="non-empty string"):
        service.retrieve(query, ["recover account password"])


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(service, top_k):
    with pytest.raises(ValueError, match="top_k"):
        service.retrieve("account recovery question", [], top_k=top_k)


def test_empty_chunk_is_rejected(service):
    with pytest.raises(ValueError, match="non-empty strings"):
        service.retrieve("account recovery question", ["   "])


def test_mismatched_embedding_dimensions_are_rejected():
    class MismatchedEmbeddingService:
        def embed_text(self, text):
            return [1.0, 0.0]

        def embed_texts(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    service = RetrievalService(MismatchedEmbeddingService())

    with pytest.raises(ValueError, match="dimensions must match"):
        service.retrieve("question", ["some document chunk"])
