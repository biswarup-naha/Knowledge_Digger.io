
import numpy as np
import pytest

from src.services.embedding_service import EmbeddingService



class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def service():
    instance = EmbeddingService()
    instance.model = FakeEmbeddingModel()
    return instance


def test_empty_list_returns_empty_list(service):
    assert service.embed_texts([]) == []


def test_embed_multiple_texts(service):
    embeddings = service.embed_texts(
        ["Reset my password", "Recover account access"]
    )
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_embed_single_text(service):
    embedding = service.embed_text("Knowledge retrieval")
    assert len(embedding) == 384


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_reject_empty_text(service, text):
    with pytest.raises(ValueError, match="non-empty strings"):
        service.embed_text(text)


def test_reject_non_string_input(service):
    with pytest.raises(ValueError, match="non-empty strings"):
        service.embed_texts(["valid text", None])


def test_model_is_loaded_lazily():
    service = EmbeddingService()
    assert service.model is None
