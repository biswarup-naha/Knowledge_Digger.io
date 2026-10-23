
import pytest

from src.services.vector_store_service import VectorStoreService


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
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(len(text)),
                float(sum(ord(char) for char in text) % 100 + 1),
                1.0,
            ]
            for text in texts
        ]


@pytest.fixture
def vector_store(tmp_path):
    service = VectorStoreService(
        embedding_service=FakeEmbeddingService(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_knowledge_documents",
    )
    return service


def get_document_chunks(vector_store, user_id: str, document_id: str):
    result = vector_store.collection.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id},
            ]
        },
        include=["documents", "metadatas"],
    )
    return result


def test_add_chunks_stores_all_chunks(vector_store):
    chunks = ["First chunk", "Second chunk", "Third chunk"]

    result = vector_store.add_chunks("user-1", "doc-1", chunks)

    stored = get_document_chunks(vector_store, "user-1", "doc-1")

    assert result == 3
    assert len(stored["ids"]) == 3
    assert set(stored["documents"] or []) == set(chunks)


def test_reindexing_removes_stale_chunks(vector_store):
    vector_store.add_chunks(
        "user-1",
        "doc-1",
        ["Original one", "Original two", "Original three", "Original four"],
    )

    updated_chunks = ["Updated one", "Updated two"]

    result = vector_store.add_chunks("user-1", "doc-1", updated_chunks)
    stored = get_document_chunks(vector_store, "user-1", "doc-1")

    assert result == 2
    assert len(stored["ids"]) == 2
    assert set(stored["documents"] or []) == set(updated_chunks)


def test_reindexing_updates_existing_chunk_content(vector_store):
    vector_store.add_chunks("user-1", "doc-1", ["Original content"])

    vector_store.add_chunks("user-1", "doc-1", ["Revised content"])

    stored = get_document_chunks(vector_store, "user-1", "doc-1")

    assert len(stored["ids"]) == 1
    assert stored["documents"] == ["Revised content"]


def test_reindexing_does_not_affect_other_documents(vector_store):
    vector_store.add_chunks("user-1", "doc-1", ["Document one"])
    vector_store.add_chunks("user-1", "doc-2", ["Document two"])
    vector_store.add_chunks("user-1", "doc-1", ["Updated document one"])

    other_document = get_document_chunks(vector_store, "user-1", "doc-2")

    assert other_document["documents"] == ["Document two"]


def test_reindexing_does_not_affect_another_user(vector_store):
    vector_store.add_chunks("user-1", "doc-1", ["User one content"])
    vector_store.add_chunks("user-2", "doc-1", ["User two content"])

    vector_store.add_chunks(
        "user-1",
        "doc-1",
        ["Updated user one content", "Another chunk"],
    )

    user_one = get_document_chunks(vector_store, "user-1", "doc-1")
    user_two = get_document_chunks(vector_store, "user-2", "doc-1")

    assert len(user_one["ids"]) == 2
    assert user_two["documents"] == ["User two content"]


def test_delete_document_deletes_only_matching_document(vector_store):
    vector_store.add_chunks("user-1", "doc-1", ["One", "Two"])
    vector_store.add_chunks("user-1", "doc-2", ["Keep this"])
    vector_store.add_chunks("user-2", "doc-1", ["Keep this too"])

    deleted_count = vector_store.delete_document("user-1", "doc-1")

    assert deleted_count == 2
    assert get_document_chunks(vector_store, "user-1", "doc-1")["ids"] == []
    assert get_document_chunks(vector_store, "user-1", "doc-2")["documents"] == [
        "Keep this"
    ]
    assert get_document_chunks(vector_store, "user-2", "doc-1")["documents"] == [
        "Keep this too"
    ]


def test_delete_document_returns_zero_when_not_found(vector_store):
    assert vector_store.delete_document("user-1", "missing-doc") == 0


def test_add_chunks_rejects_blank_user_id(vector_store):
    with pytest.raises(ValueError):
        vector_store.add_chunks("", "doc-1", ["Some content"])


def test_add_chunks_rejects_blank_document_id(vector_store):
    with pytest.raises(ValueError):
        vector_store.add_chunks("user-1", " ", ["Some content"])


def test_add_chunks_rejects_empty_chunk_content(vector_store):
    with pytest.raises(ValueError):
        vector_store.add_chunks("user-1", "doc-1", ["Valid content", "  "])


def test_add_chunks_empty_list_is_noop(vector_store):
    result = vector_store.add_chunks("user-1", "doc-1", [])

    assert result == 0
    assert get_document_chunks(vector_store, "user-1", "doc-1")["ids"] == []
