
import pytest

from src.services.query_service import QueryService


class FakeVectorStoreService:
    def __init__(self) -> None:
        self.calls = []

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        self.calls.append(
            {
                "user_id": user_id,
                "query": query,
                "top_k": top_k,
            }
        )

        return [
            {
                "rank": 1,
                "text": "FastAPI is a Python web framework.",
                "metadata": {"document_id": "doc-1"},
                "score": 0.91,
            },
            {
                "rank": 2,
                "text": "It supports automatic API documentation.",
                "metadata": {"document_id": "doc-2"},
                "score": 0.82,
            },
        ]


class FakeGenerationService:
    def __init__(self) -> None:
        self.calls = []

    def generate_answer(
        self,
        question: str,
        chunks: list[dict],
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "chunks": chunks,
            }
        )

        return "FastAPI is a Python web framework. [1]"


def test_query_returns_answer_results_and_context() -> None:
    vector_store_service = FakeVectorStoreService()
    generation_service = FakeGenerationService()

    service = QueryService(
        vector_store_service,  # type: ignore[arg-type]
        generation_service,  # type: ignore[arg-type]
    )

    result = service.query(
        user_id="user-1",
        query="What is FastAPI?",
        top_k=2,
    )

    assert result["query"] == "What is FastAPI?"
    assert result["answer"] == "FastAPI is a Python web framework. [1]"
    assert len(result["results"]) == 2
    assert result["context"] == (
        "FastAPI is a Python web framework.\n\n"
        "It supports automatic API documentation."
    )


def test_query_passes_arguments_to_vector_store() -> None:
    vector_store_service = FakeVectorStoreService()
    generation_service = FakeGenerationService()

    service = QueryService(
        vector_store_service,  # type: ignore[arg-type]
        generation_service,  # type: ignore[arg-type]
    )

    service.query("user-1", "What is FastAPI?", 2)

    assert vector_store_service.calls == [
        {
            "user_id": "user-1",
            "query": "What is FastAPI?",
            "top_k": 2,
        }
    ]


def test_query_passes_retrieved_chunks_to_generation_service() -> None:
    vector_store_service = FakeVectorStoreService()
    generation_service = FakeGenerationService()

    service = QueryService(
        vector_store_service,  # type: ignore[arg-type]
        generation_service,  # type: ignore[arg-type]
    )

    service.query("user-1", "What is FastAPI?", 2)

    assert generation_service.calls == [
        {
            "question": "What is FastAPI?",
            "chunks": vector_store_service.search(
                "user-1",
                "What is FastAPI?",
                2,
            ),
        }
    ]


@pytest.mark.parametrize(
    ("user_id", "query", "top_k"),
    [
        ("", "What is FastAPI?", 3),
        ("user-1", "   ", 3),
        ("user-1", "What is FastAPI?", 0),
        ("user-1", "What is FastAPI?", -1),
        ("user-1", "What is FastAPI?", True),
    ],
)
def test_query_rejects_invalid_inputs(
    user_id: str,
    query: str,
    top_k: int,
) -> None:
    service = QueryService(
        FakeVectorStoreService(),  # type: ignore[arg-type]
        FakeGenerationService(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        service.query(user_id, query, top_k)


def test_query_handles_no_results() -> None:
    class EmptyVectorStoreService:
        def search(
            self,
            user_id: str,
            query: str,
            top_k: int = 3,
        ) -> list[dict]:
            return []

    generation_service = FakeGenerationService()

    service = QueryService(
        EmptyVectorStoreService(),  # type: ignore[arg-type]
        generation_service,  # type: ignore[arg-type]
    )

    result = service.query("user-1", "Unknown topic")

    assert result["results"] == []
    assert result["context"] == ""
    assert result["answer"] == "FastAPI is a Python web framework. [1]"
    assert generation_service.calls[0]["chunks"] == []
