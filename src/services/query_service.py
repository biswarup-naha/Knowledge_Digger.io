
from src.services.generation_service import GenerationService
from src.services.vector_store_service import VectorStoreService



class QueryService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        generation_service: GenerationService,
    ) -> None:
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service

    def query(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> dict:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        user_id = user_id.strip()
        query = query.strip()

        results = self.vector_store_service.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )

        answer = self.generation_service.generate_answer(
            question=query,
            chunks=results,
        )

        return {
            "query": query,
            "answer": answer,
            "results": results,
            "context": "\n\n".join(
                result["text"] for result in results
            ),
        }
