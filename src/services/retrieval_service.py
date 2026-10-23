
from typing import Protocol

import numpy as np

from src.interfaces.embeddings import EmbeddingProvider

class RetrievalService:
    def __init__(self, embedding_service: EmbeddingProvider) -> None:
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 3,
    ) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not chunks:
            return []

        if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
            raise ValueError("All chunks must be non-empty strings")

        query_vector = np.asarray(
            self.embedding_service.embed_text(query),
            dtype=np.float32,
        )

        chunk_vectors = np.asarray(
            self.embedding_service.embed_texts(chunks),
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a one-dimensional vector")

        if chunk_vectors.ndim != 2 or chunk_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("Query and chunk embedding dimensions must match")

        if not np.isfinite(query_vector).all() or not np.isfinite(chunk_vectors).all():
            raise ValueError("Embeddings must contain only finite values")

        query_norm = np.linalg.norm(query_vector)
        chunk_norms = np.linalg.norm(chunk_vectors, axis=1)

        if query_norm == 0 or np.any(chunk_norms == 0):
            raise ValueError("Embeddings must not be zero vectors")

        scores = (chunk_vectors @ query_vector) / (
            chunk_norms * query_norm
        )

        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            {
                "rank": rank,
                "chunk_index": int(index),
                "text": chunks[index],
                "score": float(scores[index]),
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
