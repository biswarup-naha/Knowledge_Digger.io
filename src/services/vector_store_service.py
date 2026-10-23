
from pathlib import Path

import chromadb
import numpy as np
from src.services.embedding_service import EmbeddingService


class VectorStoreService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_directory: str = "data/chroma",
        collection_name: str = "knowledge_documents",
    ) -> None:
        self.embedding_service = embedding_service

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: list[str],
    ) -> int:
        """Inserts chunks as documents to the database"""
        if not user_id.strip() or not document_id.strip():
            raise ValueError("User ID and document ID are required")
        if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
            raise ValueError("All chunks must be non-empty strings")
        if not chunks:
            return 0

        embeddings = np.asarray(
            self.embedding_service.embed_texts(chunks),
            dtype=np.float32,
        )

        ids = [
            f"{user_id}:{document_id}:{index}"
            for index in range(len(chunks))
        ]

        metadatas: list[chromadb.Metadata] = [
            {
                "user_id": user_id,
                "document_id": document_id,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Searches the database for documents related to the query bound by top_k records
        """

        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        documents_result = results["documents"]
        metadatas_result = results["metadatas"]
        distances_result = results["distances"]

        if (
            documents_result is None
            or metadatas_result is None
            or distances_result is None
            or not documents_result
            or not metadatas_result
            or not distances_result
        ):
            return []

        documents = documents_result[0]
        metadatas = metadatas_result[0]
        distances = distances_result[0]

        if documents is None or metadatas is None or distances is None:
            return []

        return [
            {
                "rank": index + 1,
                "text": document,
                "metadata": metadata,
                "score": 1.0 - distance,
            }
            for index, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances)
            )
        ]


    def delete_document(self, user_id: str, document_id: str) -> int:
        """
        Deletes a document from the database with the document_id
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID must be a non-empty string")

        matching_chunks = self.collection.get(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id},
                ]
            },
            include=[],
        )
        chunk_ids = matching_chunks["ids"]

        if not chunk_ids:
            return 0
        
        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)
