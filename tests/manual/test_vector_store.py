
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import VectorStoreService


embedding_service = EmbeddingService()

store = VectorStoreService(
    embedding_service=embedding_service,
    persist_directory="data/chroma-test",
    collection_name="knowledge_test",
)

#region Add document
chunks = [
    "To reset your password, select Forgot Password on the login page.",
    "Contact support if account verification fails.",
    "The dashboard supports dark mode and light mode.",
    "To recover account access, follow the password reset instructions.",
]

stored_count = store.add_chunks(
    user_id="test-user-1",
    document_id="test-document-1",
    chunks=chunks
)

print("Chunks stored:", stored_count)

#endregion

#region Search documents
results = store.search(
    user_id="test-user-1",
    query="How can I regain access to my account?",
    top_k=3,
)

for result in results:
    print(f"\nRank: {result['rank']}")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Metadata: {result['metadata']}")
    print(f"Text: {result['text']}")

other_user_results = store.search(
    user_id="different-test-user",
    query="How can I regain access to my account?",
    top_k=3,
)

print("Results for another user:", other_user_results)

#endregion

# region Delete document
deleted_count = store.delete_document(
    user_id="test-user-1",
    document_id="test-document-1",
)

print("\nDeleted chunks:", deleted_count)

remaining_results = store.search(
    user_id="test-user-1",
    query="How can I regain access to my account?",
    top_k=3,
)

print("Results after deletion:", remaining_results)

#endregion