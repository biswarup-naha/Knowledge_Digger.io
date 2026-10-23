

from src.services.embedding_service import EmbeddingService


service = EmbeddingService()

texts = [
    "How do I reset my password?",
    "How can I recover access to my account?",
    "The solar system contains eight planets.",
]

embeddings = service.embed_texts(texts)

print("Number of embeddings:", len(embeddings))
print("Dimensions:", len(embeddings[0]))

for text, embedding in zip(texts, embeddings):
    print(f"\nText: {text}")
    print("First five values:", embedding[:5])