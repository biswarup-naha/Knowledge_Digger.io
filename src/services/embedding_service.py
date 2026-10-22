

from typing import Any
from src.config.app_config import get_settings
import numpy as np


class EmbeddingService:
    """
    Provides the following implementation:-
    1. Lazy loading: the model is loaded only when you first request an embedding. This helps avoid loading a large model unnecessarily when the API starts.
    2. Batch processing: embed_texts() processes multiple chunks together, which is more efficient than calling the model separately for each chunk.
    3. Normalization: normalize_embeddings=True scales the vectors to unit length, making cosine similarity and dot-product comparisons easier to use consistently.
    4. Validation: the service checks that the result has the expected dimensions and contains finite numerical values.
    5. The model's default device selection can use supported acceleration when available. We can explicitly configure CPU or CUDA later.
    """
    def __init__(
        self,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.device = device
        self.model: Any | None = None

    def _get_model(self) -> Any:
        """
        Returns the specific model with optional CUDA support
        """
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            if self.device is None:
                self.model = SentenceTransformer(self.model_name)
            else:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
        return self.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Converts chunks of text into a matrix of embeddings
        """
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings")

        model = self._get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if embeddings.shape != (len(texts), self.dimensions):
            raise RuntimeError("The embedding model returned an unexpected shape")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("The embedding model returned invalid values")
    
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Returns the first embedded text from all embeddings"""
        return self.embed_texts([text])[0]
