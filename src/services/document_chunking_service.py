
import re


class DocumentChunkingService:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap_fraction: float = 0.1,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")
        if chunk_overlap_fraction < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if chunk_overlap_fraction >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_overlap_fraction * 10) if int(chunk_overlap_fraction * chunk_size) == 0 else int(chunk_overlap_fraction * chunk_size)

    def get_chunks_with_overlap(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        words = re.findall(r"\S+", text)
        if not words:
            return []
        chunks = []
        start = 0
        # sliding window
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - self.chunk_overlap
        return chunks
