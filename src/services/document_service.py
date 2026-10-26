
from pathlib import Path
from io import BytesIO
from uuid import uuid4
from docx import Document
from pypdf import PdfReader

import re

from src.services.vector_store_service import VectorStoreService

#region Document Extraction
class DocumentExtractionService:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def extract_data(self, filename: str, content: bytes) -> str:
        if not content:
            raise ValueError("The uploaded file is empty")
        
        extension = Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension or 'unknown'}"
            )
        
        match extension:
            case ".pdf":
                return self._extract_pdf(content)
            case ".docx":
                return self._extract_docx(content)
            case ".txt":
                return self._extract_txt(content)
            case _ :
                raise ValueError("The uploaded file is empty")

    def _extract_pdf(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        return self._validate_extracted_text(text)

    def _extract_docx(self, content: bytes) -> str:
        document = Document(BytesIO(content))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )
        return self._validate_extracted_text(text)

    def _extract_txt(self, content: bytes) -> str:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "TXT files must use UTF-8 encoding"
            ) from exc
        return self._validate_extracted_text(text)

    def _validate_extracted_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError(
                "No extractable text was found in the document"
            )
        return text
    
#endregion

#region Document Chunking 


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

#endregion

#region Document Ingestion
class DocumentIngestionService:
    def __init__(
        self,
        extraction_service: DocumentExtractionService,
        chunking_service: DocumentChunkingService,
        vector_store_service: VectorStoreService,
    ) -> None:
        self.extraction_service = extraction_service
        self.chunking_service = chunking_service
        self.vector_store_service = vector_store_service

    def ingest_document(
        self,
        user_id: str,
        filename: str,
        content: bytes,
    ) -> dict:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename must be a non-empty string")

        if not content:
            raise ValueError("The uploaded file is empty")

        user_id = user_id.strip()
        filename = filename.strip()

        extracted_text = self.extraction_service.extract_data(
            filename,
            content,
        )

        chunks = self.chunking_service.get_chunks_with_overlap(
            extracted_text
        )

        if not chunks:
            raise ValueError("No chunks could be created from the document")

        document_id = str(uuid4())

        stored_chunks = self.vector_store_service.add_chunks(
            user_id=user_id,
            document_id=document_id,
            chunks=chunks,
            filename=filename,
        )

        if stored_chunks != len(chunks):
            raise RuntimeError("Not all document chunks were stored")

        return {
            "document_id": document_id,
            "filename": filename,
            "character_count": len(extracted_text),
            "chunk_count": stored_chunks,
            "status": "indexed",
        }
#endregion