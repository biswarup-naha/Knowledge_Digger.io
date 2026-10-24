
from pathlib import Path
from io import BytesIO
from docx import Document
from pypdf import PdfReader

import re

#region Document Extraction
class DocumentExtractionService:

    def extract_data(self, filename: str, content: bytes) -> str:
        """Returns a bulk text data from raw supported files"""
        if not content:
            raise ValueError("The uploaded file is empty")
        
        extension = Path(filename).suffix.lower()
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
        """Returns text data from PDF file"""
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        return self._validate_extracted_text(text)

    def _extract_docx(self, content: bytes) -> str:
        """Returns text data from Word file"""
        document = Document(BytesIO(content))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )
        return self._validate_extracted_text(text)

    def _extract_txt(self, content: bytes) -> str:
        """Returns text data from text file"""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "TXT files must use UTF-8 encoding"
            ) from exc
        return self._validate_extracted_text(text)

    def _validate_extracted_text(self, text: str) -> str:
        """Removes whitespaces from both ends of the text"""
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
        """Returns chunks of data from the raw bulk data"""
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