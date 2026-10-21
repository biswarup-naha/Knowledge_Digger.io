
from pathlib import Path
from io import BytesIO

from docx import Document
from pypdf import PdfReader


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
