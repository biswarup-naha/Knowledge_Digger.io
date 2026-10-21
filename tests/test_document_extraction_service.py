from io import BytesIO

import pytest
from docx import Document

from src.services.document_service import (
    DocumentService,
)


@pytest.fixture
def service():
    return DocumentService()


def test_extract_txt(service):
    content = b"KnowledgeDigger extracts document text."
    result = service.extract_data("notes.txt", content)
    assert result == "KnowledgeDigger extracts document text."


def test_extract_docx(service):
    document = Document()
    document.add_paragraph("Document extraction works.")
    buffer = BytesIO()
    document.save(buffer)
    result = service.extract_data("notes.docx", buffer.getvalue())
    assert result == "Document extraction works."


def test_reject_unsupported_extension(service):
    with pytest.raises(ValueError, match="Unsupported file type"):
        service.extract_data("notes.csv", b"some,data")


def test_reject_empty_file(service):
    with pytest.raises(ValueError, match="uploaded file is empty"):
        service.extract_data("notes.txt", b"")


def test_reject_invalid_txt_encoding(service):
    with pytest.raises(ValueError, match="UTF-8"):
        service.extract_data("notes.txt", b"\xff\xfe\xfa")


def test_reject_txt_without_text(service):
    with pytest.raises(ValueError, match="No extractable text"):
        service.extract_data("notes.txt", b"   \n  ")
