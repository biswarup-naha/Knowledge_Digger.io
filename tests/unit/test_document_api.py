
from io import BytesIO

import pytest
from docx import Document
from fastapi.testclient import TestClient

from src.app import get_app_instance
from src.config.app_config import get_settings


@pytest.fixture
def client():
    app_instance = get_app_instance()
    with TestClient(app_instance) as test_client:
        yield test_client


@pytest.fixture
def upload_url():
    return f"{get_settings().API_PREFIX}/documents/extract"


def test_upload_txt_success(client, upload_url):
    content = b"KnowledgeDigger extracts document text."

    response = client.post(
        upload_url,
        files={"file": ("notes.txt", content, "text/plain")},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "notes.txt"
    assert data["file_type"] == "txt"
    assert data["text"] == content.decode()
    assert data["character_count"] == len(content.decode())
    assert data["size_bytes"] == len(content)


def test_upload_docx_success(client, upload_url):
    document = Document()
    document.add_paragraph("DOCX extraction works.")

    buffer = BytesIO()
    document.save(buffer)

    response = client.post(
        upload_url,
        files={
            "file": (
                "notes.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == "DOCX extraction works."


def test_reject_unsupported_file(client, upload_url):
    response = client.post(
        upload_url,
        files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
    )

    assert response.status_code == 415


def test_reject_empty_file(client, upload_url):
    response = client.post(
        upload_url,
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400


def test_reject_txt_without_extractable_text(client, upload_url):
    response = client.post(
        upload_url,
        files={"file": ("blank.txt", b"  \n ", "text/plain")},
    )

    assert response.status_code == 422


def test_reject_invalid_txt_encoding(client, upload_url):
    response = client.post(
        upload_url,
        files={"file": ("invalid.txt", b"\xff\xfe\xfa", "text/plain")},
    )

    assert response.status_code == 422


def test_reject_file_over_10_mb(client, upload_url):
    content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        upload_url,
        files={"file": ("large.txt", content, "text/plain")},
    )

    assert response.status_code == 413


def test_reject_malformed_pdf(client, upload_url):
    response = client.post(
        upload_url,
        files={"file": ("broken.pdf", b"not a real PDF", "application/pdf")},
    )

    assert response.status_code == 422
