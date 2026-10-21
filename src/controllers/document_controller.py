
from pathlib import PurePosixPath, PureWindowsPath

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.services.document_service import (
    DocumentExtractionService,
)


MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def create_document_router(
    document_service: DocumentExtractionService,
) -> APIRouter:
    router = APIRouter(
        prefix="/documents",
        tags=["Documents"],
    )

    @router.post("/extract")
    async def extract_document(
        file: UploadFile = File(...),
    ) -> dict:
        if not file.filename or not file.filename.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A filename is required",
            )

        filename = (
            PureWindowsPath(file.filename).name
            if "\\" in file.filename
            else PurePosixPath(file.filename).name
        )

        extension = PurePosixPath(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Supported formats are PDF, DOCX, and TXT",
            )

        try:
            content = await file.read(MAX_FILE_SIZE + 1)

            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded file is empty",
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds the 10 MB limit",
                )

            try:
                extracted_text = document_service.extract_data(
                    filename,
                    content,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="The document could not be parsed",
                ) from exc

            return {
                "filename": filename,
                "file_type": extension.lstrip("."),
                "size_bytes": len(content),
                "character_count": len(extracted_text),
                "text": extracted_text,
            }

        finally:
            await file.close()

    return router
