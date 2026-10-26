
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.services.document_service import (
    DocumentExtractionService,
    DocumentIngestionService
)


MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def create_document_router(
    extraction_service: DocumentExtractionService,
    ingestion_service: DocumentIngestionService,
    get_current_user_id: Callable
) -> APIRouter:
    router = APIRouter(
        prefix="/documents",
        tags=["Documents"],
    )

    @router.post("/upload")
    async def upload_document(
        user_id: Annotated[str, Depends(get_current_user_id)],
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
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="File size exceeds the 10 MB limit",
                )

            try:
                extracted_text = extraction_service.extract_data(
                    filename=filename,
                    content=content,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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

            
    @router.post("/extract")
    async def extract_document(
        file: UploadFile = File(...),
        user_id: str = Depends(get_current_user_id)
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
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="File size exceeds the 10 MB limit",
                )

            try:
                return ingestion_service.ingest_document(
                    user_id=user_id,
                    filename=filename,
                    content=content,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The document could not be parsed",
                ) from exc
        finally:
            await file.close()

    return router
