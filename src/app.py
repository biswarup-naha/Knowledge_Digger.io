from fastapi import FastAPI

from src.config.app_config import get_settings
from src.controllers.v1.document_controller import create_document_router
from src.controllers.v1.health_controller import create_health_router
from src.controllers.v1.info_controller import create_info_router
from src.controllers.v1.query_controller import create_query_router
from src.interfaces.embeddings import EmbeddingProvider
from src.services.embedding_service import EmbeddingService
from src.services.query_service import QueryService
from src.services.generation_service import GenerationService
from src.services.document_service import DocumentExtractionService
from src.services.health_service import HealthService
from src.services.info_service import InfoService
from src.services.vector_store_service import VectorStoreService


class App:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title=self.settings.APP_NAME,
            description="A private, document-grounded knowledge API",
            version="1.0.0",
            docs_url=f"{self.settings.API_PREFIX}/docs",
            redoc_url=f"{self.settings.API_PREFIX}/redoc",
            openapi_url=f"{self.settings.API_PREFIX}/openapi.json",
        )

        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI) -> None:
        health_service = HealthService()
        info_service = InfoService()
        document_service = DocumentExtractionService()
        generation_service = GenerationService()
        embedding_service = EmbeddingService()
        vector_store_service = VectorStoreService(embedding_service)
        query_service = QueryService(vector_store_service, generation_service)

        app.include_router(
            create_health_router(health_service),
            prefix=self.settings.API_PREFIX,
        )

        app.include_router(
            create_info_router(info_service),
            prefix=self.settings.API_PREFIX,
        )

        app.include_router(
            create_document_router(document_service),
            prefix=self.settings.API_PREFIX
        )

        app.include_router(
            create_query_router(query_service, lambda : "user-1"),
            prefix=self.settings.API_PREFIX
        )

def get_app_instance():
    return App().app
