from src.config.app_config import get_settings
from src.models.info_model import InfoResponse


class InfoService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def get_info(self) -> InfoResponse:
        return InfoResponse(
            max_upload_size_mb=self._settings.MAX_UPLOAD_MB,
            embedding_model=self._settings.EMBEDDING_MODEL,
            reranker_enabled=self._settings.ENABLE_RERANKER,
            environment=self._settings.ENVIRONMENT,
        )
