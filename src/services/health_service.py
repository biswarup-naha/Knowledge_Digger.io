from src.config.app_config import get_settings
from src.schemas.health import HealthResponse


class HealthService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=self._settings.APP_NAME,
            environment=self._settings.ENVIRONMENT,
        )
