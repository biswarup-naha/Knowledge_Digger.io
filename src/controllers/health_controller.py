from fastapi import APIRouter
from src.models.health_model import HealthResponse
from src.services.health_service import HealthService


def create_health_router(
    health_service: HealthService,
) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        return health_service.get_health()

    return router
