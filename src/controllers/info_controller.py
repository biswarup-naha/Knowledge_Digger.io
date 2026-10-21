from fastapi import APIRouter
from src.models.info_model import InfoResponse
from src.services.info_service import InfoService


def create_info_router(
    info_service: InfoService,
) -> APIRouter:
    router = APIRouter(tags=["Info"])

    @router.get("/info", response_model=InfoResponse)
    def get_info() -> InfoResponse:
        return info_service.get_info()

    return router
