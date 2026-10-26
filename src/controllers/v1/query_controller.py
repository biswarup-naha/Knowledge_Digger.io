
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, HTTPException
from src.schemas.query import QueryRequest, QueryResponse
from src.services.query_service import QueryService


def create_query_router(
    query_service: QueryService,
    get_current_user_id: Callable,
) -> APIRouter:
    router = APIRouter(
        prefix="/query",
        tags=["Query"],
    )

    @router.post("", response_model=QueryResponse)
    def query_documents(
        request: QueryRequest,
        user_id: Annotated[str, Depends(get_current_user_id)],
    ) -> QueryResponse:
        try:
            result = query_service.query(
                user_id=user_id,
                query=request.query,
                top_k=request.top_k,
            )
            return QueryResponse(**result)

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        except RuntimeError as error:
            raise HTTPException(
                status_code=502,
                detail="The answer-generation service is unavailable.",
            ) from error

    return router
