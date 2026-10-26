from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=3, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    results: list[dict]
    context: str