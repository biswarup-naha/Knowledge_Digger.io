from pydantic import BaseModel


class InfoResponse(BaseModel):
    max_upload_size_mb: int
    embedding_model: str
    reranker_enabled: bool
    environment: str
