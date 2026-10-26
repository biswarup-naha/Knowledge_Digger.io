from datetime import datetime
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: str
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
