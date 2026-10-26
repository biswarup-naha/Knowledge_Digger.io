from pydantic import BaseModel


class ApiResponse[T](BaseModel):
    status: bool
    message: str
    data: T

class AppException(BaseException):
    pass