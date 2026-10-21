from src.app import get_app_instance
import uvicorn

app = get_app_instance()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
