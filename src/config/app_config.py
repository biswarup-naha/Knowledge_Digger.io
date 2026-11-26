import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Detect runtime environment to load either .env.development or .env.production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
env_file_path = f".env.{ENVIRONMENT}"

load_dotenv(env_file_path, override=False)


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Knowledge_Digger.io"
    ENVIRONMENT: str = ENVIRONMENT
    API_PREFIX: str = Field(
        default="/api/v1",
        validation_alias=AliasChoices("API_PREFIX", "API_V1_PREFIX"),
    )
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # Auth Settings
    JWT_SECRET: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_MINUTES: int = 30

    # Upload Settings
    MAX_UPLOAD_MB: int = 10
    UPLOAD_DIR: str = "/data/uploads"

    # Database Settings
    POSTGRES_DB: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    DATABASE_URL: str = ""
    CHROMA_DB: str = ""
    CHROMA_COLLECTION: str = "knowledge_documents"

    # Redis & Vector DB Settings
    REDIS_URL: str = "redis://redis:6379/0"
    CHROMA_MODE: Literal["local", "http"] = "local"
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8000
    CHROMA_SSL: bool = False
    CHROMA_AUTH_TOKEN: str = ""
    CHROMA_COLLECTION_PREFIX: str = ""

    # AI Models & Retrieval Settings
    HF_HOME: str = "/models/huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = 384
    GENERATION_MODEL: str = ""
    GENERATION_BACKEND: str = "ollama"
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    RERANKER_MODEL: str = ""
    ENABLE_RERANKER: bool = False
    TOP_K: int = 3
    MAX_CONTEXT_CHARS: int = 12000
    MAX_ANSWER_TOKENS: int = 512
    OCR_LANGUAGE: str = "eng"

    # Direct Pydantic to read dynamically based on runtime file location
    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        extra="ignore",  # Safely ignores any undocumented variables
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper()

        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log level")

        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
