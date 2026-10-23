from functools import lru_cache
import os
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


# Detect runtime environment to load either .env.development or .env.production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
env_file_path = f".env.{ENVIRONMENT}"

load_dotenv(env_file_path, override=False)


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Knowledge_Digger.io"
    ENVIRONMENT: str = ENVIRONMENT
    API_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v")
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # Auth Settings
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "")
    ACCESS_TOKEN_MINUTES: int = (int)(os.getenv("ACCESS_TOKEN_MINUTES", "5"))

    # Upload Settings
    MAX_UPLOAD_MB: int = (int)(os.getenv("MAX_UPLOAD_MB", "5"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "")

    # Database Settings
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    CHROMA_DB: str = os.getenv("CHROMA_DB", "")
    CHROMA_DB: str = os.getenv("CHROMA_COLLECTION", "")

    # Redis & Vector DB Settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chroma")
    CHROMA_PORT: int = (int)(os.getenv("CHROMA_PORT", 8000))
    CHROMA_COLLECTION_PREFIX: str = os.getenv("CHROMA_COLLECTION_PREFIX", "")

    # AI Models & Retrieval Settings
    HF_HOME: str = os.getenv("HF_HOME", "/models/huggingface")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", ""))
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "")
    ENABLE_RERANKER: bool = (bool)(os.getenv("ENABLE_RERANKER"))
    TOP_K: int = (int)(os.getenv("TOP_K", 0))
    MAX_CONTEXT_CHARS: int = (int)(os.getenv("MAX_CONTEXT_CHARS", "0"))
    MAX_ANSWER_TOKENS: int = (int)(os.getenv("MAX_ANSWER_TOKENS", "0"))
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")

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
