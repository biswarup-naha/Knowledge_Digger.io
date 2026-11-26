from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config.app_config import get_settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def _create_db_engine():
    settings = get_settings()
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL must be configured to connect to PostgreSQL")

    return create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = _create_db_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
