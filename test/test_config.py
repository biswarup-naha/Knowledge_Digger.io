import pytest
from pydantic import ValidationError

from src.config.app_config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.APP_NAME == "Knowledge_Digger.io"
    assert settings.ENVIRONMENT == "development"


def test_log_level_is_normalized():
    settings = Settings(log_level="debug")
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")
