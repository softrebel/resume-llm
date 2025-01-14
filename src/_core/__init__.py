from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from .logging import *  # noqa: F403


class Configs(BaseSettings):
    GEMINI_API_KEY: str | None = None
    JOBINJA_USERNAME: str | None = None
    JOBINJA_PASSWORD: str | None = None
    DATABASE_URL: str | None = None
    TEMP_PATH: str | None = None
    WKHTMLTOPDF_PATH: str | None = None
    TEMPLATE_HEADER: str = ""

    model_config = SettingsConfigDict(env_file=".env")


project_configs = Configs()


if project_configs.TEMP_PATH and not os.path.exists(project_configs.TEMP_PATH):
    os.makedirs(project_configs.TEMP_PATH)
