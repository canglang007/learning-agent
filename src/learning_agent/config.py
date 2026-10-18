from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LA_", env_file=".env", extra="ignore")
    base_url: str = "https://surplustoken.com/v1"
    model: str = "gpt-6.1-sol"
    api_key: SecretStr = SecretStr("")
    protocol: Literal["chat_completions", "responses"] = "chat_completions"
    data_root: Path = Path("data")
    timeout_seconds: float = Field(default=90, gt=0)
    max_steps: int = Field(default=8, ge=1, le=20)
