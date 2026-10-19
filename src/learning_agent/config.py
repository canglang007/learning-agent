from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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
    enable_vision: bool = True
    max_images_per_turn: int = Field(default=3, ge=1, le=6)
    context_budget_tokens: int = Field(default=65536, ge=1024)
    context_output_reserve_tokens: int = Field(default=4096, ge=1)
    context_headroom_tokens: int = Field(default=4096, ge=0)
    max_context_images: int | None = Field(default=None, ge=1, le=6)
    context_image_allowance: int = Field(default=4096, ge=1)
    context_max_request_bytes: int = Field(default=64 * 1024**2, ge=1024)
    output_token_field: Literal[
        "none", "max_tokens", "max_completion_tokens", "max_output_tokens"
    ] = "none"
    asr_provider: Literal["disabled", "local", "compatible"] = "disabled"
    asr_model: str = "base"
    asr_base_url: str = "https://api.openai.com/v1"
    asr_api_key: SecretStr = SecretStr("")
    media_max_bytes: int = Field(default=2 * 1024**3, ge=1)
    media_max_seconds: float = Field(default=14400, ge=0.1, le=86400)
    asr_local_part_seconds: int = Field(default=300, ge=10, le=600)
    asr_api_part_seconds: int = Field(default=60, ge=10, le=120)
    asr_context_seconds: int = Field(default=5, ge=0, le=10)
    retrieval_mode: Literal["lexical_baseline", "bm25", "hybrid"] = "hybrid"
    embedding_provider: Literal["disabled", "ollama", "compatible"] = "disabled"
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3:latest"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_timeout_seconds: float = Field(default=30, gt=0, le=180)

    @model_validator(mode="after")
    def context_limits(self):
        if self.context_budget_tokens <= (
            self.context_output_reserve_tokens + self.context_headroom_tokens
        ):
            raise ValueError("上下文总预算必须大于输出预留与误差余量之和")
        if self.max_context_images is None:
            self.max_context_images = self.max_images_per_turn
        if self.max_context_images < self.max_images_per_turn:
            raise ValueError("上下文图片上限不能小于本轮图片上限")
        if self.output_token_field != "none" and (
            (self.protocol == "responses") != (self.output_token_field == "max_output_tokens")
        ):
            raise ValueError("输出限制字段与所选协议不匹配")
        return self
