"""Modelos Pydantic: mensajes, configuración, respuestas y settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from llm_client.errors import ErrorType

Role = Literal["system", "user", "assistant"]
ProviderName = Literal["openai", "anthropic", "groq"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class GenerationConfig(BaseModel):
    """Parámetros de generación comunes a todos los proveedores."""

    model: str | None = Field(None, description="None → modelo por defecto del proveedor")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, gt=0, le=32_000)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    stop: list[str] | None = Field(None, max_length=4)
    timeout_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(BaseModel):
    type: ErrorType
    message: str
    retryable: bool
    provider: str
    status_code: int | None = None


class ModelResponse(BaseModel):
    provider: ProviderName
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Settings(BaseModel):
    """Configuración cargada desde variables de entorno (ver `.env.example`)."""

    provider: ProviderName = "groq"

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"
    groq_model: str = "openai/gpt-oss-20b"

    timeout_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)

    @model_validator(mode="after")
    def _key_for_selected_provider(self) -> Settings:
        key = getattr(self, f"{self.provider}_api_key")
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"Falta la API key para el proveedor '{self.provider}' "
                f"(variable {self.provider.upper()}_API_KEY)"
            )
        return self

    def credentials(self) -> tuple[str, str]:
        """Devuelve (api_key, modelo por defecto) del proveedor seleccionado."""
        key: SecretStr = getattr(self, f"{self.provider}_api_key")
        return key.get_secret_value(), getattr(self, f"{self.provider}_model")
