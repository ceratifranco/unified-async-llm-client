"""AsyncLLMManager: elige el proveedor según configuración y delega en él."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Self

from dotenv import load_dotenv

from llm_client.anthropic_client import AnthropicClient
from llm_client.base import BaseLLMClient
from llm_client.groq_client import GroqClient
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, GenerationConfig, ModelResponse, Settings

PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "groq": GroqClient,
}

# Campo de Settings → variable de entorno
ENV_VARS = {
    "provider": "LLM_PROVIDER",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "anthropic_model": "ANTHROPIC_MODEL",
    "groq_model": "GROQ_MODEL",
    "timeout_s": "LLM_TIMEOUT_S",
    "max_retries": "LLM_MAX_RETRIES",
}


class AsyncLLMManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key, model = settings.credentials()
        self._client = PROVIDERS[settings.provider](api_key=api_key, default_model=model)

    @classmethod
    def from_env(cls, provider: str | None = None) -> Self:
        """Carga `.env` y arma la configuración. `provider` sobrescribe LLM_PROVIDER."""
        load_dotenv()
        data = {field: os.environ[var] for field, var in ENV_VARS.items() if os.environ.get(var)}
        if provider:
            data["provider"] = provider
        return cls(Settings.model_validate(data))

    @property
    def provider(self) -> str:
        return self.settings.provider

    @property
    def client(self) -> BaseLLMClient:
        return self._client

    def _resolve(self, config: GenerationConfig | None) -> GenerationConfig:
        if config is not None:
            return config
        return GenerationConfig(
            timeout_s=self.settings.timeout_s, max_retries=self.settings.max_retries
        )

    async def generate(
        self, messages: Sequence[ChatMessage], config: GenerationConfig | None = None
    ) -> ModelResponse:
        return await self._client.generate(messages, self._resolve(config))

    def stream(
        self, messages: Sequence[ChatMessage], config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        return self._client.stream(messages, self._resolve(config))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
