"""Lógica compartida para APIs estilo OpenAI Chat Completions (OpenAI y Groq)."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import ModuleType
from typing import Any, ClassVar

from llm_client.base import BaseLLMClient, classify_sdk_error
from llm_client.schemas import ChatMessage, GenerationConfig, LLMError, ModelResponse, Usage


class OpenAICompatibleClient(BaseLLMClient):
    sdk_module: ClassVar[ModuleType]

    def __init__(self, api_key: str, default_model: str) -> None:
        super().__init__(api_key, default_model)
        self._sdk = self._build_sdk(api_key)

    @abstractmethod
    def _build_sdk(self, api_key: str) -> Any:
        """Devuelve el cliente async del SDK (con `max_retries=0`: el retry es nuestro)."""

    def _build_params(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": config.temperature,
            "max_completion_tokens": config.max_tokens,
            "timeout": config.timeout_s,
        }
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.stop:
            params["stop"] = config.stop
        return params

    async def _generate_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> ModelResponse:
        response = await self._sdk.chat.completions.create(
            **self._build_params(messages, model, config)
        )
        choice = response.choices[0]
        usage = (
            Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            if response.usage
            else None
        )
        return ModelResponse(
            provider=self.provider_name,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    async def _stream_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        stream = await self._sdk.chat.completions.create(
            **self._build_params(messages, model, config), stream=True
        )
        async with stream:  # cierra la conexión aunque el consumidor corte antes
            async for chunk in stream:
                # El último chunk puede venir sin choices (sólo usage) o con delta vacío.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_sdk_error(exc, self.sdk_module, self.provider_name)

    async def aclose(self) -> None:
        await self._sdk.close()
