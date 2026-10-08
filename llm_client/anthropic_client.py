"""Cliente para Anthropic usando el SDK oficial async."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from llm_client.base import BaseLLMClient, classify_sdk_error
from llm_client.schemas import ChatMessage, GenerationConfig, LLMError, ModelResponse, Usage

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"

    def __init__(self, api_key: str, default_model: str) -> None:
        super().__init__(api_key, default_model)
        self._sdk = AsyncAnthropic(api_key=api_key, max_retries=0)

    def _build_params(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> dict[str, Any]:
        # Anthropic recibe el system prompt como parámetro aparte, no en la lista.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": config.max_tokens,  # obligatorio en Anthropic
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "timeout": config.timeout_s,
        }
        if system:
            params["system"] = system
        if config.stop:
            params["stop_sequences"] = config.stop
        # El SDK anthropic >= 1.x ya no acepta temperature/top_p en messages.create:
        # se ignoran para mantener la misma interfaz que el resto de los proveedores.
        logger.debug("anthropic: temperature/top_p no soportados por el SDK, se ignoran")
        return params

    async def _generate_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> ModelResponse:
        response = await self._sdk.messages.create(**self._build_params(messages, model, config))
        content = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            provider=self.provider_name,
            model=response.model,
            content=content,
            finish_reason=response.stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def _stream_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        async with self._sdk.messages.stream(**self._build_params(messages, model, config)) as s:
            async for text in s.text_stream:
                yield text

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_sdk_error(exc, anthropic, self.provider_name)

    async def aclose(self) -> None:
        await self._sdk.close()
