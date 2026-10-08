"""Clase base abstracta: interfaz común, retry y manejo de errores."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import ModuleType
from typing import ClassVar, Self

from llm_client.errors import ErrorType, LLMClientError
from llm_client.schemas import (
    ChatMessage,
    GenerationConfig,
    LLMError,
    ModelResponse,
    ProviderName,
)

logger = logging.getLogger(__name__)

BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 20.0
BACKOFF_JITTER_S = 0.5


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores.

    `generate()` y `stream()` se implementan una sola vez acá (template method):
    validan, reintentan errores transitorios y traducen excepciones del SDK.
    Cada proveedor sólo implementa los métodos `_..._impl` y `_classify_error`.
    """

    provider_name: ClassVar[ProviderName]

    def __init__(self, api_key: str, default_model: str) -> None:
        self.default_model = default_model

    # ---- API pública -------------------------------------------------------

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """Genera una respuesta completa. Nunca lanza por errores de la API:
        los devuelve en `ModelResponse.error`."""
        config = config or GenerationConfig()
        model = config.model or self.default_model
        self._validate_messages(messages)
        start = time.perf_counter()

        attempt = 0
        while True:
            try:
                response = await self._generate_impl(messages, model, config)
                return response.model_copy(update={"latency_ms": _elapsed_ms(start)})
            except Exception as exc:
                error = self._classify_error(exc)
                if not error.retryable or attempt >= config.max_retries:
                    logger.warning("%s: generate falló (%s)", self.provider_name, error.message)
                    return ModelResponse(
                        provider=self.provider_name,
                        model=model,
                        error=error,
                        latency_ms=_elapsed_ms(start),
                    )
                await self._wait_before_retry(attempt, exc, error)
                attempt += 1

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto.

        Reintenta sólo si el error ocurre antes del primer fragmento (si no,
        duplicaría texto). Los errores se exponen como `LLMClientError`.
        """
        config = config or GenerationConfig()
        model = config.model or self.default_model
        self._validate_messages(messages)

        attempt = 0
        while True:
            emitted = False
            try:
                async for chunk in self._stream_impl(messages, model, config):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                error = self._classify_error(exc)
                if emitted or not error.retryable or attempt >= config.max_retries:
                    raise LLMClientError(error) from exc
                await self._wait_before_retry(attempt, exc, error)
                attempt += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---- A implementar por cada proveedor --------------------------------

    @abstractmethod
    async def _generate_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> ModelResponse: ...

    @abstractmethod
    def _stream_impl(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    # ---- Helpers ------------------------------------------------------------

    @staticmethod
    def _validate_messages(messages: Sequence[ChatMessage]) -> None:
        if not any(m.role != "system" for m in messages):
            raise ValueError("Se necesita al menos un mensaje 'user' o 'assistant'")

    async def _wait_before_retry(self, attempt: int, exc: Exception, error: LLMError) -> None:
        delay = _retry_after_s(exc)
        if delay is None:
            delay = BACKOFF_BASE_S * 2**attempt + random.uniform(0, BACKOFF_JITTER_S)
        delay = min(delay, BACKOFF_CAP_S)
        logger.info(
            "%s: %s, reintento %d en %.1fs", self.provider_name, error.type, attempt + 1, delay
        )
        await asyncio.sleep(delay)


def classify_sdk_error(exc: Exception, sdk: ModuleType, provider: str) -> LLMError:
    """Traduce excepciones de SDKs generados con Stainless (openai, anthropic, groq),
    que comparten nombres de clases, a un `LLMError`."""
    status = getattr(exc, "status_code", None)

    def err(type_: ErrorType, retryable: bool) -> LLMError:
        return LLMError(
            type=type_,
            message=_error_message(exc),
            retryable=retryable,
            provider=provider,
            status_code=status,
        )

    # El orden importa: APITimeoutError es subclase de APIConnectionError.
    if isinstance(exc, sdk.APITimeoutError):
        return err(ErrorType.TIMEOUT, True)
    if isinstance(exc, sdk.APIConnectionError):
        return err(ErrorType.NETWORK, True)
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return err(ErrorType.AUTH, False)
    if isinstance(exc, sdk.RateLimitError):
        # Cuota agotada (sin crédito) también llega como 429 pero no se arregla reintentando.
        if getattr(exc, "code", None) == "insufficient_quota":
            return err(ErrorType.QUOTA, False)
        return err(ErrorType.RATE_LIMIT, True)
    if isinstance(exc, (sdk.BadRequestError, sdk.NotFoundError, sdk.UnprocessableEntityError)):
        return err(ErrorType.BAD_REQUEST, False)
    if isinstance(exc, sdk.APIStatusError) and (status >= 500 or status in (408, 409)):
        return err(ErrorType.SERVER, True)
    return err(ErrorType.UNKNOWN, False)


def _error_message(exc: Exception) -> str:
    """Mensaje legible: prioriza el `message` del body JSON de la API."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return getattr(exc, "message", None) or str(exc) or type(exc).__name__


def _retry_after_s(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
