"""Helpers para simular respuestas y excepciones de los SDKs sin llamar a la red."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat")


def status_error(sdk: ModuleType, name: str, status: int, *, code: str | None = None,
                 headers: dict[str, str] | None = None) -> Exception:
    """Construye p.ej. `openai.RateLimitError` con un httpx.Response falso."""
    body = {"message": f"{name} simulado", "code": code}
    response = httpx.Response(status, request=REQUEST, headers=headers, json={"error": body})
    return getattr(sdk, name)(f"{name} simulado", response=response, body=body)


def connection_error(sdk: ModuleType) -> Exception:
    return sdk.APIConnectionError(request=REQUEST)


class FakeStream:
    """Imita el AsyncStream de openai/groq: iterable async y context manager."""

    def __init__(self, chunks: list, fail_after: Exception | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._fail_after:
            raise self._fail_after


def chat_chunk(text: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def chat_completion(text: str, model: str = "test-model"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Los reintentos no esperan de verdad en los tests."""
    sleep = AsyncMock()
    monkeypatch.setattr("llm_client.base.asyncio.sleep", sleep)
    return sleep
