from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import groq
import openai
import pytest

from conftest import FakeStream, chat_chunk, chat_completion, connection_error, status_error
from llm_client import (
    AnthropicClient,
    ChatMessage,
    ErrorType,
    GenerationConfig,
    GroqClient,
    LLMClientError,
    OpenAIClient,
)

MESSAGES = [
    ChatMessage(role="system", content="Sé breve."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]

# OpenAI y Groq comparten implementación: los mismos tests corren para ambos.
OPENAI_COMPATIBLE = [(OpenAIClient, openai), (GroqClient, groq)]


def make_compat_client(client_cls, create: AsyncMock):
    client = client_cls(api_key="test", default_model="test-model")
    client._sdk = MagicMock()
    client._sdk.chat.completions.create = create
    return client


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_generate_ok(client_cls, sdk):
    create = AsyncMock(return_value=chat_completion("Es una medida del desorden."))
    client = make_compat_client(client_cls, create)

    resp = await client.generate(MESSAGES, GenerationConfig(temperature=0.2, max_tokens=50))

    assert resp.ok
    assert resp.content == "Es una medida del desorden."
    assert resp.provider == client_cls.provider_name
    assert resp.usage.input_tokens == 10 and resp.usage.output_tokens == 5
    assert resp.latency_ms is not None
    kwargs = create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_completion_tokens"] == 50
    assert kwargs["messages"][0] == {"role": "system", "content": "Sé breve."}


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_generate_retries_rate_limit_then_succeeds(client_cls, sdk, no_sleep):
    create = AsyncMock(
        side_effect=[
            status_error(sdk, "RateLimitError", 429),
            connection_error(sdk),
            chat_completion("ok"),
        ]
    )
    client = make_compat_client(client_cls, create)

    resp = await client.generate(MESSAGES)

    assert resp.ok and resp.content == "ok"
    assert create.await_count == 3
    assert no_sleep.await_count == 2


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_generate_respects_retry_after(client_cls, sdk, no_sleep):
    create = AsyncMock(
        side_effect=[
            status_error(sdk, "RateLimitError", 429, headers={"retry-after": "7"}),
            chat_completion("ok"),
        ]
    )
    client = make_compat_client(client_cls, create)

    await client.generate(MESSAGES)

    no_sleep.assert_awaited_once_with(7.0)


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_generate_auth_error_is_not_retried(client_cls, sdk):
    create = AsyncMock(side_effect=status_error(sdk, "AuthenticationError", 401))
    client = make_compat_client(client_cls, create)

    resp = await client.generate(MESSAGES)

    assert not resp.ok
    assert resp.error.type == ErrorType.AUTH
    assert resp.error.status_code == 401
    assert create.await_count == 1


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_generate_gives_up_after_max_retries(client_cls, sdk):
    create = AsyncMock(side_effect=status_error(sdk, "InternalServerError", 500))
    client = make_compat_client(client_cls, create)

    resp = await client.generate(MESSAGES, GenerationConfig(max_retries=2))

    assert resp.error.type == ErrorType.SERVER
    assert create.await_count == 3


async def test_insufficient_quota_is_not_retried():
    create = AsyncMock(
        side_effect=status_error(openai, "RateLimitError", 429, code="insufficient_quota")
    )
    client = make_compat_client(OpenAIClient, create)

    resp = await client.generate(MESSAGES)

    assert resp.error.type == ErrorType.QUOTA
    assert create.await_count == 1


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_stream_yields_text_and_skips_empty_deltas(client_cls, sdk):
    stream = FakeStream([chat_chunk("Hola"), chat_chunk(None), chat_chunk(" mundo")])
    client = make_compat_client(client_cls, AsyncMock(return_value=stream))

    chunks = [c async for c in client.stream(MESSAGES)]

    assert chunks == ["Hola", " mundo"]
    assert stream.closed


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_stream_retries_before_first_chunk(client_cls, sdk):
    create = AsyncMock(
        side_effect=[connection_error(sdk), FakeStream([chat_chunk("ok")])]
    )
    client = make_compat_client(client_cls, create)

    assert [c async for c in client.stream(MESSAGES)] == ["ok"]
    assert create.await_count == 2


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_stream_does_not_retry_mid_stream(client_cls, sdk):
    stream = FakeStream([chat_chunk("Hola")], fail_after=connection_error(sdk))
    create = AsyncMock(return_value=stream)
    client = make_compat_client(client_cls, create)

    received = []
    with pytest.raises(LLMClientError) as exc_info:
        async for chunk in client.stream(MESSAGES):
            received.append(chunk)

    assert received == ["Hola"]
    assert exc_info.value.error.type == ErrorType.NETWORK
    assert create.await_count == 1


@pytest.mark.parametrize(("client_cls", "sdk"), OPENAI_COMPATIBLE)
async def test_stream_auth_error_raises_controlled_error(client_cls, sdk):
    create = AsyncMock(side_effect=status_error(sdk, "AuthenticationError", 401))
    client = make_compat_client(client_cls, create)

    with pytest.raises(LLMClientError) as exc_info:
        _ = [c async for c in client.stream(MESSAGES)]

    assert exc_info.value.error.type == ErrorType.AUTH


async def test_requires_non_system_message():
    client = make_compat_client(GroqClient, AsyncMock())
    with pytest.raises(ValueError):
        await client.generate([ChatMessage(role="system", content="sólo system")])


# ---- Anthropic -------------------------------------------------------------


def make_anthropic_client():
    client = AnthropicClient(api_key="test", default_model="claude-test")
    client._sdk = MagicMock()
    return client


async def test_anthropic_generate_extracts_system_prompt():
    client = make_anthropic_client()
    client._sdk.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            model="claude-test",
            content=[SimpleNamespace(type="text", text="Desorden.")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        )
    )

    resp = await client.generate(MESSAGES, GenerationConfig(max_tokens=64, stop=["FIN"]))

    assert resp.ok and resp.content == "Desorden."
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == "Sé breve."
    assert kwargs["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert kwargs["max_tokens"] == 64
    assert kwargs["stop_sequences"] == ["FIN"]
    assert "temperature" not in kwargs


async def test_anthropic_stream():
    async def text_stream():
        for t in ["La ", "entropía"]:
            yield t

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=SimpleNamespace(text_stream=text_stream()))
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    client = make_anthropic_client()
    client._sdk.messages.stream = MagicMock(return_value=stream_ctx)

    assert [c async for c in client.stream(MESSAGES)] == ["La ", "entropía"]


async def test_anthropic_overloaded_is_retryable():
    client = make_anthropic_client()
    client._sdk.messages.create = AsyncMock(
        side_effect=status_error(anthropic, "OverloadedError", 529)
    )

    resp = await client.generate(MESSAGES, GenerationConfig(max_retries=1))

    assert resp.error.type == ErrorType.SERVER and resp.error.retryable
    assert client._sdk.messages.create.await_count == 2
