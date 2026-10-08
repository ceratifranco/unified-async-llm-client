import pytest
from pydantic import ValidationError

from llm_client import ChatMessage, GenerationConfig, Settings


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_out_of_range(temperature):
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=temperature)


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_temperature_in_range(temperature):
    assert GenerationConfig(temperature=temperature).temperature == temperature


def test_max_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        GenerationConfig(max_tokens=0)


def test_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")


def test_empty_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_settings_requires_key_for_selected_provider():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(provider="groq", openai_api_key="sk-x")


def test_settings_hides_secret_in_repr():
    settings = Settings(provider="groq", groq_api_key="gsk_supersecreta")
    assert "supersecreta" not in repr(settings)
    assert settings.credentials() == ("gsk_supersecreta", "openai/gpt-oss-20b")
