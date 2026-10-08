import pytest
from pydantic import ValidationError

from llm_client import AnthropicClient, AsyncLLMManager, GroqClient, OpenAIClient, Settings


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
                "OPENAI_MODEL", "ANTHROPIC_MODEL", "GROQ_MODEL"):
        monkeypatch.delenv(var, raising=False)
    # Evita que un .env real del desarrollador interfiera.
    monkeypatch.setattr("llm_client.manager.load_dotenv", lambda: None)
    return monkeypatch


@pytest.mark.parametrize(
    ("provider", "client_cls"),
    [("openai", OpenAIClient), ("anthropic", AnthropicClient), ("groq", GroqClient)],
)
async def test_manager_selects_provider(provider, client_cls):
    settings = Settings.model_validate({"provider": provider, f"{provider}_api_key": "k"})
    async with AsyncLLMManager(settings) as llm:
        assert isinstance(llm.client, client_cls)
        assert llm.provider == provider


async def test_from_env_reads_provider_and_model(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")
    clean_env.setenv("GROQ_API_KEY", "gsk_test")
    clean_env.setenv("GROQ_MODEL", "otro-modelo")

    async with AsyncLLMManager.from_env() as llm:
        assert isinstance(llm.client, GroqClient)
        assert llm.client.default_model == "otro-modelo"


async def test_from_env_provider_argument_overrides_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")
    clean_env.setenv("GROQ_API_KEY", "gsk_test")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")

    async with AsyncLLMManager.from_env("openai") as llm:
        assert isinstance(llm.client, OpenAIClient)


def test_from_env_fails_fast_without_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        AsyncLLMManager.from_env()
