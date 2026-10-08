"""Cliente para OpenAI usando el SDK oficial async."""

from __future__ import annotations

import openai
from openai import AsyncOpenAI

from llm_client.openai_compat import OpenAICompatibleClient


class OpenAIClient(OpenAICompatibleClient):
    provider_name = "openai"
    sdk_module = openai

    def _build_sdk(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key, max_retries=0)
