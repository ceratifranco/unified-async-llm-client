"""Cliente para Groq (free tier) usando el SDK oficial async.

La API de Groq es compatible con OpenAI Chat Completions, así que toda la
lógica vive en `OpenAICompatibleClient`; acá sólo cambia el SDK.
"""

from __future__ import annotations

import groq
from groq import AsyncGroq

from llm_client.openai_compat import OpenAICompatibleClient


class GroqClient(OpenAICompatibleClient):
    provider_name = "groq"
    sdk_module = groq

    def _build_sdk(self, api_key: str) -> AsyncGroq:
        return AsyncGroq(api_key=api_key, max_retries=0)
