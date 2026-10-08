# Unified Async LLM Client

Cliente asíncrono en Python 3.12 que expone **OpenAI**, **Anthropic** y **Groq** bajo una misma interfaz, con streaming, validación con Pydantic, reintentos y errores controlados.

> **Se puede probar gratis con Groq** (no pide tarjeta). OpenAI y Anthropic son opcionales.

## Características

- **Intercambiable:** `AsyncLLMManager` elige el proveedor según `LLM_PROVIDER`; el código que lo usa no cambia.
- **100% async:** usa `AsyncOpenAI`, `AsyncAnthropic` y `AsyncGroq`; ninguna llamada bloquea el event loop.
- **Streaming:** `stream()` es un generador asíncrono que hace `yield` de cada fragmento de texto a medida que llega.
- **Validación:** `ChatMessage`, `GenerationConfig` (temperatura 0–2, `max_tokens` > 0, etc.), `ModelResponse` y `Settings` son modelos Pydantic.
- **Robusto:**
  - Reintenta con backoff exponencial y jitter los errores transitorios (rate limit 429, timeout, red, 5xx/529), respetando el header `Retry-After`.
  - Los errores definitivos (key inválida, request mal formado, cuota agotada) no se reintentan y vuelven como error estructurado: `generate()` nunca lanza y `stream()` lanza una única `LLMClientError`.

## Requisitos

- Python 3.12
- Una API key de al menos un proveedor. La de **Groq** es gratis: https://console.groq.com/keys

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Con [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env
```

Completá `.env` con al menos una key:

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `LLM_PROVIDER` | No | `groq` | Proveedor por defecto: `groq`, `openai` o `anthropic` |
| `GROQ_API_KEY` | Si usás Groq | — | Key gratuita de https://console.groq.com/keys |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Ver modelos vigentes en https://console.groq.com/docs/models |
| `OPENAI_API_KEY` | Si usás OpenAI | — | https://platform.openai.com/api-keys |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | — | https://console.anthropic.com/settings/keys |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | |
| `LLM_TIMEOUT_S` | No | `30` | Timeout por request (segundos) |
| `LLM_MAX_RETRIES` | No | `3` | Reintentos ante errores transitorios |

Si el proveedor seleccionado no tiene key, el programa falla al arrancar con un mensaje claro, no a mitad de una llamada.

## Ejecutar el script de prueba

```bash
python main.py
```

Por cada proveedor con key configurada, pregunta **"¿Qué es la entropía?"**:

1. En modo normal (`generate()`), mostrando modelo, latencia y tokens.
2. En modo streaming (`stream()`), imprimiendo el texto a medida que llega. Al terminar muestra cuántos fragmentos llegaron y en qué momento llegó el primero y el último.
3. Al final lanza 3 preguntas en paralelo con `asyncio.gather`: el tiempo total se acerca al de la más lenta y no a la suma, lo que demuestra que el event loop no se bloquea.

Otras opciones:

```bash
python main.py --provider groq   # un solo proveedor
python main.py --show-chunks     # separa con | cada fragmento recibido en el stream
python main.py --bad-key         # usa una key inválida: muestra el error controlado sin crash
python main.py -v                # muestra los logs de reintentos
```

## Uso como librería

```python
import asyncio
from llm_client import AsyncLLMManager, ChatMessage, GenerationConfig, LLMClientError

async def main():
    async with AsyncLLMManager.from_env() as llm:          # o from_env("anthropic")
        messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
        config = GenerationConfig(temperature=0.3, max_tokens=200)

        resp = await llm.generate(messages, config)
        print(resp.content if resp.ok else resp.error)

        try:
            async for chunk in llm.stream(messages, config):
                print(chunk, end="", flush=True)
        except LLMClientError as e:
            print(e.error.type, e.error.message)

asyncio.run(main())
```

## Estructura

```
llm_client/
├── schemas.py           # Pydantic: ChatMessage, GenerationConfig, ModelResponse, LLMError, Settings
├── errors.py            # ErrorType, LLMClientError
├── base.py              # BaseLLMClient (ABC): generate/stream, retry, clasificación de errores
├── openai_compat.py     # Lógica compartida para APIs estilo Chat Completions
├── openai_client.py     # OpenAIClient    (AsyncOpenAI)
├── groq_client.py       # GroqClient      (AsyncGroq)
├── anthropic_client.py  # AnthropicClient (AsyncAnthropic)
└── manager.py           # AsyncLLMManager: elige el proveedor según la configuración
tests/                   # Tests con los SDKs mockeados (no consumen tokens)
main.py                  # Script de validación
docs/DESIGN.md           # Decisiones de diseño y trade-offs
```

## Manejo de errores

| Error | Tipo | ¿Reintenta? |
|---|---|---|
| Key inválida / sin permisos (401/403) | `auth` | No |
| Request inválido / modelo inexistente (400/404/422) | `bad_request` | No |
| Rate limit (429) | `rate_limit` | Sí, respetando `Retry-After` |
| Cuota agotada (429 `insufficient_quota`) | `quota` | No |
| Timeout | `timeout` | Sí |
| Error de conexión | `network` | Sí |
| Error del servidor / sobrecarga (5xx, 529) | `server` | Sí |

En streaming solo se reintenta si el error ocurre **antes** del primer fragmento; reintentar después duplicaría texto.

## Tests

```bash
pytest
```

## Notas

- **Anthropic:**
  - El system prompt se envía como parámetro `system`, no dentro de la lista de mensajes.
  - El SDK `anthropic` 1.x ya no acepta `temperature` ni `top_p` en `messages.create`, así que para este proveedor esos campos se ignoran.
- **Groq:**
  - Su API es compatible con la de OpenAI, por eso `GroqClient` reutiliza toda la lógica de `OpenAICompatibleClient`.
  - El plan gratuito tiene límites bajos: un 429 es esperable y el cliente lo reintenta solo.
