# Diseño — Unified Async LLM Client (Pre-entrega 1)

## 1. Requisitos

### Funcionales
| # | Requisito | Dónde se resuelve |
|---|-----------|-------------------|
| F1 | Interfaz común para OpenAI, Anthropic y Groq | `BaseLLMClient` (ABC) + `OpenAIClient` / `AnthropicClient` / `GroqClient` |
| F2 | Todas las llamadas no bloqueantes | `AsyncOpenAI` / `AsyncAnthropic` / `AsyncGroq` + `await` |
| F3 | Streaming como generador asíncrono de texto | `async def stream(...) -> AsyncIterator[str]` con `yield` |
| F4 | Validación de mensajes y config con Pydantic | `schemas.py` |
| F5 | Elegir proveedor por configuración | `AsyncLLMManager` + `LLM_PROVIDER` en `.env` |
| F6 | Errores de red / rate limit / auth → error controlado, no crash | Capa de traducción de excepciones + retry |
| F7 | Script de prueba normal + streaming | `main.py` |

### No funcionales
- Python 3.12, sin frameworks extra (nada de LangChain): sólo `openai`, `anthropic`, `groq`, `pydantic`, `python-dotenv`.
- **Debe poder correrse gratis**: Groq (free tier, sin tarjeta) es el proveedor por defecto para desarrollo y para el corrector.
- Código legible para corrección: pocas capas, nombres explícitos, type hints completos.
- Testeable sin gastar tokens: los SDKs se pueden mockear.

### Supuestos
- Uso desde scripts / notebooks, no un servicio con carga. No hay requisitos de escala, caching ni persistencia.
- Una conversación = una lista de `ChatMessage`; no se maneja historial ni tools/function-calling.

## 2. Diseño de alto nivel

```
                ┌──────────────┐
  main.py ────► │AsyncLLMManager│  lee Settings (.env) → elige proveedor
                └──────┬───────┘
                       │ delega (composición)
                       ▼
               ┌────────────────┐
               │ BaseLLMClient  │  ABC: generate(), stream()
               │  (abstracta)   │  + retry / mapeo de errores comunes
               └───────┬────────┘
          ┌────────────┴─────────────┐
          ▼                          ▼
 ┌──────────────────────┐    ┌────────────────┐
 │OpenAICompatibleClient│    │ AnthropicClient│   traducen schemas ⇄ formato SDK
 │   (abstracta)        │    └───────┬────────┘
 └─────────┬────────────┘            ▼
     ┌─────┴──────┐            AsyncAnthropic
     ▼            ▼
┌────────────┐ ┌──────────┐
│OpenAIClient│ │GroqClient│
└─────┬──────┘ └────┬─────┘
      ▼             ▼
 AsyncOpenAI     AsyncGroq                       (SDKs oficiales, async)

  schemas.py: ChatMessage, GenerationConfig, ModelResponse, LLMError, Settings
```

### Flujo `generate()`
```
main → manager.generate(messages, config)
     → client.generate()                       # BaseLLMClient (template method)
         ├─ valida inputs (Pydantic ya lo hizo al construirlos)
         ├─ loop de retry (backoff exponencial, sólo errores transitorios)
         │     └─ await self._generate_impl()  # específico del proveedor
         └─ captura excepciones del SDK → ModelResponse(error=LLMError(...))
     ← ModelResponse (siempre; nunca lanza por errores de API)
```

### Flujo `stream()`
```
async for chunk in manager.stream(messages, config):
    print(chunk, end="", flush=True)
```
- El cliente abre el stream del SDK y hace `yield` de cada delta de texto.
- Si falla **antes** del primer token → se reintenta (seguro, no se emitió nada).
- Si falla **a mitad** del stream → no se reintenta (duplicaría texto); se emite un `StreamError` final o se lanza `LLMClientError` controlado (ver §3.4).

## 3. Deep dive

### 3.1 Estructura del repo

```
entrega1/
├── llm_client/
│   ├── __init__.py          # exporta AsyncLLMManager, schemas
│   ├── schemas.py           # Pydantic: mensajes, config, respuesta, errores, settings
│   ├── base.py              # BaseLLMClient (ABC) + retry + mapeo de errores
│   ├── openai_client.py     # OpenAIClient
│   ├── openai_compat.py     # OpenAICompatibleClient (lógica compartida OpenAI/Groq)
│   ├── anthropic_client.py  # AnthropicClient
│   ├── groq_client.py       # GroqClient
│   ├── manager.py           # AsyncLLMManager (factory por config)
│   └── errors.py            # LLMClientError + ErrorType (enum)
├── tests/
│   ├── test_schemas.py
│   ├── test_clients.py      # SDK mockeado
│   └── test_manager.py
├── main.py
├── .env.example
├── .gitignore               # .env, .venv, __pycache__
├── requirements.txt
└── README.md
```

> El enunciado pide `schemas.py`: vive en `llm_client/schemas.py`. Si el corrector espera verlo en la raíz, se puede dejar todo plano; el paquete es más prolijo y no cuesta nada.

### 3.2 Modelo de datos (`schemas.py`)

```python
Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Role
    content: str = Field(min_length=1)

class GenerationConfig(BaseModel):
    model: str | None = None                 # None → default del proveedor
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, gt=0, le=32_000)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    stop: list[str] | None = None
    timeout_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

class LLMError(BaseModel):
    type: ErrorType          # auth | rate_limit | network | timeout | bad_request | server | unknown
    message: str
    retryable: bool
    provider: str

class ModelResponse(BaseModel):
    provider: Literal["openai", "anthropic", "groq"]
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool: return self.error is None

class Settings(BaseModel):          # se arma desde os.environ en manager.py
    provider: Literal["openai", "anthropic", "groq"] = "groq"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"
    groq_model: str = "openai/gpt-oss-20b"

    @model_validator(mode="after")
    def key_for_selected_provider(self): ...   # falla temprano si falta la key
```

Decisiones:
- **`SecretStr`** para las keys → no se filtran en `repr`/logs.
- **Temperatura en Anthropic**: el schema común permite `0–2` (lo pide el enunciado). El diseño original recortaba a 1.0 para Anthropic, pero el SDK `anthropic` 1.x instalado **ya no acepta `temperature`/`top_p`** en `messages.create`, así que el `AnthropicClient` los ignora. Alternativa descartada: rechazar en el schema → rompe la intercambiabilidad.
- `ModelResponse.error` en vez de lanzar: el `main` puede hacer `if not resp.ok:` sin `try/except`. Esto resuelve "Fuga de Excepciones".
- No uso `pydantic-settings` para no sumar dependencias fuera de la lista; `Settings.model_validate({...os.environ...})` alcanza.

### 3.3 Interfaz (`base.py`)

```python
class BaseLLMClient(ABC):
    provider_name: ClassVar[str]

    def __init__(self, api_key: str, default_model: str): ...

    # API pública — implementada UNA vez en la base (template method)
    async def generate(self, messages: list[ChatMessage],
                       config: GenerationConfig | None = None) -> ModelResponse: ...
    async def stream(self, messages: list[ChatMessage],
                     config: GenerationConfig | None = None) -> AsyncIterator[str]: ...
    async def aclose(self) -> None: ...
    async def __aenter__ / __aexit__          # cierra el httpx client del SDK

    # Lo que cada proveedor implementa
    @abstractmethod
    async def _generate_impl(self, messages, config) -> ModelResponse: ...
    @abstractmethod
    def _stream_impl(self, messages, config) -> AsyncIterator[str]: ...
    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError | None: ...
```

Por qué template method: retry, medición de latencia y captura de errores quedan en un solo lugar; cada proveedor sólo traduce formatos y excepciones.

### 3.4 Proveedores — diferencias que hay que resolver

| Aspecto | OpenAI (`AsyncOpenAI`) | Anthropic (`AsyncAnthropic`) |
|---|---|---|
| Llamada | `await client.chat.completions.create(...)` | `await client.messages.create(...)` |
| System prompt | mensaje con `role="system"` en la lista | parámetro aparte `system=`; se **extrae** de la lista |
| `max_tokens` | opcional (`max_completion_tokens` en modelos nuevos) | **obligatorio** |
| Temperatura | 0–2 | no soportada por el SDK 1.x → se ignora |
| Texto | `resp.choices[0].message.content` | concatenar bloques `type == "text"` de `resp.content` |
| Streaming | `create(..., stream=True)` → `async for chunk`: `chunk.choices[0].delta.content` (puede ser `None` / `choices` vacío) | `async with client.messages.stream(...) as s: async for text in s.text_stream` |
| Excepciones | `openai.AuthenticationError`, `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `BadRequestError`, `InternalServerError` | mismas clases bajo `anthropic.*` (+ `OverloadedError` / 529) |

Notas:
- Ambos SDKs ya traen retry interno (`max_retries=2` por defecto). **Lo desactivo** (`max_retries=0` en el constructor) y hago el retry propio en la base → un solo lugar de verdad, y el retry es visible para el corrector.
- `timeout` se pasa al constructor del SDK (o por request con `with_options`).

### 3.4.1 Groq (proveedor gratuito)

Groq expone una API **compatible con OpenAI Chat Completions**, y su SDK oficial (`groq`, cliente `AsyncGroq`) es casi idéntico al de OpenAI (ambos generados con Stainless). Por eso:

```python
class OpenAICompatibleClient(BaseLLMClient, ABC):
    """Lógica compartida para APIs estilo Chat Completions."""
    @abstractmethod
    def _build_sdk(self, api_key: str, timeout: float): ...   # AsyncOpenAI | AsyncGroq
    @property
    @abstractmethod
    def _errors(self) -> ModuleType: ...                       # módulo openai | groq (para isinstance)

    async def _generate_impl(...):  # chat.completions.create(...)
    async def _stream_impl(...):    # create(stream=True) + async for chunk
    def _classify_error(...):       # usa self._errors.RateLimitError, etc.

class OpenAIClient(OpenAICompatibleClient):
    provider_name = "openai"
    def _build_sdk(...): return AsyncOpenAI(api_key=..., max_retries=0, timeout=...)

class GroqClient(OpenAICompatibleClient):
    provider_name = "groq"
    def _build_sdk(...): return AsyncGroq(api_key=..., max_retries=0, timeout=...)
    # override sólo de lo que difiere (usage en stream, params no soportados)
```

Particularidades de Groq a tener en cuenta:

| Aspecto | Groq |
|---|---|
| Key | Gratis en https://console.groq.com/keys (prefijo `gsk_`) |
| Modelos | Open-weight (Llama, Qwen, GPT-OSS, etc.). Default configurable por `GROQ_MODEL`; **verificar el catálogo vigente** en https://console.groq.com/docs/models porque Groq depreca modelos seguido |
| Temperatura | 0–2 (igual que OpenAI, sin clamp) |
| `max_tokens` | Aceptado (`max_tokens` / `max_completion_tokens`) |
| Parámetros no soportados | `n > 1`, `logprobs`, `logit_bias` → no los mandamos |
| Rate limits del free tier | Bajos (RPM / TPM / tokens por día por modelo). Los **429 son frecuentes** y traen `retry-after` → es el caso real que ejercita el retry |
| Excepciones | `groq.AuthenticationError`, `groq.RateLimitError`, `groq.APIConnectionError`, `groq.APITimeoutError`, etc. (mismos nombres que OpenAI) |

Por qué el **SDK oficial `groq`** y no `AsyncOpenAI` con `base_url` de Groq: el enunciado pide SDKs oficiales, y las excepciones quedan tipadas por proveedor. La alternativa con `base_url` funciona y ahorra una dependencia; se menciona en el README como opción.

### 3.5 Manejo de errores y retry

Clasificación (`ErrorType` + `retryable`):

| Error SDK | ErrorType | ¿Reintenta? |
|---|---|---|
| `AuthenticationError` (401), `PermissionDeniedError` (403) | `auth` | No |
| `BadRequestError` (400), `NotFoundError` (modelo inválido) | `bad_request` | No |
| `RateLimitError` (429) | `rate_limit` | Sí — respetando `Retry-After` si viene |
| `APITimeoutError` | `timeout` | Sí |
| `APIConnectionError` | `network` | Sí |
| `InternalServerError` / 5xx / 529 overloaded | `server` | Sí |
| Otro | `unknown` | No |

> Ojo: un 429 por **cuota agotada** (`insufficient_quota` en OpenAI) no se arregla reintentando. Se detecta por el `code` del body y se marca `retryable=False`.

Backoff: `delay = min(base * 2**attempt, cap) + random.uniform(0, jitter)` con `base=1s`, `cap=20s`; `await asyncio.sleep(delay)` (nunca `time.sleep`).

Contrato de errores:
- `generate()` **nunca lanza** por errores de API → devuelve `ModelResponse(error=...)`.
- `stream()` es un generador: no puede "devolver" un objeto al final. Opciones:
  1. **(Elegida)** Lanzar una única excepción propia `LLMClientError(error: LLMError)`. El caller hace `try/except LLMClientError`. Es lo idiomático en Python y no mezcla texto con errores.
  2. Hacer `yield` de un string tipo `"[ERROR] ..."` → simple pero el consumidor no distingue error de contenido. Descartada.
  3. Yield de objetos `StreamChunk(text|error)` → más robusto pero cambia el contrato "generador de tokens" que pide el enunciado. Se puede agregar luego como `stream_events()`.
- `asyncio.CancelledError` **no se captura** (hay que dejarla propagar para no romper la cancelación).

### 3.6 `AsyncLLMManager` (`manager.py`)

```python
class AsyncLLMManager:
    _REGISTRY: dict[str, type[BaseLLMClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "groq": GroqClient,
    }

    def __init__(self, settings: Settings): ...
    @classmethod
    def from_env(cls, provider: str | None = None) -> "AsyncLLMManager":
        # load_dotenv() + Settings.model_validate(os.environ ...)
    @property
    def client(self) -> BaseLLMClient: ...
    async def generate(...) -> ModelResponse     # delega
    def stream(...) -> AsyncIterator[str]          # delega
    async def __aenter__/__aexit__                 # cierra el cliente
```

- Registry dict en vez de `if/elif`: agregar un proveedor = una línea + una clase (así entró Groq).
- `from_env(provider=...)` permite sobrescribir `LLM_PROVIDER` → `main.py` puede probar los dos proveedores en la misma corrida.

### 3.7 `main.py`

```python
async def run(provider: str) -> None:
    async with AsyncLLMManager.from_env(provider) as llm:
        msgs = [ChatMessage(role="system", content="Respondé en 2-3 oraciones."),
                ChatMessage(role="user", content="¿Qué es la entropía?")]
        cfg = GenerationConfig(temperature=0.3, max_tokens=200)

        resp = await llm.generate(msgs, cfg)
        print(resp.content if resp.ok else f"Error controlado: {resp.error}")

        try:
            async for tok in llm.stream(msgs, cfg):
                print(tok, end="", flush=True)
        except LLMClientError as e:
            print(f"\nError controlado en stream: {e.error}")

async def main():
    load_dotenv()
    providers = [p for p in ("groq", "openai", "anthropic") if has_key(p)]
    for p in providers:          # secuencial para que la salida sea legible
        await run(p)
    # Bonus: demostrar no-bloqueo → asyncio.gather de los generate() disponibles y medir tiempo total

if __name__ == "__main__":
    asyncio.run(main())
```

Bonus recomendado: un tercer bloque que corra `asyncio.gather(...)` sobre los proveedores con key (con sólo Groq: 3 preguntas distintas en paralelo) e imprima que el tiempo total ≈ max(latencias) y no la suma. Es la prueba concreta de que no se bloquea el event loop.

Flag opcional `--bad-key` que fuerza una key inválida para demostrar el error controlado en vivo.

### 3.8 Configuración (`.env.example`)

```dotenv
# Proveedor por defecto: groq | openai | anthropic
LLM_PROVIDER=groq

# Groq — gratis: https://console.groq.com/keys
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5

# Opcionales
LLM_TIMEOUT_S=30
LLM_MAX_RETRIES=3
```

## 4. Testing (sin gastar tokens)

- `test_schemas.py`: temperatura `-0.1`/`2.1` → `ValidationError`; `role` inválido; `content` vacío; `Settings` sin key del proveedor elegido → falla.
- `test_clients.py`: mockear `client._sdk.chat.completions.create` / `messages.create` con `AsyncMock`.
  - happy path → `ModelResponse` correcto (texto, usage) — parametrizado para OpenAI y Groq (comparten base).
  - extracción del system prompt en Anthropic.
  - Anthropic no envía `temperature`.
  - `RateLimitError` ×2 y luego OK → se reintentó 2 veces (parchear `asyncio.sleep`).
  - `AuthenticationError` → `resp.error.type == "auth"`, sin reintentos.
  - stream: fake async iterator → los chunks llegan en orden, se ignoran deltas `None`.
- `pytest-asyncio` con `asyncio_mode = "auto"` (en `pytest.ini` o `pyproject.toml`).
- Construir las excepciones del SDK requiere un `httpx.Response`; helper en `conftest.py`.

## 5. Trade-offs explícitos

| Decisión | Alternativa | Por qué esta |
|---|---|---|
| Retry propio, SDK con `max_retries=0` | Usar el retry del SDK | Visible, unificado, controlable por config; el enunciado lo evalúa |
| Retry a mano | `tenacity` | Una dependencia menos; son ~20 líneas |
| `generate()` devuelve error en `ModelResponse` | Lanzar siempre | Pedido explícito del enunciado ("mensaje estructurado"); el loop principal no se rompe |
| `stream()` lanza `LLMClientError` | Yield de error como texto | No contamina el contenido; idiomático |
| Ignorar temperatura en Anthropic (SDK 1.x) | Rechazar en schema / mandarla por `extra_body` | Mantiene una sola interfaz; `extra_body` podría dar 400 en modelos nuevos |
| Manager con composición + registry | Manager que hereda de Base / if-elif | Desacoplado, extensible |
| `Settings` con Pydantic plano | `pydantic-settings` | Respeta la lista de dependencias pedida |
| Groq vía SDK oficial `groq` (`AsyncGroq`) | `AsyncOpenAI(base_url="https://api.groq.com/openai/v1")` | Coherente con "usar SDKs oficiales" y excepciones propias de Groq. Costo: una dependencia más |
| `OpenAICompatibleClient` compartido | Duplicar código en `GroqClient` | Groq replica la API de Chat Completions; ~90% del código es idéntico |
| Groq como default | OpenAI como default | Gratis → cualquiera puede correr `main.py` sin pagar |
| Paquete `llm_client/` | Archivos planos en raíz | Más ordenado; imports claros. Costo: `python main.py` desde la raíz |

## 6. Qué revisaría si esto crece

- **Tool calling / structured output**: extender `ChatMessage` a contenido multimodal (`list[ContentBlock]`) y agregar `tools` a la config.
- **`stream_events()`** con objetos tipados (texto, usage, finish, error) para UIs.
- **Concurrencia controlada**: `asyncio.Semaphore` por proveedor para no autoinducir 429.
- **Observabilidad**: logging estructurado (provider, model, latency, tokens, retries) y costo estimado por request.
- **Fallback entre proveedores**: si OpenAI da `server`/`rate_limit` tras N intentos, reintentar en Anthropic.
- **Circuit breaker** si se usa en un servicio de larga vida.

## 7. Orden de implementación sugerido

1. `requirements.txt` + venv (`uv venv -p 3.12 && uv pip install -r requirements.txt`).
2. `errors.py` + `schemas.py` + `test_schemas.py`.
3. `base.py` (retry, clasificación, template method).
4. `openai_client.py` + tests con mocks.
5. `openai_compat.py` → refactor: `OpenAIClient` hereda de ahí; agregar `groq_client.py` + tests.
6. `anthropic_client.py` + tests con mocks.
7. `manager.py` + `test_manager.py`.
8. `main.py` → probar primero con Groq (gratis): normal, stream, gather, `--bad-key`.
9. `.env.example`, `.gitignore`, `README.md` (incluir cómo sacar la key de Groq).
10. `git init`, primer commit, push a GitHub.
