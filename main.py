"""Script de validación: modo normal, streaming, concurrencia y error controlado.

Uso:
    python main.py                      # todos los proveedores con API key en .env
    python main.py --provider groq      # sólo uno
    python main.py --show-chunks        # marca con | cada fragmento recibido en el stream
    python main.py --bad-key            # demuestra el manejo de errores con una key inválida
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

from llm_client import (
    PROVIDERS,
    AsyncLLMManager,
    ChatMessage,
    GenerationConfig,
    LLMClientError,
    ModelResponse,
    Settings,
)

QUESTION = "¿Qué es la entropía?"
MESSAGES = [
    ChatMessage(role="system", content="Sos un profesor de física. Respondé en 2 o 3 oraciones."),
    ChatMessage(role="user", content=QUESTION),
]
CONFIG = GenerationConfig(temperature=0.3, max_tokens=200)


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_response(resp: ModelResponse) -> None:
    if resp.ok:
        print(resp.content)
        tokens = f"{resp.usage.input_tokens}→{resp.usage.output_tokens} tokens" if resp.usage else ""
        print(f"\n[{resp.model} · {resp.latency_ms} ms · {tokens} · {resp.finish_reason}]")
    else:
        print(f"Error controlado → {resp.error.type}: {resp.error.message}")


async def run_stream(
    llm: AsyncLLMManager, messages: list[ChatMessage], show_chunks: bool = False
) -> None:
    """Imprime cada fragmento apenas llega. Con `show_chunks` los separa con `|`
    para ver dónde termina uno y empieza el siguiente."""
    start = time.perf_counter()
    first_chunk_ms: float | None = None
    n_chunks = 0
    try:
        async for chunk in llm.stream(messages, CONFIG):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - start) * 1000
            n_chunks += 1
            print(f"{chunk}|" if show_chunks else chunk, end="", flush=True)
        total_ms = (time.perf_counter() - start) * 1000
        print(
            f"\n\n[{n_chunks} fragmentos · primer fragmento a los {first_chunk_ms or 0:.0f} ms"
            f" · último a los {total_ms:.0f} ms]"
        )
    except LLMClientError as exc:
        print(f"\nError controlado en stream → {exc.error.type}: {exc.error.message}")


async def demo_provider(provider: str, show_chunks: bool) -> None:
    async with AsyncLLMManager.from_env(provider) as llm:
        header(f"[{provider}] generate() — {QUESTION}")
        print_response(await llm.generate(MESSAGES, CONFIG))

        header(f"[{provider}] stream() — {QUESTION}")
        await run_stream(llm, MESSAGES, show_chunks)


async def demo_concurrency(provider: str) -> None:
    """Varias llamadas en paralelo: si el event loop no se bloquea, el tiempo
    total ≈ la más lenta, no la suma."""
    questions = ["¿Qué es un agujero negro?", "¿Qué es un fotón?", "¿Qué es la gravedad?"]
    header(f"[{provider}] {len(questions)} generate() concurrentes con asyncio.gather")
    async with AsyncLLMManager.from_env(provider) as llm:
        start = time.perf_counter()
        responses = await asyncio.gather(
            *(
                llm.generate([MESSAGES[0], ChatMessage(role="user", content=q)], CONFIG)
                for q in questions
            )
        )
        total_ms = (time.perf_counter() - start) * 1000

    for q, r in zip(questions, responses, strict=True):
        status = f"{r.latency_ms} ms" if r.ok else f"error {r.error.type}"
        print(f"  {q:<30} {status}")
    latencies = [r.latency_ms or 0 for r in responses]
    print(f"\n  Total: {total_ms:.0f} ms · suma individual: {sum(latencies):.0f} ms")


async def demo_bad_key(provider: str) -> None:
    header(f"[{provider}] API key inválida → error controlado (sin crash)")
    settings = Settings.model_validate(
        {"provider": provider, f"{provider}_api_key": "clave-invalida"}
    )
    async with AsyncLLMManager(settings) as llm:
        print_response(await llm.generate(MESSAGES, CONFIG))
        await run_stream(llm, MESSAGES)
    print("\nEl programa sigue corriendo ✔")


def available_providers() -> list[str]:
    return [p for p in ("groq", "openai", "anthropic") if os.environ.get(f"{p.upper()}_API_KEY")]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--bad-key", action="store_true", help="probar el manejo de errores")
    parser.add_argument(
        "--show-chunks", action="store_true", help="separar con | cada fragmento del stream"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="mostrar logs de reintentos")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Silenciar los logs de los SDKs/httpx para que la salida sea legible.
    for noisy in ("httpx", "openai", "anthropic", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()

    if args.bad_key:
        await demo_bad_key(args.provider or "groq")
        return

    providers = [args.provider] if args.provider else available_providers()
    if not providers:
        raise SystemExit("No hay API keys configuradas. Copiá .env.example a .env y completalo.")

    for provider in providers:
        await demo_provider(provider, args.show_chunks)
    await demo_concurrency(providers[0])


if __name__ == "__main__":
    asyncio.run(main())
