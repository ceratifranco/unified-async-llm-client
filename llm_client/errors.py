"""Tipos de error controlados del cliente."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client.schemas import LLMError


class ErrorType(StrEnum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    NETWORK = "network"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    SERVER = "server"
    UNKNOWN = "unknown"


class LLMClientError(Exception):
    """Excepción única que expone el cliente (usada por `stream()`).

    Envuelve un `LLMError` estructurado para que el caller no tenga que
    conocer las excepciones de cada SDK.
    """

    def __init__(self, error: LLMError) -> None:
        super().__init__(f"[{error.provider}:{error.type}] {error.message}")
        self.error = error
