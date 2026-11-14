"""Resilient REST client for the limit order book matching engine.

The quoting loop must survive a flaky exchange: a transient 5xx, a dropped
connection or a 401 on one cancel used to propagate straight out of the loop
and terminate the process.  This layer collapses every exchange failure into
either a value (``(status, payload)``) or :class:`ExchangeUnavailable`, so the
caller decides what to do instead of crashing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .execution import (
    CancelResult,
    OrderAck,
    PayloadError,
    interpret_cancel,
    is_retryable_status,
    parse_order_ack,
)

__all__ = ["ExchangeUnavailable", "ExchangeClient"]


class ExchangeUnavailable(RuntimeError):
    """The exchange could not be reached, after every retry."""


class ExchangeClient:
    """Small async wrapper: retries, timeouts and typed results.

    ``session`` is duck-typed (anything with ``post(path, json=...)`` that
    returns an async context manager), so tests inject a fake without a socket.
    """

    def __init__(
        self,
        session: Any,
        *,
        base_url: str = "http://localhost:8000",
        attempts: int = 2,
        backoff: float = 0.1,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {attempts!r}")
        if backoff < 0:
            raise ValueError(f"backoff must be non-negative, got {backoff!r}")
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts
        self.backoff = backoff
        self.log = logger or logging.getLogger("as_mm.client")
        self._sleep = sleep

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    async def post_json(self, path: str, payload: dict) -> tuple[int, Any]:
        """POST ``payload`` and return ``(status, body)``.

        Retries 5xx/429 and transport errors, never retries 4xx.  Raises
        :class:`ExchangeUnavailable` when every attempt failed for transport
        reasons; HTTP error statuses are returned to the caller.
        """
        url = self.url(path)
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    status = int(response.status)
                    body: Any
                    try:
                        body = await response.json()
                    except Exception:  # noqa: BLE001 - non-JSON error pages
                        body = None
                if is_retryable_status(status) and attempt < self.attempts:
                    self.log.warning("%s -> HTTP %s, retrying", path, status)
                    await self._sleep(self.backoff * attempt)
                    continue
                return status, body
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport/protocol
                last_error = exc
                if attempt < self.attempts:
                    self.log.warning("%s -> %s, retrying", path, type(exc).__name__)
                    await self._sleep(self.backoff * attempt)
        raise ExchangeUnavailable(f"{path}: {last_error}") from last_error

    async def place(self, order: dict) -> OrderAck | None:
        """Submit an order, returning ``None`` when it could not be placed."""
        try:
            status, body = await self.post_json("/order", order)
        except ExchangeUnavailable as exc:
            self.log.error("order rejected by transport: %s", exc)
            return None
        if status != 200:
            self.log.error("order rejected: HTTP %s %s", status, body)
            return None
        try:
            return parse_order_ack(body)
        except PayloadError as exc:
            self.log.error("order response unusable: %s", exc)
            return None

    async def cancel(self, order: dict, resting_amount: float | None = None) -> CancelResult:
        """Cancel a resting order, never raising for an exchange answer.

        ``resting_amount`` is the size that was live at cancel time; it is what
        makes the fill quantity (``resting - cancelled``) computable, so the
        engine always passes it explicitly rather than relying on the payload.
        """
        resting = resting_amount if resting_amount is not None else order.get("amount", 0.0)
        try:
            status, body = await self.post_json("/cancel", order)
        except ExchangeUnavailable as exc:
            return CancelResult.in_doubt(f"transport: {exc}")
        return interpret_cancel(status, body, resting)

    async def get_json(self, path: str) -> Any:
        """GET a REST resource and return its decoded body."""
        url = self.url(path)
        try:
            async with self.session.get(url) as response:
                status = int(response.status)
                body = await response.json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport/protocol
            raise ExchangeUnavailable(f"{path}: {exc}") from exc
        if status != 200:
            raise ExchangeUnavailable(f"{path}: HTTP {status} {body}")
        return body
