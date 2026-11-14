"""Shared fixtures: a stateful fake matching engine and a synthetic ledger.

The fake mirrors the REST/WS surface of Romi071/limit-order-book
(``server.py`` + ``order_book.py``): the ``/order`` acknowledgement shape, the
three cancel answers (200 / 404 / 401), and a book broadcast after every
accepted mutation.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `pytest` (bare binary) does not add the cwd
    sys.path.insert(0, str(ROOT))


class FakeResponse:
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeExchange:
    """Duck-typed replacement for ``aiohttp.ClientSession``.

    ``on_order`` lets a test decide how much of an order fills immediately:
    it receives the order dict and returns ``(remaining, fills)``.
    """

    def __init__(self, engine: Any = None, book: dict | None = None) -> None:
        self.engine = engine
        self.book = book if book is not None else {"bids": [[99.5, 2.0]], "asks": [[100.5, 2.0]]}
        self.orders: dict[str, dict] = {}
        self.calls: list[tuple[str, dict | None]] = []
        self.cancel_status = 200
        self.cancel_amount_override: float | None = None
        self.order_status = 200
        self.order_payload: dict | None = None
        self.on_order = None
        self.broadcast_on_cancel = True
        self._next = 0
        self.raise_on_post: Exception | None = None
        self.fail_first_n_posts = 0

    # -- helpers ---------------------------------------------------------
    def _ack(self, order: dict) -> FakeResponse:
        self._next += 1
        order_id = f"order-{self._next}"
        remaining = order["amount"]
        fills: list[dict] = []
        if self.on_order is not None:
            remaining, fills = self.on_order(order)
        self.orders[order_id] = {"order": order, "remaining": remaining}
        payload = {
            "orderId": order_id,
            "secret_token": f"token-{self._next}",
            "client_tag": order.get("client_tag"),
            "active": remaining > 0,
            "amount": remaining,
            "fills": fills,
        }
        if self.order_payload is not None:
            payload = self.order_payload
        return FakeResponse(self.order_status, payload)

    def _cancel(self, order: dict) -> FakeResponse:
        order_id = order.get("orderId")
        if self.cancel_status == 200:
            resting = self.orders.get(order_id, {}).get("remaining", 0.0)
            if self.cancel_amount_override is not None:
                resting = self.cancel_amount_override
            self.orders.pop(order_id, None)
            if self.broadcast_on_cancel and self.engine is not None:
                self.engine.ingest(self.book)
            return FakeResponse(200, {"success": True, "cancelled_amount": resting})
        if self.cancel_status == 404:
            return FakeResponse(404, {"success": False, "cancelled_amount": 0})
        return FakeResponse(self.cancel_status, {"success": False, "detail": "Unauthorized"})

    # -- aiohttp surface -------------------------------------------------
    def post(self, url: str, json: dict | None = None) -> FakeResponse:
        self.calls.append((url, json))
        if self.raise_on_post is not None:
            raise self.raise_on_post
        if self.fail_first_n_posts > 0:
            self.fail_first_n_posts -= 1
            raise ConnectionResetError("connection reset by peer")
        if url.endswith("/cancel"):
            return self._cancel(json or {})
        return self._ack(json or {})

    def get(self, url: str) -> FakeResponse:
        self.calls.append((url, None))
        return FakeResponse(200, self.book)

    def order_calls(self) -> list[dict]:
        return [payload for url, payload in self.calls if url.endswith("/order") and payload]

    def cancel_calls(self) -> list[dict]:
        return [payload for url, payload in self.calls if url.endswith("/cancel") and payload]


class FakeMessage:
    def __init__(self, payload: Any, kind: int = 1) -> None:
        self._payload = payload
        self.type = kind

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeWebSocket:
    def __init__(self, messages: list[FakeMessage], *, error: Exception | None = None) -> None:
        self.messages = messages
        self.error = error

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def __aiter__(self):
        for message in self.messages:
            yield message
            await asyncio.sleep(0)
        if self.error is not None:
            raise self.error


@pytest.fixture
def book() -> dict:
    return {"bids": [[99.5, 2.0], [99.4, 1.0]], "asks": [[100.5, 3.0], [100.6, 1.0]]}


@pytest.fixture
def exchange(book):
    def factory(engine=None, **kwargs):
        fake = FakeExchange(engine, book)
        for key, value in kwargs.items():
            setattr(fake, key, value)
        return fake

    return factory


LEDGER_SCHEMA = """
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    takerSide TEXT,
    takerId TEXT,
    takerTag TEXT,
    makerId TEXT,
    makerTag TEXT,
    price REAL,
    amount REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


@pytest.fixture
def ledger(tmp_path):
    """Build a ledger .db from ``(takerSide, takerTag, makerTag, price, amount)`` rows."""

    def build(rows: list[tuple], name: str = "ledger.db") -> Path:
        path = tmp_path / name
        connection = sqlite3.connect(path)
        connection.execute(LEDGER_SCHEMA)
        connection.executemany(
            "INSERT INTO trades (takerSide, takerTag, makerTag, price, amount)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        connection.close()
        return path

    return build
