"""Retry/backoff behaviour of the exchange client."""

from __future__ import annotations

import asyncio

import pytest

from as_mm.client import ExchangeClient, ExchangeUnavailable
from as_mm.execution import OrderState


class StubResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class ScriptedSession:
    """Returns queued responses / raises queued exceptions, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None):
        self.calls.append(url)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return StubResponse(*item)

    def get(self, url):
        self.calls.append(url)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return StubResponse(*item)


async def no_sleep(_seconds):
    return None


def client(session, **kwargs):
    kwargs.setdefault("attempts", 2)
    kwargs.setdefault("backoff", 0.0)
    return ExchangeClient(session, sleep=no_sleep, **kwargs)


async def test_retries_a_5xx_then_succeeds():
    session = ScriptedSession([(503, {"detail": "busy"}), (200, {"ok": True})])
    status, body = await client(session).post_json("/order", {})
    assert (status, body) == (200, {"ok": True})
    assert len(session.calls) == 2


async def test_does_not_retry_a_4xx():
    session = ScriptedSession([(422, {"detail": "bad price"})])
    status, body = await client(session).post_json("/order", {})
    assert status == 422
    assert body == {"detail": "bad price"}
    assert len(session.calls) == 1


async def test_retries_a_transport_error():
    session = ScriptedSession([ConnectionResetError("reset"), (200, {"ok": True})])
    status, _ = await client(session).post_json("/order", {})
    assert status == 200
    assert len(session.calls) == 2


async def test_gives_up_after_the_attempt_budget():
    session = ScriptedSession([ConnectionResetError("a"), ConnectionResetError("b")])
    with pytest.raises(ExchangeUnavailable):
        await client(session, attempts=2).post_json("/order", {})
    assert len(session.calls) == 2


async def test_non_json_error_bodies_are_tolerated():
    class TextResponse(StubResponse):
        async def json(self):
            raise ValueError("not json")

    class Session:
        def post(self, url, json=None):
            return TextResponse(502, None)

    status, body = await client(Session(), attempts=1).post_json("/order", {})
    assert status == 502
    assert body is None


async def test_place_returns_none_when_the_engine_rejects():
    session = ScriptedSession([(422, {"detail": "price must be > 0"})])
    assert await client(session).place({"side": "buy"}) is None


async def test_place_returns_none_when_the_response_is_unusable():
    session = ScriptedSession([(200, {"amount": 1.0})])  # no orderId
    assert await client(session).place({"side": "buy"}) is None


async def test_place_returns_an_ack_on_success():
    payload = {
        "orderId": "o1",
        "secret_token": "t1",
        "active": True,
        "amount": 1.0,
        "fills": [],
    }
    session = ScriptedSession([(200, payload)])
    ack = await client(session).place({"side": "buy", "amount": 1.0})
    assert ack is not None
    assert ack.order_id == "o1"
    assert ack.remaining == pytest.approx(1.0)


async def test_cancel_maps_a_401_to_unknown_instead_of_raising():
    session = ScriptedSession([(401, {"detail": "Unauthorized"})])
    result = await client(session).cancel({"orderId": "o1", "amount": 1.0})
    assert result.state is OrderState.UNKNOWN
    assert not result.resolved


async def test_cancel_maps_a_transport_failure_to_unknown():
    session = ScriptedSession([ConnectionResetError("a"), ConnectionResetError("b")])
    result = await client(session).cancel({"orderId": "o1", "amount": 1.0})
    assert result.state is OrderState.UNKNOWN
    assert "transport" in result.reason


async def test_get_json_reports_a_non_200():
    session = ScriptedSession([(500, {"detail": "boom"})])
    with pytest.raises(ExchangeUnavailable):
        await client(session).get_json("/depth")


async def test_get_json_returns_the_body():
    session = ScriptedSession([(200, {"bids": [], "asks": []})])
    assert await client(session).get_json("/depth") == {"bids": [], "asks": []}


def test_client_validates_its_budget():
    with pytest.raises(ValueError):
        ExchangeClient(object(), attempts=0)
    with pytest.raises(ValueError):
        ExchangeClient(object(), backoff=-1.0)


async def test_url_joining_is_predictable():
    exchange = ExchangeClient(object(), base_url="http://localhost:8000/")
    assert exchange.url("/order") == "http://localhost:8000/order"
    assert exchange.url("order") == "http://localhost:8000/order"


async def test_cancelled_error_is_never_swallowed():
    class Cancelling:
        def post(self, url, json=None):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await client(Cancelling()).post_json("/order", {})
