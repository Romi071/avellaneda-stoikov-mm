"""Behaviour of the quoting engine against the fake exchange."""

from __future__ import annotations

import asyncio
import json

import pytest

from as_mm.client import ExchangeClient
from as_mm.model import ModelParams, quote_prices
from as_mm.telemetry import decode_datagram
from market_maker import (
    Avellaneda_Stoikov,
    RestingOrder,
    build_parser,
    params_from_args,
    run_once,
)


class FakeSocket:
    def __init__(self):
        self.sent: list[tuple[bytes, tuple]] = []

    def sendto(self, payload, address):
        self.sent.append((payload, address))


def make_engine(exchange=None, **kwargs):
    sock = kwargs.pop("telemetry_socket", None) or FakeSocket()
    kwargs.setdefault("ingest_timeout", 0.05)
    engine = Avellaneda_Stoikov(telemetry_socket=sock, **kwargs)
    client = ExchangeClient(exchange or object(), base_url="http://localhost:8000")
    return engine, client, sock


def make_client(exchange):
    return ExchangeClient(exchange, base_url="http://localhost:8000")


# --------------------------------------------------------------------------
# one quoting cycle
# --------------------------------------------------------------------------
async def test_quote_once_posts_both_sides_at_the_model_prices(exchange):
    engine, _, sock = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)
    client = make_client(fake)

    quote = await engine.quote_once(client)

    assert quote is not None
    assert quote == quote_prices(100.0, 0.0, engine.params)
    orders = fake.order_calls()
    assert [order["side"] for order in orders] == ["buy", "sell"]
    assert orders[0]["price"] == pytest.approx(quote.bid)
    assert orders[1]["price"] == pytest.approx(quote.ask)
    assert orders[0]["amount"] == pytest.approx(1.0)
    assert orders[0]["client_tag"] == "as_mm"
    assert sock.sent, "telemetry should be published every cycle"


async def test_quote_once_is_flat_when_the_book_is_one_sided(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": []})
    fake = exchange(engine)

    assert engine.mid_price is None
    assert await engine.quote_once(make_client(fake)) is None
    assert fake.order_calls() == [], "must park with zero resting quotes"


async def test_quote_once_cancels_before_requoting(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)
    client = make_client(fake)

    await engine.quote_once(client)
    fake.calls.clear()
    await engine.quote_once(client)

    paths = [url.rsplit("/", 1)[-1] for url, _ in fake.calls]
    assert paths[0] == "cancel"
    assert paths[1] == "cancel"
    assert paths[2:] == ["order", "order"]


# --------------------------------------------------------------------------
# inventory bounds
# --------------------------------------------------------------------------
async def test_no_bid_is_posted_at_the_inventory_cap(exchange):
    engine, _, _ = make_engine(max_inventory=100.0)
    engine.inventory = 100.0
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)

    await engine.quote_once(make_client(fake))

    sides = [order["side"] for order in fake.order_calls()]
    assert sides == ["sell"], "the shipped <= gate still posted a bid at q_max"


async def test_the_bid_is_sized_to_the_remaining_headroom(exchange):
    engine, _, _ = make_engine(max_inventory=100.0, lot=1.0)
    engine.inventory = 99.5
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)

    await engine.quote_once(make_client(fake))

    bid = next(order for order in fake.order_calls() if order["side"] == "buy")
    assert bid["amount"] == pytest.approx(0.5)


async def test_no_ask_is_posted_at_the_short_cap(exchange):
    engine, _, _ = make_engine(max_inventory=100.0)
    engine.inventory = -100.0
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)

    await engine.quote_once(make_client(fake))

    sides = [order["side"] for order in fake.order_calls()]
    assert sides == ["buy"]


# --------------------------------------------------------------------------
# fill accounting
# --------------------------------------------------------------------------
async def test_instant_fills_are_booked_and_the_remainder_is_tracked(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)
    fake.on_order = lambda order: (0.6, [{"price": order["price"], "amount": 0.4}])

    await engine.quote_once(make_client(fake))

    # 0.4 bought at the bid and 0.4 sold at the ask
    assert engine.inventory == pytest.approx(0.0, abs=1e-9)
    assert engine.balance == pytest.approx(0.4 * (100.82 - 99.18), abs=1e-9)
    assert set(engine.active_orders) == {"buy", "sell"}
    assert engine.active_orders["buy"].amount == pytest.approx(0.6)


async def test_a_cancel_books_the_fill_of_the_resting_quantity(exchange):
    engine, _, _ = make_engine()
    engine.active_orders["buy"] = RestingOrder(
        order_id="o1", secret_token="t", amount=1.0, price=99.0, side="buy"
    )
    fake = exchange(engine)
    fake.orders["o1"] = {"order": {}, "remaining": 1.0}
    fake.cancel_amount_override = 0.6  # 0.4 traded while the quote was resting

    assert await engine.cancel_side(make_client(fake), "buy") is True
    assert engine.inventory == pytest.approx(0.4)
    assert engine.balance == pytest.approx(-0.4 * 99.0)
    assert "buy" not in engine.active_orders


async def test_an_already_filled_order_404_books_the_full_resting_fill(exchange):
    engine, _, _ = make_engine()
    engine.active_orders["sell"] = RestingOrder(
        order_id="o2", secret_token="t", amount=2.0, price=101.0, side="sell"
    )
    fake = exchange(engine, cancel_status=404)
    client = make_client(fake)

    assert await engine.cancel_side(client, "sell") is True
    assert engine.inventory == pytest.approx(-2.0)
    assert engine.balance == pytest.approx(202.0)


async def test_an_unconfirmed_cancel_keeps_the_order_and_books_nothing(exchange):
    engine, _, _ = make_engine()
    engine.active_orders["buy"] = RestingOrder(
        order_id="o1", secret_token="t", amount=1.0, price=99.0, side="buy"
    )
    fake = exchange(engine, cancel_status=401)

    assert await engine.cancel_side(make_client(fake), "buy") is False
    assert engine.inventory == pytest.approx(0.0), "no phantom fill on an unknown state"
    assert "buy" in engine.active_orders


async def test_no_new_quote_is_sent_while_a_cancel_is_unconfirmed(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    engine.active_orders["buy"] = RestingOrder(
        order_id="o1", secret_token="t", amount=1.0, price=99.0, side="buy"
    )
    fake = exchange(engine, cancel_status=401)

    assert await engine.quote_once(make_client(fake)) is None
    assert fake.order_calls() == [], "stacking a second quote per side would double exposure"


async def test_a_cancel_without_a_following_broadcast_parks_the_cycle(exchange):
    engine, _, _ = make_engine(ingest_timeout=0.02)
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine, broadcast_on_cancel=False)
    engine.active_orders["buy"] = RestingOrder(
        order_id="o1", secret_token="t", amount=1.0, price=99.0, side="buy"
    )

    assert await engine.quote_once(make_client(fake)) is None
    assert fake.order_calls() == []


# --------------------------------------------------------------------------
# resilience
# --------------------------------------------------------------------------
async def test_a_degenerate_quote_is_parked_instead_of_raised(exchange):
    engine, _, _ = make_engine(params=ModelParams(gamma=250.0, sigma=9.0, tau=3600.0))
    engine.inventory = 33.0
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)

    assert await engine.quote_once(make_client(fake)) is None
    assert fake.order_calls() == [], "a negative reservation price must not be posted"


async def test_a_rejected_order_does_not_kill_the_cycle(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine, order_status=422)

    quote = await engine.quote_once(make_client(fake))
    assert quote is not None
    assert engine.active_orders == {}


async def test_a_dead_transport_does_not_kill_the_cycle(exchange):
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine, raise_on_post=ConnectionResetError("down"))
    fake.fail_first_n_posts = 0

    assert await engine.quote_once(make_client(fake)) is not None
    assert engine.active_orders == {}


async def test_the_quoting_loop_survives_a_failing_cycle(exchange, monkeypatch):
    engine, _, _ = make_engine(quote_interval=0.01)
    calls = {"n": 0}

    async def boom(client):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "quote_once", boom)
    task = asyncio.create_task(engine.quoting_loop(object()))
    await asyncio.sleep(0.08)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls["n"] >= 2, "the loop must keep cycling after an exception"


async def test_cancel_all_sweeps_what_it_can_and_never_raises(exchange):
    engine, _, _ = make_engine()
    for side in ("buy", "sell"):
        engine.active_orders[side] = RestingOrder(
            order_id=f"c-{side}", secret_token="t", amount=1.0, price=100.0, side=side
        )
    fake = exchange(engine, raise_on_post=ConnectionResetError("down"))

    swept = await engine.cancel_all(make_client(fake))
    assert swept == 0
    assert engine.active_orders == {}, "the tracker is cleared even when the sweep fails"


async def test_socket_listener_reconnects_after_a_transport_error(exchange):
    from tests.conftest import FakeMessage, FakeWebSocket

    engine, _, _ = make_engine(initial_backoff=0.01)
    attempts = {"n": 0}

    class Session:
        def ws_connect(self, url):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionRefusedError("[Errno 111] Connection refused")
            return FakeWebSocket([FakeMessage({"bids": [[99.0, 1.0]], "asks": [[101.0, 1.0]]})])

    task = asyncio.create_task(engine.socket_listener(Session(), "ws://localhost:8000/stream"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts["n"] >= 2, "a refused connection must be retried, not fatal"
    assert engine.mid_price == pytest.approx(100.0)


async def test_malformed_snapshots_are_dropped_without_raising():
    engine, _, _ = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    assert engine.ingest({"bids": "nope", "asks": []}) == pytest.approx(100.0)
    assert engine.ingest({"nope": 1}) == pytest.approx(100.0)


async def test_the_book_barrier_does_not_lose_a_wakeup():
    engine, _, _ = make_engine(ingest_timeout=0.5)
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    # The event is already set: a naive clear()-then-wait() would block here.
    assert engine.event.is_set()
    assert await engine._await_ingest(0) is True


async def test_the_book_barrier_times_out_on_a_stalled_feed():
    engine, _, _ = make_engine(ingest_timeout=0.02)
    assert await engine._await_ingest(engine._book_seq) is False


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
async def test_telemetry_payload_matches_the_visualizer_contract(exchange):
    engine, _, sock = make_engine()
    engine.ingest({"bids": [[99.5, 1.0]], "asks": [[100.5, 1.0]]})
    fake = exchange(engine)

    await engine.quote_once(make_client(fake))

    raw, address = sock.sent[-1]
    assert address == ("127.0.0.1", 9999)
    payload = decode_datagram(raw)
    assert set(payload) == {"balance", "inventory", "res_price", "wealth"}
    assert payload["wealth"] == pytest.approx(payload["balance"] + payload["inventory"] * 100.0)
    assert json.loads(raw) == payload


def test_engine_validates_its_construction_parameters():
    for kwargs in (
        {"max_inventory": 0},
        {"max_inventory": -5},
        {"lot": 0},
        {"quote_interval": 0},
        {"ingest_timeout": 0},
    ):
        with pytest.raises(ValueError):
            Avellaneda_Stoikov(telemetry_socket=FakeSocket(), **kwargs)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_params_from_args_rejects_an_invalid_gamma():
    parser = build_parser()
    args = parser.parse_args(["--gamma", "0"])
    with pytest.raises(SystemExit):
        params_from_args(parser, args)


def test_params_from_args_passes_valid_values():
    parser = build_parser()
    args = parser.parse_args(["--gamma", "2", "--kappa", "3", "--tick", "0.5"])
    params = params_from_args(parser, args)
    assert isinstance(params, ModelParams)
    assert (params.gamma, params.kappa, params.tick) == (2.0, 3.0, 0.5)


async def test_run_once_dry_run_prints_without_trading(exchange):
    engine, _, _ = make_engine()
    fake = exchange(engine)

    summary = await run_once(engine, make_client(fake), dry_run=True)

    assert summary["dry_run"] is True
    assert summary["mid"] == pytest.approx(100.0)
    assert summary["bid"] < summary["mid"] < summary["ask"]
    assert fake.order_calls() == []


async def test_run_once_places_and_sweeps(exchange):
    engine, _, _ = make_engine()
    fake = exchange(engine)

    summary = await run_once(engine, make_client(fake), dry_run=False)

    assert "orders" in summary
    assert engine.active_orders == {}
    assert len(fake.order_calls()) == 2
    assert len(fake.cancel_calls()) == 2


async def test_run_once_fails_loudly_on_a_one_sided_book(exchange):
    engine, _, _ = make_engine()
    fake = exchange(engine)
    fake.book = {"bids": [], "asks": [[100.5, 1.0]]}

    with pytest.raises(SystemExit):
        await run_once(engine, make_client(fake), dry_run=True)
