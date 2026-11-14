"""Order-flow simulator maths and its tolerance for a failing exchange."""

from __future__ import annotations

import random

import pytest

from as_mm.client import ExchangeClient
from noise_simulator import MarketSimulator, abm_step, pareto_size, taker_limit_price


class FakeExchange:
    def __init__(self, *, fail_orders: bool = False):
        self.calls = []
        self.fail_orders = fail_orders
        self._n = 0

    def post(self, url, json=None):
        self.calls.append((url, json))
        if self.fail_orders:
            raise ConnectionResetError("down")
        return _Resp(url, json)

    def get(self, url):  # pragma: no cover - unused
        return _Resp(url, {})


class _Resp:
    def __init__(self, url, payload):
        self._url = url
        self._payload = payload
        self.status = 200

    async def json(self):
        if self._url.endswith("/cancel"):
            return {"success": True, "cancelled_amount": 1.0}
        return {
            "orderId": f"o-{id(self)}",
            "secret_token": "t",
            "active": True,
            "amount": self._payload["amount"],
            "fills": [],
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def client_for(fake):
    async def no_sleep(_):
        return None

    return ExchangeClient(fake, base_url="http://localhost:8000", sleep=no_sleep)


def test_abm_step_matches_the_discrete_process():
    assert abm_step(100.0, 0.5, 0.25, 0.0) == pytest.approx(100.0)
    assert abm_step(100.0, 0.5, 0.25, 2.0) == pytest.approx(100.5)


def test_abm_step_validates_its_inputs():
    with pytest.raises(ValueError):
        abm_step(100.0, 0.5, 0.0, 1.0)
    with pytest.raises(ValueError):
        abm_step(100.0, -0.5, 0.25, 1.0)


def test_taker_limit_always_crosses():
    assert taker_limit_price(100.0, "buy") == pytest.approx(500.0)
    assert taker_limit_price(100.0, "sell") == pytest.approx(20.0)


def test_taker_limit_validates_its_inputs():
    with pytest.raises(ValueError):
        taker_limit_price(100.0, "hold")
    with pytest.raises(ValueError):
        taker_limit_price(0.0, "buy")
    with pytest.raises(ValueError):
        taker_limit_price(100.0, "buy", cross_factor=1.0)


def test_pareto_size_is_clamped_to_its_caps():
    rng = random.Random(7)
    for _ in range(200):
        size = pareto_size(1.5, rng=rng)
        assert 0.1 <= size <= 25.0


def test_pareto_size_rejects_a_bad_alpha():
    rng = random.Random(7)
    with pytest.raises(ValueError):
        pareto_size(0.0, rng=rng)
    with pytest.raises(ValueError):
        pareto_size(1.5, rng=rng, cap_min=0.0)


def test_maker_quotes_straddle_the_fair_price():
    sim = MarketSimulator(rng=random.Random(11))
    bid, ask = sim.maker_quote()
    assert bid["side"] == "buy" and ask["side"] == "sell"
    assert bid["price"] < ask["price"]
    assert bid["client_tag"] == "sim_maker"


def test_simulator_validates_its_parameters():
    for kwargs in (
        {"fair_price": 0},
        {"sigma": -1},
        {"poisson_intensity": 0},
        {"pareto_alpha": 0},
        {"dt": 0},
        {"n_max": -1},
        {"quote_offset": (0.2, 0.1)},
        {"taker_size": (0.0, 1.0)},
    ):
        with pytest.raises(ValueError):
            MarketSimulator(**kwargs)


async def test_maker_cycle_tracks_and_sweeps_quotes():
    fake = FakeExchange()
    sim = MarketSimulator(n_max=2, rng=random.Random(3))
    await sim.maker_cycle(client_for(fake))

    assert sim.quotes_posted == 2
    assert len(sim.active_maker_orders) == 2

    await sim.maker_cycle(client_for(fake))
    assert len(sim.active_maker_orders) == 2, "the FIFO sweeper caps the tracker"
    assert sim.cancels_sent == 2


async def test_a_dead_exchange_does_not_stop_the_simulation():
    fake = FakeExchange(fail_orders=True)
    sim = MarketSimulator(n_max=0, rng=random.Random(5))
    await sim.maker_cycle(client_for(fake))
    assert sim.quotes_posted == 0
    assert sim.active_maker_orders == []


async def test_taker_order_is_an_ioc_that_crosses():
    sim = MarketSimulator(rng=random.Random(13))
    order = sim.taker_order()
    assert order["order_type"] == "IOC"
    assert order["client_tag"] == "sim_taker"
    if order["side"] == "buy":
        assert order["price"] > sim.fair_price
    else:
        assert order["price"] < sim.fair_price
