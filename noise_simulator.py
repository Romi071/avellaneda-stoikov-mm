"""Exogenous stochastic order flow for the matching engine.

Two decoupled coroutines share one connection pool:

``maker_loop``
    Evolves an unobservable fair price with discrete Arithmetic Brownian Motion
    ``S_{t+dt} = S_t + sigma * sqrt(dt) * Z`` and streams two-sided ``GTC``
    quotes at uniform offsets around it, sweeping the oldest quote once more
    than ``n_max`` are resting.
``taker_loop``
    Fires spread-crossing ``IOC`` orders on a Poisson schedule with a capped
    Pareto size, so the book sees both retail noise and multi-level sweeps.

Nothing executes on import; a failing request is logged and retried instead of
killing the simulation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random

import aiohttp

from as_mm.client import ExchangeClient

DEFAULT_BASE_URL = "http://localhost:8000"

log = logging.getLogger("as_mm.noise_simulator")


def abm_step(price: float, sigma: float, dt: float, z: float) -> float:
    """One Arithmetic Brownian Motion step of the latent fair price."""
    if dt <= 0:
        raise ValueError(f"dt must be strictly positive, got {dt!r}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    return price + sigma * math.sqrt(dt) * z


def taker_limit_price(fair_price: float, side: str, cross_factor: float = 5.0) -> float:
    """Aggressive limit that always crosses the book, rounded to the tick grid.

    The taker wants certainty of execution, not price discovery: a buy is sent
    far above the fair price and a sell far below, and the engine fills it
    against whatever liquidity is resting at a better price.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if fair_price <= 0:
        raise ValueError(f"fair_price must be strictly positive, got {fair_price!r}")
    if cross_factor <= 1:
        raise ValueError(f"cross_factor must be greater than 1, got {cross_factor!r}")
    price = fair_price * cross_factor if side == "buy" else fair_price / cross_factor
    return round(price, 2)


def pareto_size(
    alpha: float,
    *,
    rng: random.Random | random.SystemRandom,
    scale: float = 0.1,
    cap_min: float = 0.1,
    cap_max: float = 25.0,
) -> float:
    """Bounded Pareto order size ``clamp(scale * Pareto(alpha))``."""
    if alpha <= 0:
        raise ValueError(f"alpha must be strictly positive, got {alpha!r}")
    if scale <= 0:
        raise ValueError(f"scale must be strictly positive, got {scale!r}")
    if cap_min <= 0 or cap_max < cap_min:
        raise ValueError(f"invalid size caps: {cap_min!r}..{cap_max!r}")
    return max(cap_min, min(cap_max, scale * rng.paretovariate(alpha)))


class MarketSimulator:
    """ABM latent price, passive maker quotes and Poisson IOC takers."""

    def __init__(
        self,
        *,
        fair_price: float = 100.0,
        sigma: float = 0.5,
        poisson_intensity: float = 2.0,
        pareto_alpha: float = 1.5,
        dt: float = 0.25,
        n_max: int = 50,
        quote_offset: tuple[float, float] = (0.02, 0.20),
        maker_size: tuple[float, float] = (0.1, 5.0),
        taker_size: tuple[float, float] = (0.1, 25.0),
        cross_factor: float = 5.0,
        rng: random.Random | random.SystemRandom | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if fair_price <= 0:
            raise ValueError(f"fair_price must be strictly positive, got {fair_price!r}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        if poisson_intensity <= 0:
            raise ValueError(
                f"poisson_intensity must be strictly positive, got {poisson_intensity!r}"
            )
        if pareto_alpha <= 0:
            raise ValueError(f"pareto_alpha must be strictly positive, got {pareto_alpha!r}")
        if dt <= 0:
            raise ValueError(f"dt must be strictly positive, got {dt!r}")
        if n_max < 0:
            raise ValueError(f"n_max must be non-negative, got {n_max!r}")
        for name, bounds in (("quote_offset", quote_offset), ("maker_size", maker_size)):
            low, high = bounds
            if low < 0 or high < low:
                raise ValueError(f"{name} must be 0 <= low <= high, got {bounds!r}")
        if taker_size[0] <= 0 or taker_size[1] < taker_size[0]:
            raise ValueError(f"taker_size must be positive and ordered, got {taker_size!r}")

        self.fair_price = float(fair_price)
        self.sigma = float(sigma)
        self.poisson_intensity = float(poisson_intensity)
        self.pareto_alpha = float(pareto_alpha)
        self.dt = float(dt)
        self.n_max = int(n_max)
        self.quote_offset = (float(quote_offset[0]), float(quote_offset[1]))
        self.maker_size = (float(maker_size[0]), float(maker_size[1]))
        self.taker_size = (float(taker_size[0]), float(taker_size[1]))
        self.cross_factor = float(cross_factor)
        self.rng = rng or random.SystemRandom()
        self.log = logger or log

        self.active_maker_orders: list[dict] = []
        self.quotes_posted = 0
        self.cancels_sent = 0

    def maker_quote(self) -> tuple[dict, dict]:
        """Next two-sided passive quote around the current fair price."""
        self.fair_price = abm_step(self.fair_price, self.sigma, self.dt, self.rng.gauss(0.0, 1.0))
        low, high = self.quote_offset
        bid_price = round(self.fair_price - self.rng.uniform(low, high), 2)
        ask_price = round(self.fair_price + self.rng.uniform(low, high), 2)
        size_low, size_high = self.maker_size
        bid = {
            "side": "buy",
            "price": bid_price,
            "amount": self.rng.uniform(size_low, size_high),
            "client_tag": "sim_maker",
        }
        ask = {
            "side": "sell",
            "price": ask_price,
            "amount": self.rng.uniform(size_low, size_high),
            "client_tag": "sim_maker",
        }
        return bid, ask

    def taker_order(self) -> dict:
        """Aggressive IOC sized from the capped Pareto law."""
        side = "buy" if self.rng.random() < 0.5 else "sell"
        return {
            "side": side,
            "price": taker_limit_price(self.fair_price, side, self.cross_factor),
            "amount": pareto_size(
                self.pareto_alpha,
                rng=self.rng,
                cap_min=self.taker_size[0],
                cap_max=self.taker_size[1],
            ),
            "order_type": "IOC",
            "client_tag": "sim_taker",
        }

    async def maker_loop(self, client: ExchangeClient) -> None:
        while True:
            await asyncio.sleep(self.dt)
            try:
                await self.maker_cycle(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep simulating
                self.log.exception("maker cycle failed: %s", exc)

    async def maker_cycle(self, client: ExchangeClient) -> None:
        bid, ask = self.maker_quote()
        for order in (bid, ask):
            ack = await client.place(order)
            if ack is None:
                continue
            self.quotes_posted += 1
            if ack.remaining > 0:
                self.active_maker_orders.append(
                    {"orderId": ack.order_id, "secret_token": ack.secret_token}
                )

        while len(self.active_maker_orders) > self.n_max:
            stale = self.active_maker_orders.pop(0)
            self.cancels_sent += 1
            result = await client.cancel(stale, stale.get("amount", 0.0))
            if not result.resolved:
                self.log.debug("stale quote cancel unconfirmed: %s", result.reason)

    async def taker_loop(self, client: ExchangeClient) -> None:
        while True:
            await asyncio.sleep(self.rng.expovariate(self.poisson_intensity))
            try:
                await client.place(self.taker_order())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep simulating
                self.log.exception("taker cycle failed: %s", exc)

    async def run(self, client: ExchangeClient) -> None:
        await asyncio.gather(self.maker_loop(client), self.taker_loop(client))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exogenous ABM maker / Pareto IOC taker order-flow generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="matching engine REST base URL"
    )
    parser.add_argument("--fair-price", type=float, default=100.0, help="initial latent price")
    parser.add_argument("--sigma", type=float, default=0.5, help="ABM volatility per sqrt(second)")
    parser.add_argument("--dt", type=float, default=0.25, help="maker heartbeat (seconds)")
    parser.add_argument("--intensity", type=float, default=2.0, help="taker arrivals per second")
    parser.add_argument("--pareto-alpha", type=float, default=1.5, help="taker size tail exponent")
    parser.add_argument("--n-max", type=int, default=50, help="resting maker quotes before GC")
    parser.add_argument("--cross-factor", type=float, default=5.0, help="taker limit distance")
    parser.add_argument("--seed", type=int, default=None, help="seed the RNG for a repeatable run")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser


async def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper()),
            format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        )
    except AttributeError:
        parser.error(f"unknown log level {args.log_level!r}")

    try:
        rng = random.Random(args.seed) if args.seed is not None else None
        simulator = MarketSimulator(
            fair_price=args.fair_price,
            sigma=args.sigma,
            poisson_intensity=args.intensity,
            pareto_alpha=args.pareto_alpha,
            dt=args.dt,
            n_max=args.n_max,
            cross_factor=args.cross_factor,
            rng=rng,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=5.0, sock_read=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = ExchangeClient(session, base_url=args.base_url)
        await simulator.run(client)
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Console-script entry point (``as-bot`` / ``as-sim``)."""
    try:
        return asyncio.run(main(argv))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
