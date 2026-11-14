"""Avellaneda-Stoikov (2008) quoting engine for the local matching engine.

Consumes live L2 depth from ``ws://localhost:8000/stream``, tracks inventory
``q_t``, cash ``X_t`` and mark-to-market wealth ``W_t = X_t + q_t S_t``, and
continuously quotes two-sided liquidity (``client_tag: 'as_mm'``) around the
stationary reservation price and optimal spread::

    r(s, q, tau) = s - q * gamma * sigma**2 * tau
    Delta(tau)   = gamma * sigma**2 * tau + (2 / gamma) * ln(1 + gamma / kappa)

The quoting maths lives in :mod:`as_mm.model`, the exchange contract in
:mod:`as_mm.execution` and :mod:`as_mm.client`, and the telemetry wire format
in :mod:`as_mm.telemetry`.  Nothing executes on import.

Usage::

    python market_maker.py                       # quote until interrupted
    python market_maker.py --once --dry-run      # one quote, print it, exit
    python market_maker.py --gamma 0.5 --max-inventory 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
from dataclasses import dataclass

import aiohttp

from as_mm.client import ExchangeClient, ExchangeUnavailable
from as_mm.execution import CancelResult, OrderState, book_execution, fill_totals
from as_mm.model import ModelParams, Quote, inventory_headroom, quote_prices
from as_mm.telemetry import encode_telemetry, mid_price

DEFAULT_EXCHANGE_URL = "http://localhost:8000"
DEFAULT_STREAM_URL = "ws://localhost:8000/stream"
CLIENT_TAG = "as_mm"

log = logging.getLogger("as_mm.market_maker")


@dataclass
class RestingOrder:
    """A quote the engine still holds in its book."""

    order_id: str
    secret_token: str
    amount: float
    price: float
    side: str

    def as_payload(self) -> dict:
        return {"orderId": self.order_id, "secret_token": self.secret_token}


class Avellaneda_Stoikov:
    """Two-sided quoting engine with symmetric inventory bounds."""

    def __init__(
        self,
        params: ModelParams | None = None,
        *,
        max_inventory: float = 100.0,
        lot: float = 1.0,
        quote_interval: float = 0.5,
        client_tag: str = CLIENT_TAG,
        udp_target: tuple[str, int] = ("127.0.0.1", 9999),
        telemetry_socket: socket.socket | None = None,
        ingest_timeout: float = 2.0,
        initial_backoff: float = 0.5,
        max_ws_backoff: float = 8.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.params = params or ModelParams()
        if max_inventory <= 0:
            raise ValueError(f"max_inventory must be strictly positive, got {max_inventory!r}")
        if lot <= 0:
            raise ValueError(f"lot must be strictly positive, got {lot!r}")
        if quote_interval <= 0:
            raise ValueError(f"quote_interval must be strictly positive, got {quote_interval!r}")
        if ingest_timeout <= 0:
            raise ValueError(f"ingest_timeout must be strictly positive, got {ingest_timeout!r}")
        if initial_backoff <= 0:
            raise ValueError(f"initial_backoff must be strictly positive, got {initial_backoff!r}")

        self.max_inventory = float(max_inventory)
        self.lot = float(lot)
        self.quote_interval = float(quote_interval)
        self.client_tag = client_tag
        self.ingest_timeout = float(ingest_timeout)
        self.initial_backoff = float(initial_backoff)
        self.max_ws_backoff = float(max_ws_backoff)
        self.log = logger or log

        self.mid_price: float | None = None
        self.balance = 0.0
        self.inventory = 0.0
        self.active_orders: dict[str, RestingOrder] = {}
        self.quote: Quote | None = None
        self.last_telemetry: dict | None = None

        # Book-update barrier: a monotonic counter plus an event avoids the
        # lost wakeup the bare Event had (a broadcast arriving between the
        # clear() and the wait() was dropped, leaving the loop waiting for a
        # message that might never come on a quiet book).
        self._book_seq = 0
        self.event = asyncio.Event()

        self.udp_target = udp_target
        self.udp_socket = telemetry_socket or self._open_telemetry_socket(udp_target)

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------
    @staticmethod
    def _open_telemetry_socket(target: tuple[str, int]) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        log.debug("telemetry socket ready for %s:%s", *target)
        return sock

    def telemetry_payload(self, mid: float, quote: Quote) -> dict:
        wealth = self.balance + self.inventory * mid
        return {
            "balance": self.balance,
            "inventory": self.inventory,
            "res_price": quote.reservation,
            "wealth": wealth,
        }

    def publish_telemetry(self, mid: float, quote: Quote) -> dict:
        """Fire-and-forget one datagram; a dead listener is not an error."""
        payload = self.telemetry_payload(mid, quote)
        self.last_telemetry = payload
        try:
            self.udp_socket.sendto(
                encode_telemetry(
                    payload["balance"],
                    payload["inventory"],
                    payload["res_price"],
                    payload["wealth"],
                ),
                self.udp_target,
            )
        except OSError as exc:  # pragma: no cover - host dependent
            self.log.warning("telemetry send failed: %s", exc)
        return payload

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------
    def ingest(self, snapshot: object) -> float | None:
        """Apply one depth snapshot, returning the new mid price (or ``None``)."""
        try:
            mid = mid_price(snapshot)
        except ValueError as exc:
            self.log.warning("dropping malformed book snapshot: %s", exc)
            return self.mid_price
        self.mid_price = mid
        self._book_seq += 1
        self.event.set()
        return mid

    async def _await_ingest(self, target: int, timeout: float | None = None) -> bool:
        """Wait until a snapshot newer than ``target`` has been ingested."""
        budget = self.ingest_timeout if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        while self._book_seq <= target:
            self.event.clear()
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self.event.wait(), remaining)
            except (TimeoutError, asyncio.TimeoutError):
                return False
        return True

    async def socket_listener(self, session: aiohttp.ClientSession, url: str = DEFAULT_STREAM_URL):
        """Stream depth forever, reconnecting with exponential backoff."""
        delay = self.initial_backoff
        while True:
            try:
                async with session.ws_connect(url) as ws:
                    self.log.info("depth stream connected: %s", url)
                    delay = self.initial_backoff
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                snapshot = message.json()
                            except (ValueError, TypeError):
                                self.log.warning("dropping non-JSON depth frame")
                                continue
                            self.ingest(snapshot)
                        elif message.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any transport failure
                self.log.warning("depth stream error (%s); reconnecting in %.1fs", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.max_ws_backoff)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def _book(self, side: str, shares: float, notional: float, price: float) -> None:
        self.inventory, self.balance = book_execution(
            self.inventory, self.balance, side, shares, notional
        )
        if shares:
            self.log.info(
                "fill %s %.4f @ %.2f (inventory %+.4f, cash %+.2f)",
                side,
                shares,
                price,
                self.inventory,
                self.balance,
            )

    async def cancel_side(self, client: ExchangeClient, side: str) -> bool:
        """Cancel the resting quote on ``side``; ``False`` when unconfirmed."""
        order = self.active_orders.get(side)
        if order is None:
            return True

        target = self._book_seq
        result: CancelResult = await client.cancel(order.as_payload(), order.amount)
        if not result.resolved:
            self.log.warning("cancel %s unconfirmed (%s); keeping it tracked", side, result.reason)
            return False

        self._book(side, result.filled, result.filled * order.price, order.price)
        del self.active_orders[side]

        if result.state is OrderState.FILLED:
            # 404: the order left the book when it traded, so the depth we have
            # is already external-only - nothing new is broadcast to wait for.
            return True

        # A 200 cancel is followed by a broadcast; wait for it so the mid
        # reflects strictly external liquidity (no self-referential BBO skew).
        if not await self._await_ingest(target):
            self.log.warning("no depth update after cancelling %s; parking cycle", side)
            return False
        return True

    async def requote_side(self, client: ExchangeClient, side: str, quote: Quote) -> None:
        """Place one side of the quote, sized to the remaining headroom."""
        amount = inventory_headroom(self.inventory, self.max_inventory, side, self.lot)
        if amount <= 0:
            self.log.debug("%s bound reached at %+.4f; skipping", side, self.inventory)
            return

        price = quote.bid if side == "buy" else quote.ask
        order = {
            "side": side,
            "price": price,
            "amount": amount,
            "client_tag": self.client_tag,
        }
        ack = await client.place(order)
        if ack is None:
            return

        if ack.fills:
            shares, notional = fill_totals(ack.fills)
            self._book(side, shares, notional, price)

        if ack.remaining > 0:
            self.active_orders[side] = RestingOrder(
                order_id=ack.order_id,
                secret_token=ack.secret_token,
                amount=ack.remaining,
                price=price,
                side=side,
            )

    async def quote_once(self, client: ExchangeClient) -> Quote | None:
        """One cancel-and-replace cycle; returns the quote that was posted."""
        cancelled_both = await self.cancel_side(client, "buy")
        cancelled_both = await self.cancel_side(client, "sell") and cancelled_both

        if not cancelled_both:
            # Keep exposure flat instead of stacking a second quote per side.
            return None
        if self.mid_price is None:
            self.log.debug("one-sided book; parked with no resting quotes")
            return None

        mid = self.mid_price
        try:
            quote = quote_prices(mid, self.inventory, self.params)
        except ValueError as exc:
            # A reservation price dragged below zero by an extreme inventory
            # has no valid quote; refuse to post a negative price.
            self.log.warning(
                "cannot quote at mid %.4f with inventory %+.4f: %s", mid, self.inventory, exc
            )
            return None
        self.quote = quote

        await self.requote_side(client, "buy", quote)
        await self.requote_side(client, "sell", quote)
        self.publish_telemetry(mid, quote)
        return quote

    async def quoting_loop(self, client: ExchangeClient) -> None:
        """Cancel-and-replace every ``quote_interval`` seconds, forever."""
        while True:
            await asyncio.sleep(self.quote_interval)
            try:
                await self.quote_once(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                self.log.exception("quote cycle failed: %s", exc)

    async def cancel_all(self, client: ExchangeClient) -> int:
        """Best-effort sweep of every resting quote; never raises."""
        orders = list(self.active_orders.items())
        self.active_orders.clear()
        if not orders:
            return 0

        results = await asyncio.gather(
            *(client.cancel(order.as_payload(), order.amount) for _, order in orders),
            return_exceptions=True,
        )
        failed = 0
        for (side, order), result in zip(orders, results, strict=True):
            if isinstance(result, BaseException):
                failed += 1
                self.log.warning("cancel_all %s failed: %s", side, result)
            elif not result.resolved:
                failed += 1
                self.log.warning("cancel_all %s unconfirmed: %s", side, result.reason)
            else:
                self._book(side, result.filled, result.filled * order.price, order.price)
        self.log.info("cancel_all swept %d/%d quotes", len(orders) - failed, len(orders))
        return len(orders) - failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avellaneda-Stoikov market maker for the local limit order book.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exchange-url", default=DEFAULT_EXCHANGE_URL, help="REST base URL")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL, help="depth WebSocket URL")
    parser.add_argument("--gamma", type=float, default=1.0, help="risk aversion (> 0)")
    parser.add_argument("--sigma", type=float, default=0.5, help="volatility per sqrt(second)")
    parser.add_argument("--kappa", type=float, default=1.0, help="book liquidity density (> 0)")
    parser.add_argument("--tau", type=float, default=1.0, help="stationary horizon (seconds)")
    parser.add_argument("--tick", type=float, default=0.01, help="price grid")
    parser.add_argument("--max-inventory", type=float, default=100.0, help="hard |q| bound")
    parser.add_argument("--lot", type=float, default=1.0, help="per-side order size")
    parser.add_argument("--quote-interval", type=float, default=0.5, help="re-quote cadence (s)")
    parser.add_argument(
        "--ingest-timeout", type=float, default=2.0, help="post-cancel book wait (s)"
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="telemetry host")
    parser.add_argument("--udp-port", type=int, default=9999, help="telemetry port")
    parser.add_argument("--client-tag", default=CLIENT_TAG, help="tag sent with every order")
    parser.add_argument(
        "--once",
        action="store_true",
        help="quote a single cycle from a REST snapshot, print it, exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --once: compute and print the quote without sending any order",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def params_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ModelParams:
    try:
        return ModelParams(
            gamma=args.gamma,
            sigma=args.sigma,
            kappa=args.kappa,
            tau=args.tau,
            tick=args.tick,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - parser.error exits


async def run_once(engine: Avellaneda_Stoikov, client: ExchangeClient, dry_run: bool) -> dict:
    """Quote one cycle from a REST snapshot; useful as a connectivity check."""
    try:
        snapshot = await client.get_json("/depth")
    except ExchangeUnavailable as exc:
        raise SystemExit(f"cannot read /depth from the exchange: {exc}") from exc

    mid = engine.ingest(snapshot)
    if mid is None:
        raise SystemExit("the book is one-sided; nothing to quote against")

    quote = quote_prices(mid, engine.inventory, engine.params)
    summary = {
        "mid": mid,
        "bid": quote.bid,
        "ask": quote.ask,
        "reservation": quote.reservation,
        "spread": quote.spread,
        "inventory": engine.inventory,
        "max_inventory": engine.max_inventory,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    await engine.requote_side(client, "buy", quote)
    await engine.requote_side(client, "sell", quote)
    engine.publish_telemetry(mid, quote)
    summary["orders"] = {
        side: order.as_payload() | {"amount": order.amount, "price": order.price}
        for side, order in engine.active_orders.items()
    }
    await engine.cancel_all(client)
    return summary


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

    params = params_from_args(parser, args)
    try:
        engine = Avellaneda_Stoikov(
            params,
            max_inventory=args.max_inventory,
            lot=args.lot,
            quote_interval=args.quote_interval,
            client_tag=args.client_tag,
            udp_target=(args.udp_host, args.udp_port),
            ingest_timeout=args.ingest_timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=5.0, sock_read=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = ExchangeClient(session, base_url=args.exchange_url)
        if args.once:
            summary = await run_once(engine, client, args.dry_run)
            print(json.dumps(summary, indent=2))
            return 0

        tasks = [
            asyncio.create_task(engine.socket_listener(session, args.stream_url)),
            asyncio.create_task(engine.quoting_loop(client)),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.cancel_all(client)
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Console-script entry point (``as-bot`` / ``as-sim``)."""
    try:
        return asyncio.run(main(argv))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
