"""Execution-layer primitives: response interpretation and fill accounting.

The matching engine (Romi071/limit-order-book, ``server.py`` + ``order_book.py``)
answers with three shapes that matter to a market maker:

``POST /order``   200 ``{orderId, secret_token, client_tag, active, amount, fills}``
    ``amount`` is the *remaining* (unfilled) size after immediate matching and
    ``fills`` is the list of execution receipts.
``POST /cancel``  200 ``{success: true,  cancelled_amount: X}``
                  404 ``{success: false, cancelled_amount: 0}``  (already filled)
                  401 ``{success: false, detail: ...}``          (**no** ``cancelled_amount``)
                  422 for a malformed body.

The shipped quoting loop read ``cancel_logs['cancelled_amount']`` inline and
called ``raise_for_status()`` for anything that was not 200/404, so a single
401/5xx propagated out of the loop and terminated the whole bot.  Worse, an
error that left the fill quantity *unknown* was indistinguishable from a real
zero.  Everything here is a pure function of ``(status, payload, resting)`` so
each branch is testable without a server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "OrderState",
    "CancelResult",
    "OrderAck",
    "PayloadError",
    "interpret_cancel",
    "parse_order_ack",
    "fill_totals",
    "apply_fill",
    "book_execution",
    "is_retryable_status",
]


class PayloadError(ValueError):
    """The exchange answered with a body the client cannot trust."""


class OrderState(Enum):
    """What a cancel attempt tells us about a resting order."""

    #: The engine acknowledged the cancel (200). ``filled`` is now known.
    GONE = "gone"
    #: The engine no longer knows the order (404) - it was fully executed.
    FILLED = "filled"
    #: Authorization/transport/engine problem: the order may still be live.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CancelResult:
    """Outcome of one ``POST /cancel`` for an order resting ``resting`` shares."""

    state: OrderState
    cancelled: float
    filled: float
    reason: str = ""

    @property
    def resolved(self) -> bool:
        """True when the exchange state is known and fills can be booked."""
        return self.state is not OrderState.UNKNOWN

    @classmethod
    def in_doubt(cls, reason: str) -> CancelResult:
        """No usable answer: the order may still be live, book nothing."""
        return cls(OrderState.UNKNOWN, 0.0, 0.0, reason)


@dataclass(frozen=True)
class OrderAck:
    """A validated ``POST /order`` response."""

    order_id: str
    secret_token: str
    remaining: float
    fills: tuple[dict, ...]
    active: bool

    @property
    def is_resting(self) -> bool:
        return self.remaining > 0


def _as_amount(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise PayloadError(f"{name} must be finite, got {number!r}")
    return number


def is_retryable_status(status: int) -> bool:
    """5xx and 429 are worth another attempt; 4xx using our credentials are not."""
    return status == 429 or 500 <= status < 600


def interpret_cancel(status: int, payload: object, resting_amount: float) -> CancelResult:
    """Map a cancel response onto the shares that actually traded.

    ``resting_amount`` is the size that was live at cancel time, so the fill is
    ``resting - cancelled``.  Nothing is booked unless the exchange state is
    known: an ``UNKNOWN`` result leaves the order tracked for the next cycle
    instead of silently corrupting inventory and cash.
    """
    resting = _as_amount("resting_amount", resting_amount)
    if resting < 0:
        raise PayloadError(f"resting_amount must be non-negative, got {resting!r}")
    body = payload if isinstance(payload, dict) else {}

    if status == 200:
        if "cancelled_amount" not in body:
            return CancelResult(OrderState.UNKNOWN, 0.0, 0.0, "200 without cancelled_amount")
        try:
            cancelled = _as_amount("cancelled_amount", body["cancelled_amount"])
        except PayloadError as exc:
            return CancelResult(OrderState.UNKNOWN, 0.0, 0.0, str(exc))
        if cancelled < -1e-9 or cancelled > resting + 1e-9:
            # A cancel can never return more than was resting; clamp and say so.
            clamped = min(max(cancelled, 0.0), resting)
            return CancelResult(
                OrderState.GONE,
                clamped,
                resting - clamped,
                f"cancelled_amount {cancelled!r} outside [0, {resting!r}]",
            )
        return CancelResult(OrderState.GONE, cancelled, resting - cancelled)

    if status == 404:
        # "The order is already filled (or it doesn't exist)": nothing to cancel.
        return CancelResult(OrderState.FILLED, 0.0, resting, "404: order no longer resting")

    if status in (401, 403):
        return CancelResult(
            OrderState.UNKNOWN, 0.0, 0.0, f"{status}: unauthorized cancel, order state unknown"
        )
    if is_retryable_status(status):
        return CancelResult(OrderState.UNKNOWN, 0.0, 0.0, f"{status}: transient engine error")
    return CancelResult(OrderState.UNKNOWN, 0.0, 0.0, f"{status}: unexpected cancel response")


def parse_order_ack(payload: object) -> OrderAck:
    """Validate a ``POST /order`` response or raise :class:`PayloadError`."""
    if not isinstance(payload, dict):
        raise PayloadError(f"order response must be an object, got {type(payload).__name__}")
    for key in ("orderId", "secret_token", "amount"):
        if key not in payload:
            raise PayloadError(f"order response is missing {key!r}")

    remaining = _as_amount("amount", payload["amount"])
    if remaining < 0:
        raise PayloadError(f"remaining amount must be non-negative, got {remaining!r}")

    fills = payload.get("fills", [])
    if not isinstance(fills, list):
        raise PayloadError(f"fills must be a list, got {type(fills).__name__}")

    order_id = payload["orderId"]
    token = payload["secret_token"]
    if not isinstance(order_id, str) or not order_id:
        raise PayloadError("orderId must be a non-empty string")
    if not isinstance(token, str) or not token:
        raise PayloadError("secret_token must be a non-empty string")

    return OrderAck(
        order_id=order_id,
        secret_token=token,
        remaining=remaining,
        fills=tuple(f for f in fills if isinstance(f, dict)),
        active=bool(payload.get("active", remaining > 0)),
    )


def fill_totals(fills: object) -> tuple[float, float]:
    """Return ``(shares, notional)`` executed by an order, in price order.

    A taker sweep can walk several price levels (price improvement), so the
    cash impact has to be summed per receipt rather than applied at the quoted
    price.  Malformed receipts are rejected instead of silently skipped, which
    would understate the cash spent.
    """
    if not isinstance(fills, (list, tuple)):
        raise PayloadError(f"fills must be a list, got {type(fills).__name__}")
    shares = 0.0
    notional = 0.0
    for index, fill in enumerate(fills):
        if not isinstance(fill, dict):
            raise PayloadError(f"fill {index} must be an object")
        for key in ("price", "amount"):
            if key not in fill:
                raise PayloadError(f"fill {index} is missing {key!r}")
        price = _as_amount(f"fill {index} price", fill["price"])
        amount = _as_amount(f"fill {index} amount", fill["amount"])
        if price <= 0 or amount < 0:
            raise PayloadError(f"fill {index} has an invalid price/amount: {fill!r}")
        shares += amount
        notional += price * amount
    return shares, notional


def book_execution(
    inventory: float, balance: float, side: str, shares: float, notional: float
) -> tuple[float, float]:
    """Book ``shares`` executed for ``notional`` cash and return the new state.

    A single taker order can sweep several price levels, so cash has to be
    booked from the summed notional rather than at one quoted price.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    inventory = _as_amount("inventory", inventory)
    balance = _as_amount("balance", balance)
    shares = _as_amount("shares", shares)
    notional = _as_amount("notional", notional)
    if shares < 0:
        raise ValueError(f"shares must be non-negative, got {shares!r}")
    if notional < 0:
        raise ValueError(f"notional must be non-negative, got {notional!r}")

    if side == "buy":
        return inventory + shares, balance - notional
    return inventory - shares, balance + notional


def apply_fill(
    inventory: float, balance: float, side: str, price: float, amount: float
) -> tuple[float, float]:
    """Book one execution at a single price and return the new ``(inventory, balance)``.

    Buying adds inventory and spends cash, selling does the reverse.  This is
    the single place where the sign convention lives.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    price = _as_amount("price", price)
    amount = _as_amount("amount", amount)
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount!r}")
    if price <= 0:
        raise ValueError(f"price must be strictly positive, got {price!r}")
    return book_execution(inventory, balance, side, amount, price * amount)
