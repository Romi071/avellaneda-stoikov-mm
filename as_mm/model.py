"""Avellaneda-Stoikov (2008) quoting model.

Pure functions and validated parameters: no I/O, no sockets, no event loop.

The two equations from *High-frequency trading in a limit order book*
(Avellaneda & Stoikov, 2008) implemented here are

    r(s, q, tau) = s - q * gamma * sigma**2 * tau
    Delta(tau)   = gamma * sigma**2 * tau + (2 / gamma) * ln(1 + gamma / kappa)

with ``s`` the mid price, ``q`` the signed inventory, ``gamma`` the risk
aversion, ``sigma`` the price volatility per sqrt(second), ``tau`` the
remaining horizon in seconds and ``kappa`` the order book liquidity density.

Every input is validated up front.  The shipped formula produced ``ZeroDivisionError``
for ``gamma == 0`` and *crossed* quotes (bid above ask) whenever ``gamma / kappa``
was negative enough to drive :func:`optimal_spread` below zero, so a bad
configuration could make the bot trade against itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = [
    "ModelParams",
    "Quote",
    "reservation_price",
    "optimal_spread",
    "quote_prices",
    "inventory_headroom",
    "snap_to_tick",
    "TICK_SIZE",
]

#: Strict price grid of the matching engine (``PriceTime`` engine uses 0.01).
TICK_SIZE = 0.01

_EPS = 1e-12


def _as_finite(name: str, value: object) -> float:
    """Coerce ``value`` to a finite float, naming it in every error."""
    # bool is an int subclass: True/False are configuration mistakes, not prices.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return number


def snap_to_tick(price: float, tick: float = TICK_SIZE) -> float:
    """Round ``price`` to the nearest multiple of ``tick``.

    Rounds half away from zero (Python's ``round`` uses banker's rounding, which
    puts 0.5 cents down for even steps) and tolerates the binary representation
    error that makes ``1.005 / 0.01`` come out as ``100.49999999999999``, so a
    decimal half-cent lands where a trader expects it.
    """
    if tick <= 0:
        raise ValueError(f"tick must be strictly positive, got {tick!r}")
    scaled = price / tick
    steps = math.floor(abs(scaled) + 0.5 + 1e-9)
    if scaled < 0:
        steps = -steps
    return round(steps * tick, 10)


@dataclass(frozen=True)
class ModelParams:
    """Validated Avellaneda-Stoikov parameters.

    ``gamma`` must be strictly positive (it divides the spread), ``kappa``
    strictly positive (it feeds a logarithm), ``sigma`` and ``tau``
    non-negative, ``tick`` strictly positive.
    """

    gamma: float = 1.0
    sigma: float = 0.5
    kappa: float = 1.0
    tau: float = 1.0
    tick: float = TICK_SIZE

    def __post_init__(self) -> None:
        gamma = _as_finite("gamma", self.gamma)
        sigma = _as_finite("sigma", self.sigma)
        kappa = _as_finite("kappa", self.kappa)
        tau = _as_finite("tau", self.tau)
        tick = _as_finite("tick", self.tick)

        if gamma <= 0:
            raise ValueError(f"gamma must be strictly positive, got {gamma!r}")
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa!r}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau!r}")
        if tick <= 0:
            raise ValueError(f"tick must be strictly positive, got {tick!r}")

        # Normalise ints to float so downstream arithmetic is uniform.
        for field, value in (
            ("gamma", gamma),
            ("sigma", sigma),
            ("kappa", kappa),
            ("tau", tau),
            ("tick", tick),
        ):
            object.__setattr__(self, field, value)

    def with_(self, **changes: object) -> ModelParams:
        """Return a copy with ``changes`` applied (validated again)."""
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Quote:
    """A two-sided quote around the reservation price."""

    bid: float
    ask: float
    reservation: float
    spread: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    def as_dict(self) -> dict[str, float]:
        return {
            "bid": self.bid,
            "ask": self.ask,
            "reservation": self.reservation,
            "spread": self.spread,
        }


def reservation_price(mid: float, inventory: float, params: ModelParams) -> float:
    """Inventory-skewed fair price ``s - q * gamma * sigma**2 * tau``.

    Long inventory (``q > 0``) pushes the reservation price below the mid, so
    both quotes shift down and the ask gets more aggressive than the bid.
    """
    mid = _as_finite("mid", mid)
    inventory = _as_finite("inventory", inventory)
    if mid <= 0:
        raise ValueError(f"mid must be strictly positive, got {mid!r}")
    drift = params.gamma * params.sigma**2 * params.tau
    return mid - drift * inventory


def optimal_spread(params: ModelParams) -> float:
    """Stationary optimal spread, always strictly positive for valid params.

    The second term is strictly positive whenever ``gamma > 0`` and
    ``kappa > 0``, and the first is non-negative, so the sum cannot go
    negative - which is exactly what the unvalidated version could do.
    """
    inventory_term = params.gamma * params.sigma**2 * params.tau
    liquidity_term = (2.0 / params.gamma) * math.log1p(params.gamma / params.kappa)
    spread = inventory_term + liquidity_term
    if not spread > 0 or not math.isfinite(spread):
        # Unreachable for validated params; kept as a tripwire.
        raise ValueError(f"degenerate spread {spread!r} for params {params!r}")
    return spread


def quote_prices(mid: float, inventory: float, params: ModelParams) -> Quote:
    """Best bid/ask around the reservation price, snapped to the tick grid.

    The spread is floored at one tick so that snapping can never collapse the
    two sides onto each other, and the result is asserted to be uncrossed.
    """
    reservation = reservation_price(mid, inventory, params)
    spread = max(optimal_spread(params), params.tick)

    bid = snap_to_tick(reservation - spread / 2, params.tick)
    ask = snap_to_tick(reservation + spread / 2, params.tick)

    # Snapping both sides independently can collapse them; widen symmetrically.
    if ask - bid < params.tick - _EPS:
        bid = snap_to_tick(bid - params.tick / 2, params.tick)
        ask = snap_to_tick(ask + params.tick / 2, params.tick)
    if ask - bid < params.tick - _EPS:  # pragma: no cover - defensive
        raise ValueError(f"could not widen quote to one tick: bid={bid!r} ask={ask!r}")
    if bid <= 0:
        raise ValueError(f"bid must be strictly positive, got {bid!r}")

    return Quote(bid=bid, ask=ask, reservation=reservation, spread=ask - bid)


def inventory_headroom(
    inventory: float,
    max_inventory: float,
    side: str,
    lot: float,
) -> float:
    """Largest order allowed on ``side`` without breaching ``max_inventory``.

    Returns ``0.0`` when the bound is already reached, so the caller simply
    skips quoting that side instead of posting an order that pushes the
    position past the documented hard limit.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    inventory = _as_finite("inventory", inventory)
    max_inventory = _as_finite("max_inventory", max_inventory)
    lot = _as_finite("lot", lot)
    if max_inventory <= 0:
        raise ValueError(f"max_inventory must be strictly positive, got {max_inventory!r}")
    if lot <= 0:
        raise ValueError(f"lot must be strictly positive, got {lot!r}")

    headroom = max_inventory - inventory if side == "buy" else max_inventory + inventory
    return max(0.0, min(lot, headroom))
