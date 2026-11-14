"""Avellaneda-Stoikov market making toolkit.

Modules
-------
``as_mm.model``      pure quoting maths and validated parameters
``as_mm.execution``  exchange response interpretation and fill accounting
``as_mm.client``     resilient REST client with retries
``as_mm.telemetry``  depth curves, HUD state and the UDP wire format
``as_mm.analytics``  cold-path attribution over ``ledger.db``
"""

from __future__ import annotations

from .model import (
    ModelParams,
    Quote,
    inventory_headroom,
    optimal_spread,
    quote_prices,
    reservation_price,
)

__all__ = [
    "ModelParams",
    "Quote",
    "reservation_price",
    "optimal_spread",
    "quote_prices",
    "inventory_headroom",
    "__version__",
]

__version__ = "1.1.0"
