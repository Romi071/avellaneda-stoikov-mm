"""Market-depth and telemetry helpers shared by the bot, its tests and the GUI.

Everything here is Qt-free so the rendering maths can be tested headlessly.
The visualizer only turns these values into ``pyqtgraph`` items.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DepthCurves",
    "HudState",
    "levels",
    "best_bid_ask",
    "mid_price",
    "depth_curves",
    "wealth",
    "hud_state",
    "encode_telemetry",
    "decode_datagram",
    "TELEMETRY_KEYS",
]

#: Keys the bot broadcasts; the visualizer tolerates missing ones.
TELEMETRY_KEYS = ("balance", "inventory", "res_price", "wealth")

PNL_UP = "#00FF66"
PNL_DOWN = "#FF3355"
INV_LONG = "#00D4FF"
INV_SHORT = "#FFAA00"
INV_FLAT = "#FFFFFF"
RES_COLOR = "#FFD700"
HUD_TEXT = "#888888"


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {number!r}")
    return number


def levels(snapshot: object, side: str) -> list[tuple[float, float]]:
    """Validated ``[(price, volume), ...]`` for ``side`` ('bids' or 'asks').

    The engine broadcasts ``{"bids": [[price, volume], ...], "asks": [...]}``
    with bids best-first (descending) and asks best-first (ascending).
    """
    if side not in ("bids", "asks"):
        raise ValueError(f"side must be 'bids' or 'asks', got {side!r}")
    if not isinstance(snapshot, dict):
        raise ValueError(f"snapshot must be an object, got {type(snapshot).__name__}")
    if side not in snapshot:
        raise ValueError(f"snapshot is missing {side!r}")
    raw = snapshot[side]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{side} must be a list, got {type(raw).__name__}")

    parsed: list[tuple[float, float]] = []
    for index, level in enumerate(raw):
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise ValueError(f"{side}[{index}] must be [price, volume]")
        price = _number(level[0], f"{side}[{index}] price")
        volume = _number(level[1], f"{side}[{index}] volume")
        if price <= 0:
            raise ValueError(f"{side}[{index}] price must be positive, got {price!r}")
        if volume < 0:
            raise ValueError(f"{side}[{index}] volume must be non-negative, got {volume!r}")
        parsed.append((price, volume))
    return parsed


def best_bid_ask(snapshot: object) -> tuple[float | None, float | None]:
    """Best bid and best ask, ``None`` for a side with no resting liquidity."""
    bids = levels(snapshot, "bids")
    asks = levels(snapshot, "asks")
    return (bids[0][0] if bids else None, asks[0][0] if asks else None)


def mid_price(snapshot: object) -> float | None:
    """Microprice-free mid, or ``None`` whenever the book is one-sided.

    ``None`` (not ``0.0``) is the sentinel: a zero price is not a price, and
    the quoting loop must park rather than quote around it.
    """
    bid, ask = best_bid_ask(snapshot)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2


@dataclass(frozen=True)
class DepthCurves:
    """Cumulative depth per side, ready for ``PlotDataItem.setData``."""

    bid_prices: np.ndarray
    bid_cumulative: np.ndarray
    ask_prices: np.ndarray
    ask_cumulative: np.ndarray

    @property
    def empty(self) -> bool:
        return self.bid_prices.size == 0 and self.ask_prices.size == 0


def depth_curves(snapshot: object) -> DepthCurves:
    """Vectorised cumulative depth for both sides of ``snapshot``."""
    bids = levels(snapshot, "bids")
    asks = levels(snapshot, "asks")

    if bids:
        bid_arr = np.asarray(bids, dtype=float)
        bid_prices = bid_arr[:, 0]
        bid_cumulative = np.cumsum(bid_arr[:, 1])
    else:
        bid_prices = np.empty(0, dtype=float)
        bid_cumulative = np.empty(0, dtype=float)

    if asks:
        ask_arr = np.asarray(asks, dtype=float)
        ask_prices = ask_arr[:, 0]
        ask_cumulative = np.cumsum(ask_arr[:, 1])
    else:
        ask_prices = np.empty(0, dtype=float)
        ask_cumulative = np.empty(0, dtype=float)

    return DepthCurves(bid_prices, bid_cumulative, ask_prices, ask_cumulative)


def wealth(telemetry: object, live_mid: float | None = None) -> float | None:
    """Mark-to-market wealth ``X + q * S``.

    Prefers re-valuing the reported inventory against a live mid (the GUI does
    this between quoting cycles) and falls back to the bot's own snapshot.
    Returns ``None`` when neither is available instead of raising.
    """
    if not isinstance(telemetry, dict):
        return None
    balance = telemetry.get("balance")
    inventory = telemetry.get("inventory")
    if (
        live_mid is not None
        and isinstance(balance, (int, float))
        and not isinstance(balance, bool)
        and isinstance(inventory, (int, float))
        and not isinstance(inventory, bool)
    ):
        return float(balance) + float(inventory) * float(live_mid)
    snapshot = telemetry.get("wealth")
    if isinstance(snapshot, (int, float)) and not isinstance(snapshot, bool):
        return float(snapshot)
    return None


@dataclass(frozen=True)
class HudState:
    """Everything the on-chart scoreboard needs, already formatted."""

    pnl: float | None
    cash: float | None
    inventory: float | None
    res_price: float | None
    pnl_text: str
    pnl_color: str
    inventory_text: str
    inventory_color: str
    cash_text: str
    res_price_text: str
    title_html: str


def _money(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "n/a"


def hud_state(telemetry: object, live_mid: float | None = None) -> HudState:
    """Build the monospace HTML title bar; tolerant of partial telemetry."""
    data = telemetry if isinstance(telemetry, dict) else {}
    pnl = wealth(data, live_mid)

    cash_raw = data.get("balance")
    cash = (
        float(cash_raw)
        if isinstance(cash_raw, (int, float)) and not isinstance(cash_raw, bool)
        else None
    )
    inv_raw = data.get("inventory")
    inventory = (
        float(inv_raw)
        if isinstance(inv_raw, (int, float)) and not isinstance(inv_raw, bool)
        else None
    )
    res_raw = data.get("res_price")
    res_price = (
        float(res_raw)
        if isinstance(res_raw, (int, float)) and not isinstance(res_raw, bool)
        else None
    )

    if pnl is None:
        pnl_text = "n/a"
        pnl_color = HUD_TEXT
    else:
        pnl_text = f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}"
        pnl_color = PNL_UP if pnl >= 0 else PNL_DOWN

    if inventory is None:
        inventory_text = "n/a"
        inventory_color = INV_FLAT
    else:
        inventory_text = f"{'+' if inventory > 0 else ''}{inventory:.2f}"
        inventory_color = INV_LONG if inventory > 0 else (INV_SHORT if inventory < 0 else INV_FLAT)

    title_html = (
        "<span style='font-family: Consolas, monospace; font-size: 14pt; "
        f"color: {HUD_TEXT};'>"
        f"PnL: <b style='color: {pnl_color};'>{pnl_text}</b> &nbsp;|&nbsp; "
        f"Cash: <b style='color: #FFFFFF;'>{_money(cash)}</b> &nbsp;|&nbsp; "
        f"Inv: <b style='color: {inventory_color};'>{inventory_text}</b> &nbsp;|&nbsp; "
        f"Res Px: <b style='color: {RES_COLOR};'>{_money(res_price)}</b>"
        "</span>"
    )
    return HudState(
        pnl=pnl,
        cash=cash,
        inventory=inventory,
        res_price=res_price,
        pnl_text=pnl_text,
        pnl_color=pnl_color,
        inventory_text=inventory_text,
        inventory_color=inventory_color,
        cash_text=_money(cash),
        res_price_text=_money(res_price),
        title_html=title_html,
    )


def encode_telemetry(
    balance: float,
    inventory: float,
    res_price: float,
    wealth_value: float,
) -> bytes:
    """Serialise one UDP datagram; the key set is the visualizer contract."""
    payload = {
        "balance": float(balance),
        "inventory": float(inventory),
        "res_price": float(res_price),
        "wealth": float(wealth_value),
    }
    return json.dumps(payload).encode("utf-8")


def decode_datagram(raw: bytes) -> dict | None:
    """Parse a telemetry datagram, returning ``None`` for anything unusable.

    A stray datagram (another process on 9999, a truncated packet) must never
    take down the 20 FPS render loop, so every failure mode collapses to
    ``None`` rather than an exception.
    """
    if not isinstance(raw, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not any(key in payload for key in TELEMETRY_KEYS):
        return None
    return payload
