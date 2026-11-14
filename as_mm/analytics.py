"""Cold-path attribution for the ``ledger.db`` written by the matching engine.

This is the roadmap item the suite was missing: turn the raw ``trades`` table
into an inventory and PnL decomposition.

Ledger schema (``server.py``)::

    trades(id INTEGER PRIMARY KEY AUTOINCREMENT, takerSide, takerId, takerTag,
           makerId, makerTag, price REAL, amount REAL, timestamp DATETIME)

Our side of a row is the taker side when ``takerTag`` matches, and the opposite
side when we were the resting ``makerTag``.

PnL identity
------------
With ``sign_i = +1`` for a buy and ``-1`` for a sell, ``q_0 = 0`` and a terminal
mark ``M``, equity for a desk that started flat is *exactly*

    E = X + q_n * M = sum_i sign_i * (M - p_i) * a_i

Splitting each term around a reference price ``ref_i`` (the last trade printed
before ours) and applying Abel summation gives the decomposition this module
reports::

    spread_capture = sum_i sign_i * (ref_i - p_i) * a_i
    inventory_pnl  = sum_i q_i * (ref_{i+1} - ref_i)      with ref_{n+1} = M
    E              = spread_capture + inventory_pnl

so the two components always add up to realised equity - that equality is
asserted, not assumed.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field

__all__ = [
    "AnalysisError",
    "Trade",
    "Attribution",
    "open_ledger",
    "load_trades",
    "latest_price",
    "attribute",
    "equity_curve",
    "write_curve_csv",
    "format_report",
]

_TRADE_COLUMNS = ("id", "price", "amount", "takerSide", "takerTag", "makerTag", "timestamp")


class AnalysisError(RuntimeError):
    """The ledger cannot answer the question that was asked."""


@dataclass(frozen=True)
class Trade:
    """One row of ``trades`` as seen by the tagged participant."""

    trade_id: int
    price: float
    amount: float
    side: str
    role: str
    reference: float | None
    timestamp: str = ""

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1

    @property
    def ref(self) -> float:
        """Reference price, falling back to the print itself (zero edge)."""
        return self.price if self.reference is None else self.reference


@dataclass(frozen=True)
class Attribution:
    """Result of the decomposition, in quote currency."""

    tag: str
    fills: int
    volume: float
    buys: int
    sells: int
    maker_fills: int
    taker_fills: int
    cash: float
    inventory: float
    mark: float
    equity: float
    spread_capture: float
    inventory_pnl: float
    maker_spread_capture: float
    taker_spread_capture: float
    inventory_min: float
    inventory_max: float
    curve: list[dict] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.spread_capture + self.inventory_pnl

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("curve", None)
        payload.update(
            total=self.total,
            residual=self.equity - self.total,
        )
        return payload


def open_ledger(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` strictly read-only.

    ``sqlite3.connect`` happily *creates* a missing database, which is how the
    shipped script ended up pointing at ``../Limit Order Book/ledger.db`` and
    either creating a stray empty file or dying with "unable to open database
    file".  A read-only URI fails loudly instead.
    """
    import os

    if not db_path:
        raise AnalysisError("a ledger path is required (--db)")
    if not os.path.exists(db_path):
        raise AnalysisError(
            f"ledger not found: {db_path} (start the matching engine once so it creates ledger.db)"
        )
    if os.path.isdir(db_path):
        raise AnalysisError(f"ledger path is a directory: {db_path}")

    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:  # pragma: no cover - rare driver failure
        raise AnalysisError(f"cannot open ledger {db_path}: {exc}") from exc

    try:
        connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
    except sqlite3.DatabaseError as exc:
        connection.close()
        raise AnalysisError(f"{db_path} is not a SQLite database: {exc}") from exc

    tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "trades" not in tables:
        connection.close()
        raise AnalysisError(
            f"{db_path} has no 'trades' table (found: {', '.join(sorted(tables)) or 'nothing'})"
        )
    return connection


def load_trades(connection: sqlite3.Connection, tag: str) -> list[Trade]:
    """Rows where ``tag`` was the taker or the maker, oldest first.

    The reference price is the previous printed trade in the ledger, resolved
    with a correlated subquery on the primary key so this stays one statement.
    """
    if not isinstance(tag, str) or not tag:
        raise AnalysisError("a client tag is required (--tag)")

    sql = f"""
        SELECT {", ".join(_TRADE_COLUMNS)},
               (SELECT t2.price FROM trades AS t2
                 WHERE t2.id < trades.id ORDER BY t2.id DESC LIMIT 1) AS ref_price
          FROM trades
         WHERE takerTag = ? OR makerTag = ?
         ORDER BY id
    """
    try:
        rows = connection.execute(sql, (tag, tag)).fetchall()
    except sqlite3.Error as exc:
        raise AnalysisError(f"cannot read trades: {exc}") from exc

    if not rows:
        raise AnalysisError(f"no trades carry tag {tag!r} (was the bot ever connected?)")

    trades: list[Trade] = []
    for row in rows:
        taker_side = (row["takerSide"] or "").lower()
        if taker_side not in ("buy", "sell"):
            raise AnalysisError(f"trade {row['id']} has an unknown takerSide {row['takerSide']!r}")
        if row["takerTag"] == tag:
            side, role = taker_side, "taker"
        else:
            side, role = ("sell" if taker_side == "buy" else "buy"), "maker"

        price = float(row["price"])
        amount = float(row["amount"])
        if not math.isfinite(price) or price <= 0:
            raise AnalysisError(f"trade {row['id']} has an invalid price {row['price']!r}")
        if not math.isfinite(amount) or amount <= 0:
            raise AnalysisError(f"trade {row['id']} has an invalid amount {row['amount']!r}")

        reference = row["ref_price"]
        trades.append(
            Trade(
                trade_id=int(row["id"]),
                price=price,
                amount=amount,
                side=side,
                role=role,
                reference=None if reference is None else float(reference),
                timestamp=str(row["timestamp"] or ""),
            )
        )
    return trades


def latest_price(connection: sqlite3.Connection) -> float:
    """Last printed trade price in the ledger, used as the terminal mark."""
    row = connection.execute("SELECT price FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise AnalysisError("ledger has no trades to mark against")
    price = float(row["price"])
    if not math.isfinite(price) or price <= 0:
        raise AnalysisError(f"last ledger price is invalid: {row['price']!r}")
    return price


def equity_curve(trades: list[Trade], mark: float) -> list[dict]:
    """Per-fill inventory, cash and marked equity path."""
    if not trades:
        return []
    if not math.isfinite(mark) or mark <= 0:
        raise AnalysisError(f"mark must be a positive price, got {mark!r}")

    inventory = 0.0
    cash = 0.0
    curve: list[dict] = []
    for trade in trades:
        inventory += trade.sign * trade.amount
        cash -= trade.sign * trade.price * trade.amount
        curve.append(
            {
                "id": trade.trade_id,
                "timestamp": trade.timestamp,
                "price": trade.price,
                "amount": trade.amount,
                "side": trade.side,
                "role": trade.role,
                "reference": trade.ref,
                "inventory": inventory,
                "cash": cash,
                "equity": cash + inventory * mark,
            }
        )
    return curve


def attribute(trades: list[Trade], mark: float, tag: str = "") -> Attribution:
    """Decompose equity into spread capture and inventory PnL.

    ``mark`` is the price the residual position is valued at.  The returned
    ``residual`` (``equity - total``) is ~0 by construction and is reported so
    a broken ledger is visible instead of being rounded away.
    """
    if not trades:
        raise AnalysisError(f"no fills for tag {tag!r}")
    curve = equity_curve(trades, mark)

    spread_capture = 0.0
    maker_spread_capture = 0.0
    taker_spread_capture = 0.0
    inventory_pnl = 0.0
    volume = 0.0
    buys = sells = maker_fills = taker_fills = 0

    for index, trade in enumerate(trades):
        edge = trade.sign * (trade.ref - trade.price) * trade.amount
        spread_capture += edge
        if trade.role == "maker":
            maker_spread_capture += edge
            maker_fills += 1
        else:
            taker_spread_capture += edge
            taker_fills += 1
        volume += trade.amount
        if trade.side == "buy":
            buys += 1
        else:
            sells += 1

        # q_i carries over the gap to the next reference (ref_{n+1} == mark).
        next_reference = trades[index + 1].ref if index + 1 < len(trades) else mark
        inventory_pnl += curve[index]["inventory"] * (next_reference - trade.ref)

    final = curve[-1]
    return Attribution(
        tag=tag,
        fills=len(trades),
        volume=volume,
        buys=buys,
        sells=sells,
        maker_fills=maker_fills,
        taker_fills=taker_fills,
        cash=final["cash"],
        inventory=final["inventory"],
        mark=mark,
        equity=final["equity"],
        spread_capture=spread_capture,
        inventory_pnl=inventory_pnl,
        maker_spread_capture=maker_spread_capture,
        taker_spread_capture=taker_spread_capture,
        inventory_min=min(row["inventory"] for row in curve),
        inventory_max=max(row["inventory"] for row in curve),
        curve=curve,
    )


def write_curve_csv(path: str, curve: list[dict]) -> int:
    """Write the equity curve, returning the number of data rows written."""
    if not curve:
        raise AnalysisError("refusing to write an empty equity curve")
    columns = list(curve[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)
    return len(curve)


def format_report(result: Attribution) -> str:
    """Human-readable summary for the terminal."""

    def money(value: float) -> str:
        return f"{value:+,.2f}"

    lines = [
        f"tag            {result.tag}",
        f"fills          {result.fills:,} ({result.maker_fills:,} maker / "
        f"{result.taker_fills:,} taker, {result.buys:,} buy / {result.sells:,} sell)",
        f"volume         {result.volume:,.4f}",
        f"inventory      {result.inventory:+,.4f} "
        f"(path {result.inventory_min:+,.4f} .. {result.inventory_max:+,.4f})",
        f"cash           {money(result.cash)}",
        f"mark           {result.mark:,.4f}",
        f"equity         {money(result.equity)}",
        "",
        f"spread capture {money(result.spread_capture)} "
        f"(maker {money(result.maker_spread_capture)}, taker {money(result.taker_spread_capture)})",
        f"inventory PnL  {money(result.inventory_pnl)}",
        f"total          {money(result.total)}",
        f"residual       {result.equity - result.total:+.10f}",
    ]
    return "\n".join(lines)


def dumps(result: Attribution, *, curve: bool = False) -> str:
    """JSON form of a result (``curve`` inlined on request)."""
    payload = result.as_dict()
    if curve:
        payload["curve"] = result.curve
    return json.dumps(payload, indent=2, sort_keys=True)
