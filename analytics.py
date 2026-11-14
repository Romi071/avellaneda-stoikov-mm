"""Cold-path attribution over the matching engine's ``ledger.db``.

Decomposes the tagged participant's PnL into spread capture and inventory PnL,
and writes the equity curve.  See :mod:`as_mm.analytics` for the derivation and
the identity that is asserted below.

Usage::

    python analytics.py --db "Limit Order Book/ledger.db"
    python analytics.py --db ledger.db --json
    python analytics.py --db ledger.db --curve equity.csv --plot equity.png

The ledger is opened read-only: a wrong path fails loudly instead of silently
creating an empty database (which is what the previous relative hard-coded path
did).
"""

from __future__ import annotations

import argparse
import logging

from as_mm.analytics import (
    AnalysisError,
    Attribution,
    attribute,
    dumps,
    format_report,
    latest_price,
    load_trades,
    open_ledger,
    write_curve_csv,
)

log = logging.getLogger("as_mm.analytics")

DEFAULT_TAG = "as_mm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribution and equity curve for a tagged market maker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="path to the engine's ledger.db")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="client tag to isolate")
    parser.add_argument(
        "--mark", type=float, default=None, help="terminal mark price (default: last print)"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--curve", default=None, help="write the per-fill equity curve to CSV")
    parser.add_argument("--plot", default=None, help="write an equity-curve PNG (needs matplotlib)")
    parser.add_argument("--limit", type=int, default=None, help="analyse only the last N fills")
    parser.add_argument("--log-level", default="WARNING", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def plot_curve(path: str, curve: list[dict], tag: str) -> None:
    """Render the equity curve; matplotlib stays an optional extra."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AnalysisError(
            "matplotlib is not installed; drop --plot or pip install matplotlib"
        ) from exc

    equity = [row["equity"] for row in curve]
    plt.figure(figsize=(10, 4.5))
    plt.plot(range(len(equity)), equity, linewidth=1.4)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.title(f"Marked equity - {tag}")
    plt.xlabel("fill")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> Attribution:
    if args.limit is not None and args.limit < 1:
        raise AnalysisError(f"--limit must be at least 1, got {args.limit}")

    connection = open_ledger(args.db)
    try:
        trades = load_trades(connection, args.tag)
        if args.limit is not None:
            trades = trades[-args.limit :]
        mark = args.mark if args.mark is not None else latest_price(connection)
    finally:
        connection.close()

    if mark <= 0:
        raise AnalysisError(f"--mark must be a positive price, got {mark!r}")

    result = attribute(trades, mark, tag=args.tag)
    residual = abs(result.equity - result.total)
    if residual > 1e-6:
        raise AnalysisError(
            f"attribution does not reconcile: equity {result.equity:.10f} vs "
            f"components {result.total:.10f} (residual {residual:.3e})"
        )

    if args.curve:
        write_curve_csv(args.curve, result.curve)
        log.info("wrote %d rows to %s", len(result.curve), args.curve)
    if args.plot:
        plot_curve(args.plot, result.curve, args.tag)
        log.info("wrote %s", args.plot)

    return result


def main(argv: list[str] | None = None) -> int:
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
        result = run(args)
    except AnalysisError as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.json:
        print(dumps(result, curve=bool(args.curve or args.plot)), end="\n")
    else:
        print(format_report(result))
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Console-script entry point (``as-analytics``)."""
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
