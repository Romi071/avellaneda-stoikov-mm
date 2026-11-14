"""Real-time L2 cumulative depth chart with the bot's telemetry HUD.

Bridges Qt's event loop with ``asyncio`` through ``qasync`` on a single thread:
depth frames arrive on a WebSocket, telemetry datagrams on UDP, and a 50 ms
``QTimer`` repaints the cumulative curves and the reservation price line.

All rendering maths (cumulative depth, mark-to-market wealth, HUD text) lives in
:mod:`as_mm.telemetry` so it can be tested without a display.  Nothing executes
on import.

Usage::

    python depth_visualizer.py
    python depth_visualizer.py --stream-url ws://localhost:8000/stream --udp-port 9999
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys

import aiohttp
import pyqtgraph as pg
import qasync
from PyQt6 import QtCore, QtGui, QtWidgets

from as_mm.telemetry import decode_datagram, depth_curves, hud_state, mid_price

DEFAULT_STREAM_URL = "ws://localhost:8000/stream"

log = logging.getLogger("as_mm.depth_visualizer")


class Visualizer:
    """``PlotWidget`` plus the depth/telemetry plumbing behind it."""

    REFRESH_MS = 50

    def __init__(
        self,
        *,
        stream_url: str = DEFAULT_STREAM_URL,
        udp_target: tuple[str, int] = ("127.0.0.1", 9999),
        max_backoff: float = 8.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if udp_target[1] <= 0 or udp_target[1] > 65535:
            raise ValueError(f"udp port out of range: {udp_target[1]!r}")
        self.stream_url = stream_url
        self.udp_target = udp_target
        self.max_backoff = max_backoff
        self.log = logger or log

        self.latest_snapshot: dict | None = None
        self.telemetry: dict | None = None
        self.connected = False

        self.win = pg.PlotWidget()
        self.win.setTitle("Real-Time Market Depth Chart", size="18pt")

        label_style = {"font-size": "14pt", "font-weight": "bold"}
        self.win.setLabel("bottom", "Price", units="$", **label_style)
        self.win.setLabel("left", "Cumulative Depth", units="Shares", **label_style)

        tick_font = QtGui.QFont()
        tick_font.setPointSize(11)
        self.win.getAxis("bottom").setTickFont(tick_font)
        self.win.getAxis("left").setTickFont(tick_font)

        self.win.showGrid(x=True, y=True, alpha=0.3)
        pg.setConfigOptions(antialias=True)

        self.bids_curve = self.win.plot(
            pen=pg.mkPen("g", width=2), fillLevel=0, brush=(0, 255, 0, 50)
        )
        self.asks_curve = self.win.plot(
            pen=pg.mkPen("r", width=2), fillLevel=0, brush=(255, 0, 0, 50)
        )

        self.res_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#FFD700", width=2, style=QtCore.Qt.PenStyle.DashLine),
            label="Res Px: ${value:0.2f}",
            labelOpts={
                "position": 0.85,
                "color": "#FFD700",
                "fill": (20, 20, 20, 200),
                "movable": True,
            },
        )
        self.res_line.hide()
        self.win.addItem(self.res_line)
        self.win.show()

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(self.udp_target)
        self.udp_socket.setblocking(False)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.chart_plotter)
        self.timer.start(self.REFRESH_MS)

    # ------------------------------------------------------------------
    def drain_telemetry(self) -> dict | None:
        """Read every queued datagram, keeping the freshest usable one."""
        latest = None
        while True:
            try:
                data, _ = self.udp_socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError as exc:  # pragma: no cover - socket teardown
                self.log.warning("telemetry socket error: %s", exc)
                break
            payload = decode_datagram(data)
            if payload is not None:
                latest = payload
        if latest is not None:
            self.telemetry = latest
        return self.telemetry

    def paint_depth(self) -> bool:
        """Redraw both depth curves; ``False`` when the frame was unusable."""
        if not self.latest_snapshot:
            return False
        try:
            curves = depth_curves(self.latest_snapshot)
        except ValueError as exc:
            self.log.warning("dropping malformed depth frame: %s", exc)
            return False
        self.bids_curve.setData(curves.bid_prices, curves.bid_cumulative)
        self.asks_curve.setData(curves.ask_prices, curves.ask_cumulative)
        return True

    def live_mid(self) -> float | None:
        if not self.latest_snapshot:
            return None
        try:
            return mid_price(self.latest_snapshot)
        except ValueError:
            return None

    def chart_plotter(self) -> None:
        """20 FPS repaint: curves, reservation line and the HUD title."""
        self.paint_depth()
        self.drain_telemetry()
        if not self.telemetry:
            return

        state = hud_state(self.telemetry, self.live_mid())
        if state.res_price is not None:
            self.res_line.setPos(state.res_price)
            self.res_line.show()
        self.win.setTitle(state.title_html)

    # ------------------------------------------------------------------
    async def socket_listener(self, session: aiohttp.ClientSession) -> None:
        """Stream depth forever, reconnecting with exponential backoff."""
        delay = 0.5
        while True:
            try:
                async with session.ws_connect(self.stream_url) as ws:
                    self.log.info("depth stream connected: %s", self.stream_url)
                    self.connected = True
                    delay = 0.5
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self.latest_snapshot = message.json()
                            except (ValueError, TypeError):
                                self.log.warning("dropping non-JSON depth frame")
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
            finally:
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.max_backoff)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            await self.socket_listener(session)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time L2 depth chart with the Avellaneda-Stoikov HUD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL, help="depth WebSocket URL")
    parser.add_argument("--udp-host", default="127.0.0.1", help="telemetry bind host")
    parser.add_argument("--udp-port", type=int, default=9999, help="telemetry bind port")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser


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

    # Qt would otherwise try to interpret our own flags.
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        visualizer = Visualizer(
            stream_url=args.stream_url,
            udp_target=(args.udp_host, args.udp_port),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot start the visualizer: {exc}") from exc

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.run_until_complete(visualizer.run())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
