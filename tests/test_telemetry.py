"""Depth curves, HUD state and the UDP wire format (no Qt required)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from as_mm.telemetry import (
    INV_FLAT,
    INV_LONG,
    INV_SHORT,
    PNL_DOWN,
    PNL_UP,
    best_bid_ask,
    decode_datagram,
    depth_curves,
    encode_telemetry,
    hud_state,
    levels,
    mid_price,
    wealth,
)

SNAPSHOT = {"bids": [[99.5, 2.0], [99.4, 1.0]], "asks": [[100.5, 3.0], [100.6, 1.0]]}


def test_levels_are_parsed_in_order():
    assert levels(SNAPSHOT, "bids") == [(99.5, 2.0), (99.4, 1.0)]
    assert levels(SNAPSHOT, "asks") == [(100.5, 3.0), (100.6, 1.0)]


def test_levels_reject_an_unknown_side_or_missing_key():
    with pytest.raises(ValueError):
        levels(SNAPSHOT, "mid")
    with pytest.raises(ValueError):
        levels({"bids": []}, "asks")
    with pytest.raises(ValueError):
        levels("nope", "bids")


def test_levels_reject_a_malformed_level():
    with pytest.raises(ValueError):
        levels({"bids": [[99.5]], "asks": []}, "bids")
    with pytest.raises(ValueError):
        levels({"bids": [["99.5", 1.0]], "asks": []}, "bids")
    with pytest.raises(ValueError):
        levels({"bids": [[99.5, -1.0]], "asks": []}, "bids")
    with pytest.raises(ValueError):
        levels({"bids": [[0.0, 1.0]], "asks": []}, "bids")
    with pytest.raises(ValueError):
        levels({"bids": [[99.5, 1.0]], "asks": "nope"}, "asks")


def test_best_bid_ask_and_mid():
    assert best_bid_ask(SNAPSHOT) == (99.5, 100.5)
    assert mid_price(SNAPSHOT) == pytest.approx(100.0)


def test_mid_is_none_for_a_one_sided_book():
    assert mid_price({"bids": [[99.5, 1.0]], "asks": []}) is None
    assert mid_price({"bids": [], "asks": [[100.5, 1.0]]}) is None
    assert mid_price({"bids": [], "asks": []}) is None


def test_depth_curves_are_cumulative():
    curves = depth_curves(SNAPSHOT)
    assert list(curves.bid_prices) == [99.5, 99.4]
    assert list(curves.bid_cumulative) == [2.0, 3.0]
    assert list(curves.ask_prices) == [100.5, 100.6]
    assert list(curves.ask_cumulative) == [3.0, 4.0]
    assert not curves.empty


def test_depth_curves_of_one_empty_side():
    curves = depth_curves({"bids": [[99.5, 2.0]], "asks": []})
    assert curves.bid_cumulative.tolist() == [2.0]
    assert curves.ask_prices.size == 0
    assert curves.ask_cumulative.size == 0


def test_depth_curves_of_nothing_are_empty():
    curves = depth_curves({"bids": [], "asks": []})
    assert curves.empty
    assert isinstance(curves.bid_cumulative, np.ndarray)


def test_wealth_prefers_the_live_mid():
    telemetry = {"balance": 10.0, "inventory": 2.0, "wealth": 999.0, "res_price": 98.0}
    assert wealth(telemetry, live_mid=100.0) == pytest.approx(210.0)


def test_wealth_falls_back_to_the_snapshot():
    telemetry = {"balance": 10.0, "inventory": 2.0, "wealth": 205.0}
    assert wealth(telemetry) == pytest.approx(205.0)


def test_wealth_is_none_when_nothing_is_known():
    assert wealth({}) is None
    assert wealth(None) is None
    assert wealth({"res_price": 100.0}) is None


def test_hud_state_formats_a_profit():
    state = hud_state({"balance": 5.0, "inventory": 1.5, "res_price": 99.25, "wealth": 12.5})
    assert state.pnl == pytest.approx(12.5)
    assert state.pnl_text == "+$12.50"
    assert state.pnl_color == PNL_UP
    assert state.inventory_text == "+1.50"
    assert state.inventory_color == INV_LONG
    assert state.cash_text == "$5.00"
    assert state.res_price_text == "$99.25"
    assert "PnL:" in state.title_html and "Res Px:" in state.title_html


def test_hud_state_formats_a_loss_and_a_short():
    state = hud_state({"balance": -12.0, "inventory": -3.0, "res_price": 101.0, "wealth": -7.25})
    assert state.pnl_text == "-$7.25"
    assert state.pnl_color == PNL_DOWN
    assert state.inventory_text == "-3.00"
    assert state.inventory_color == INV_SHORT


def test_hud_state_handles_a_flat_book_and_missing_fields():
    flat = hud_state({"balance": 0.0, "inventory": 0.0, "res_price": 100.0, "wealth": 0.0})
    assert flat.inventory_text == "0.00"
    assert flat.inventory_color == INV_FLAT
    assert flat.pnl_text == "+$0.00"

    empty = hud_state({})
    assert empty.pnl is None
    assert empty.pnl_text == "n/a"
    assert empty.cash_text == "n/a"
    assert "n/a" in empty.title_html


def test_hud_state_uses_the_live_mid_for_mark_to_market():
    telemetry = {"balance": 0.0, "inventory": 10.0, "res_price": 95.0, "wealth": 0.0}
    state = hud_state(telemetry, live_mid=101.0)
    assert state.pnl == pytest.approx(1010.0)


def test_encode_decode_round_trip():
    raw = encode_telemetry(1.5, -2.0, 97.5, 3.25)
    payload = decode_datagram(raw)
    assert payload == {"balance": 1.5, "inventory": -2.0, "res_price": 97.5, "wealth": 3.25}
    assert json.loads(raw) == payload


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[1, 2, 3]",
        b"{}",
        b'{"unrelated": 1}',
        b"\xff\xfe",
        "text",
        None,
    ],
)
def test_decode_returns_none_for_unusable_datagrams(raw):
    assert decode_datagram(raw) is None
