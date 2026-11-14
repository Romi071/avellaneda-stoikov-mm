"""Quoting maths: the documented equations, the validation, and the tick grid."""

from __future__ import annotations

import dataclasses
import math

import pytest

from as_mm.model import (
    ModelParams,
    inventory_headroom,
    optimal_spread,
    quote_prices,
    reservation_price,
    snap_to_tick,
)


def test_documented_defaults_match_the_readme():
    params = ModelParams()
    assert params.gamma == 1.0
    assert params.sigma == 0.5
    assert params.kappa == 1.0
    assert params.tau == 1.0


def test_reservation_price_is_the_mid_for_flat_inventory():
    params = ModelParams()
    assert reservation_price(100.0, 0.0, params) == pytest.approx(100.0)


def test_reservation_price_skews_down_with_long_inventory():
    params = ModelParams()
    flat = reservation_price(100.0, 0.0, params)
    long = reservation_price(100.0, 10.0, params)
    short = reservation_price(100.0, -10.0, params)
    # c = gamma * sigma^2 * tau = 0.25, so 10 shares skew by 2.5
    assert long == pytest.approx(97.5)
    assert short == pytest.approx(102.5)
    assert long < flat < short


def test_optimal_spread_matches_the_closed_form():
    params = ModelParams(gamma=1.0, sigma=0.5, tau=1.0, kappa=1.0)
    expected = 0.25 + 2.0 * math.log(2.0)
    assert optimal_spread(params) == pytest.approx(expected)


def test_optimal_spread_is_independent_of_mid_and_inventory():
    params = ModelParams()
    first = quote_prices(100.0, 0.0, params)
    second = quote_prices(250.0, 40.0, params)
    assert first.spread == pytest.approx(second.spread, abs=1e-9)


def test_quote_prices_land_on_the_tick_grid_and_straddle_the_reservation():
    params = ModelParams()
    quote = quote_prices(100.0, 0.0, params)
    assert quote.bid == pytest.approx(99.18)
    assert quote.ask == pytest.approx(100.82)
    assert quote.bid < quote.reservation < quote.ask
    for price in (quote.bid, quote.ask):
        assert price == pytest.approx(snap_to_tick(price, params.tick), abs=1e-9)


def test_quote_prices_are_never_crossed_across_the_parameter_space():
    checked = 0
    for gamma in (0.05, 0.5, 1.0, 7.0, 60.0):
        for kappa in (0.01, 0.5, 1.0, 40.0):
            for sigma in (0.0, 0.5, 2.0):
                for tau in (0.0, 1.0, 60.0):
                    for tick in (0.01, 0.5):
                        for inventory in (-5.0, 0.0, 5.0):
                            params = ModelParams(
                                gamma=gamma, kappa=kappa, sigma=sigma, tau=tau, tick=tick
                            )
                            reservation = reservation_price(100.0, inventory, params)
                            if reservation - optimal_spread(params) / 2 <= tick:
                                # No positive bid exists that far from the mid;
                                # quote_prices refuses it (see the test below).
                                continue
                            quote = quote_prices(100.0, inventory, params)
                            assert quote.bid < quote.ask
                            assert quote.bid > 0
                            assert quote.spread >= tick - 1e-9
                            checked += 1
    assert checked > 100, "the sweep must actually exercise the model"


def test_a_reservation_dragged_below_zero_is_refused():
    # gamma * sigma^2 * tau is huge, so the skew puts the quote off the grid.
    params = ModelParams(gamma=250.0, sigma=9.0, tau=3600.0)
    with pytest.raises(ValueError):
        quote_prices(100.0, 33.0, params)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": -1.0}, "gamma"),
        ({"kappa": 0.0}, "kappa"),
        ({"kappa": -2.0}, "kappa"),
        ({"sigma": -0.5}, "sigma"),
        ({"tau": -1.0}, "tau"),
        ({"tick": 0.0}, "tick"),
        ({"tick": -0.01}, "tick"),
        ({"gamma": float("nan")}, "gamma"),
        ({"gamma": float("inf")}, "gamma"),
        ({"sigma": float("nan")}, "sigma"),
    ],
)
def test_invalid_parameters_are_rejected_by_name(kwargs, message):
    with pytest.raises(ValueError) as excinfo:
        ModelParams(**kwargs)
    assert message in str(excinfo.value)


def test_booleans_are_not_numbers():
    with pytest.raises(TypeError):
        ModelParams(gamma=True)  # type: ignore[arg-type]


def test_strings_are_not_numbers():
    with pytest.raises(TypeError):
        ModelParams(kappa="1.0")  # type: ignore[arg-type]


def test_zero_gamma_used_to_divide_by_zero():
    # Regression: the shipped formula raised ZeroDivisionError at 2 / gamma.
    with pytest.raises(ValueError):
        ModelParams(gamma=0.0)


def test_negative_kappa_used_to_cross_the_quote():
    # Regression: gamma/kappa < -1 drove the spread negative, so bid > ask.
    params = ModelParams(gamma=1.0, kappa=1.0)
    with pytest.raises(ValueError):
        ModelParams(gamma=1.0, kappa=-2.0)
    assert quote_prices(100.0, 0.0, params).bid < quote_prices(100.0, 0.0, params).ask


def test_mid_must_be_a_positive_price():
    params = ModelParams()
    with pytest.raises(ValueError):
        reservation_price(0.0, 0.0, params)
    with pytest.raises(ValueError):
        reservation_price(-5.0, 0.0, params)


def test_quotes_shift_down_with_inventory():
    params = ModelParams()
    flat = quote_prices(100.0, 0.0, params)
    long = quote_prices(100.0, 20.0, params)
    assert long.bid < flat.bid
    assert long.ask < flat.ask


def test_snap_to_tick_rounds_half_away_from_zero_and_strips_dust():
    assert snap_to_tick(1.005, 0.01) == pytest.approx(1.01)
    assert snap_to_tick(1.004, 0.01) == pytest.approx(1.00)
    assert snap_to_tick(-1.005, 0.01) == pytest.approx(-1.01)
    assert snap_to_tick(1.25, 0.5) == pytest.approx(1.5)
    assert snap_to_tick(99.5, 0.5) == pytest.approx(99.5)


def test_snap_to_tick_rejects_a_non_positive_grid():
    with pytest.raises(ValueError):
        snap_to_tick(10.0, 0.0)


@pytest.mark.parametrize(
    "inventory, max_inventory, side, lot, expected",
    [
        (0.0, 100.0, "buy", 1.0, 1.0),
        (99.5, 100.0, "buy", 1.0, 0.5),
        (100.0, 100.0, "buy", 1.0, 0.0),
        (120.0, 100.0, "buy", 1.0, 0.0),
        (0.0, 100.0, "sell", 1.0, 1.0),
        (-99.5, 100.0, "sell", 1.0, 0.5),
        (-100.0, 100.0, "sell", 1.0, 0.0),
        (-120.0, 100.0, "sell", 1.0, 0.0),
    ],
)
def test_inventory_headroom_caps_the_order_size(inventory, max_inventory, side, lot, expected):
    assert inventory_headroom(inventory, max_inventory, side, lot) == pytest.approx(expected)


def test_inventory_headroom_validates_its_inputs():
    with pytest.raises(ValueError):
        inventory_headroom(0.0, 100.0, "hold", 1.0)
    with pytest.raises(ValueError):
        inventory_headroom(0.0, 0.0, "buy", 1.0)
    with pytest.raises(ValueError):
        inventory_headroom(0.0, 100.0, "buy", 0.0)


def test_params_are_immutable():
    params = ModelParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.gamma = 2.0  # type: ignore[misc]


def test_with_returns_a_validated_copy():
    params = ModelParams()
    tighter = params.with_(gamma=2.0)
    assert tighter.gamma == 2.0
    assert params.gamma == 1.0
    with pytest.raises(ValueError):
        params.with_(gamma=0.0)
