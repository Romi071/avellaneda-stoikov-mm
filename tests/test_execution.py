"""Exchange response interpretation and fill accounting."""

from __future__ import annotations

import pytest

from as_mm.execution import (
    CancelResult,
    OrderState,
    PayloadError,
    apply_fill,
    book_execution,
    fill_totals,
    interpret_cancel,
    is_retryable_status,
    parse_order_ack,
)


# --------------------------------------------------------------------------
# cancel interpretation
# --------------------------------------------------------------------------
def test_cancel_200_books_the_difference_between_resting_and_cancelled():
    result = interpret_cancel(200, {"success": True, "cancelled_amount": 0.3}, 1.0)
    assert result.state is OrderState.GONE
    assert result.cancelled == pytest.approx(0.3)
    assert result.filled == pytest.approx(0.7)
    assert result.resolved


def test_cancel_200_with_nothing_filled():
    result = interpret_cancel(200, {"cancelled_amount": 1.0}, 1.0)
    assert result.filled == pytest.approx(0.0)


def test_cancel_404_means_the_order_was_fully_filled():
    result = interpret_cancel(404, {"success": False, "cancelled_amount": 0}, 1.0)
    assert result.state is OrderState.FILLED
    assert result.filled == pytest.approx(1.0)
    assert result.resolved


def test_cancel_401_is_unresolved_and_books_nothing():
    # The engine omits cancelled_amount on a 401; the shipped code crashed here.
    result = interpret_cancel(401, {"success": False, "detail": "Unauthorized"}, 1.0)
    assert result.state is OrderState.UNKNOWN
    assert result.filled == pytest.approx(0.0)
    assert not result.resolved
    assert "401" in result.reason


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses_are_unresolved(status):
    result = interpret_cancel(status, {"detail": "boom"}, 1.0)
    assert result.state is OrderState.UNKNOWN
    assert not result.resolved
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert not is_retryable_status(status)


def test_cancel_200_without_the_key_is_unresolved():
    result = interpret_cancel(200, {"success": True}, 1.0)
    assert result.state is OrderState.UNKNOWN
    assert "cancelled_amount" in result.reason


def test_cancel_200_with_a_non_numeric_amount_is_unresolved():
    result = interpret_cancel(200, {"cancelled_amount": "0.5"}, 1.0)
    assert result.state is OrderState.UNKNOWN


def test_cancel_returning_more_than_resting_is_clamped_and_flagged():
    result = interpret_cancel(200, {"cancelled_amount": 5.0}, 1.0)
    assert result.state is OrderState.GONE
    assert result.cancelled == pytest.approx(1.0)
    assert result.filled == pytest.approx(0.0)
    assert "outside" in result.reason


def test_cancel_body_that_is_not_an_object_is_unresolved():
    result = interpret_cancel(200, ["nope"], 1.0)
    assert result.state is OrderState.UNKNOWN


def test_cancel_rejects_a_negative_resting_amount():
    with pytest.raises(PayloadError):
        interpret_cancel(200, {"cancelled_amount": 0}, -1.0)


def test_in_doubt_is_unknown():
    result = CancelResult.in_doubt("transport down")
    assert result.state is OrderState.UNKNOWN
    assert not result.resolved


# --------------------------------------------------------------------------
# order acknowledgements
# --------------------------------------------------------------------------
def _ack(**overrides):
    payload = {
        "orderId": "abc",
        "secret_token": "deadbeef",
        "client_tag": "as_mm",
        "active": True,
        "amount": 0.6,
        "fills": [{"price": 99.5, "amount": 0.4}],
    }
    payload.update(overrides)
    return payload


def test_order_ack_is_parsed():
    ack = parse_order_ack(_ack())
    assert ack.order_id == "abc"
    assert ack.secret_token == "deadbeef"
    assert ack.remaining == pytest.approx(0.6)
    assert ack.is_resting
    assert len(ack.fills) == 1


def test_fully_filled_order_is_not_resting():
    ack = parse_order_ack(_ack(amount=0.0, active=False))
    assert not ack.is_resting


@pytest.mark.parametrize("missing", ["orderId", "secret_token", "amount"])
def test_order_ack_missing_a_key_names_it(missing):
    payload = _ack()
    payload.pop(missing)
    with pytest.raises(PayloadError) as excinfo:
        parse_order_ack(payload)
    assert missing in str(excinfo.value)


def test_order_ack_with_a_negative_remaining_is_rejected():
    with pytest.raises(PayloadError):
        parse_order_ack(_ack(amount=-1.0))


def test_order_ack_with_non_list_fills_is_rejected():
    with pytest.raises(PayloadError):
        parse_order_ack(_ack(fills={"price": 1.0}))


def test_order_ack_with_a_non_object_body_is_rejected():
    with pytest.raises(PayloadError):
        parse_order_ack("ok")


def test_non_dict_fills_are_dropped_not_trusted():
    ack = parse_order_ack(_ack(fills=[{"price": 99.5, "amount": 0.4}, "junk"]))
    assert len(ack.fills) == 1


# --------------------------------------------------------------------------
# fill totals
# --------------------------------------------------------------------------
def test_fill_totals_sums_a_multi_level_sweep():
    shares, notional = fill_totals(
        [
            {"price": 100.5, "amount": 1.0},
            {"price": 100.6, "amount": 2.0},
        ]
    )
    assert shares == pytest.approx(3.0)
    assert notional == pytest.approx(100.5 + 201.2)


def test_fill_totals_of_nothing_is_zero():
    assert fill_totals([]) == (0.0, 0.0)


def test_fill_totals_rejects_a_receipt_without_a_price():
    with pytest.raises(PayloadError):
        fill_totals([{"amount": 1.0}])


def test_fill_totals_rejects_a_non_positive_price():
    with pytest.raises(PayloadError):
        fill_totals([{"price": 0.0, "amount": 1.0}])


def test_fill_totals_rejects_a_non_list():
    with pytest.raises(PayloadError):
        fill_totals({"price": 1.0})


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------
def test_buying_adds_inventory_and_spends_cash():
    inventory, balance = apply_fill(0.0, 1000.0, "buy", 99.5, 2.0)
    assert inventory == pytest.approx(2.0)
    assert balance == pytest.approx(1000.0 - 199.0)


def test_selling_removes_inventory_and_raises_cash():
    inventory, balance = apply_fill(5.0, 0.0, "sell", 100.5, 2.0)
    assert inventory == pytest.approx(3.0)
    assert balance == pytest.approx(201.0)


def test_book_execution_uses_the_notional_not_the_limit_price():
    inventory, balance = book_execution(0.0, 0.0, "buy", 3.0, 301.1)
    assert inventory == pytest.approx(3.0)
    assert balance == pytest.approx(-301.1)


def test_booking_round_trip_is_flat():
    inventory, balance = 0.0, 0.0
    inventory, balance = apply_fill(inventory, balance, "buy", 100.0, 1.0)
    inventory, balance = apply_fill(inventory, balance, "sell", 101.0, 1.0)
    assert inventory == pytest.approx(0.0)
    assert balance == pytest.approx(1.0)


def test_accounting_rejects_a_bad_side_and_negative_sizes():
    with pytest.raises(ValueError):
        apply_fill(0.0, 0.0, "hold", 100.0, 1.0)
    with pytest.raises(ValueError):
        apply_fill(0.0, 0.0, "buy", 100.0, -1.0)
    with pytest.raises(ValueError):
        apply_fill(0.0, 0.0, "buy", 0.0, 1.0)
    with pytest.raises(ValueError):
        book_execution(0.0, 0.0, "sell", -1.0, 0.0)
