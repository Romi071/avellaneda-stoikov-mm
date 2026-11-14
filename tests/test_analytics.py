"""Attribution over a synthetic ledger, including the reconciliation identity."""

from __future__ import annotations

import json
import random
import sqlite3

import pytest

from analytics import main as analytics_cli
from as_mm.analytics import (
    AnalysisError,
    attribute,
    dumps,
    equity_curve,
    format_report,
    latest_price,
    load_trades,
    open_ledger,
    write_curve_csv,
)

OURS = "as_mm"


def read(path):
    connection = open_ledger(str(path))
    try:
        return load_trades(connection, OURS)
    finally:
        connection.close()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def test_taker_rows_are_ours_with_the_taker_side(ledger):
    path = ledger([("buy", OURS, "someone", 100.0, 1.0)])
    trades = read(path)
    assert len(trades) == 1
    assert trades[0].side == "buy"
    assert trades[0].role == "taker"


def test_maker_rows_are_the_opposite_side(ledger):
    path = ledger([("buy", "someone", OURS, 100.0, 1.0)])
    trade = read(path)[0]
    assert trade.side == "sell"
    assert trade.role == "maker"


def test_reference_is_the_previous_print(ledger):
    path = ledger(
        [
            ("sell", "noise", "noise", 99.0, 1.0),
            ("buy", OURS, "someone", 100.0, 1.0),
        ]
    )
    trades = read(path)
    assert len(trades) == 1, "other participants are not our fills"
    assert trades[0].reference == pytest.approx(99.0)
    assert trades[0].ref == pytest.approx(99.0)


def test_other_participants_are_excluded(ledger):
    path = ledger(
        [
            ("buy", "a", "b", 100.0, 1.0),
            ("sell", "b", "a", 100.0, 1.0),
        ]
    )
    connection = open_ledger(str(path))
    with pytest.raises(AnalysisError) as excinfo:
        load_trades(connection, OURS)
    connection.close()
    assert "as_mm" in str(excinfo.value)


def test_open_ledger_does_not_create_a_missing_file(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(AnalysisError) as excinfo:
        open_ledger(str(missing))
    assert "not found" in str(excinfo.value)
    assert not missing.exists(), "sqlite3.connect would have created an empty database"


def test_open_ledger_rejects_a_non_sqlite_file(tmp_path):
    bogus = tmp_path / "bogus.db"
    bogus.write_text("this is not a database")
    with pytest.raises(AnalysisError):
        open_ledger(str(bogus))


def test_open_ledger_rejects_a_database_without_trades(tmp_path):
    path = tmp_path / "empty.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE other (id INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(AnalysisError) as excinfo:
        open_ledger(str(path))
    assert "trades" in str(excinfo.value)
    assert "other" in str(excinfo.value)


def test_open_ledger_rejects_a_directory(tmp_path):
    with pytest.raises(AnalysisError):
        open_ledger(str(tmp_path))


def test_invalid_prices_are_rejected(ledger):
    path = ledger([("buy", OURS, "x", 0.0, 1.0)])
    with pytest.raises(AnalysisError):
        read(path)


def test_latest_price_is_the_last_print(ledger):
    path = ledger(
        [
            ("buy", "a", "b", 100.0, 1.0),
            ("sell", "a", "b", 102.5, 1.0),
        ]
    )
    connection = open_ledger(str(path))
    try:
        assert latest_price(connection) == pytest.approx(102.5)
    finally:
        connection.close()


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------
def test_a_maker_sell_above_the_reference_captures_half_a_point(ledger):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("buy", "taker", OURS, 100.5, 1.0),
        ]
    )
    trades = read(path)
    result = attribute(trades, mark=100.0, tag=OURS)

    assert result.inventory == pytest.approx(-1.0)
    assert result.cash == pytest.approx(100.5)
    assert result.equity == pytest.approx(0.5)
    assert result.spread_capture == pytest.approx(0.5)
    assert result.inventory_pnl == pytest.approx(0.0)
    assert result.total == pytest.approx(result.equity, abs=1e-12)


def test_a_buy_below_the_reference_plus_drift(ledger):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("sell", "taker", OURS, 99.0, 1.0),
        ]
    )
    result = attribute(read(path), mark=101.0, tag=OURS)

    assert result.inventory == pytest.approx(1.0)
    assert result.cash == pytest.approx(-99.0)
    assert result.equity == pytest.approx(2.0)
    assert result.spread_capture == pytest.approx(1.0)  # bought a point under
    assert result.inventory_pnl == pytest.approx(1.0)  # then the mark rose
    assert result.total == pytest.approx(2.0, abs=1e-12)


def test_round_trip_reports_zero_inventory_and_pure_spread(ledger):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("buy", OURS, "counterparty", 99.5, 1.0),  # we buy a point under
            ("sell", "noise", "noise", 100.0, 1.0),  # an external print resets the ref
            ("sell", OURS, "counterparty", 100.5, 1.0),  # we sell a point over
        ]
    )
    result = attribute(read(path), mark=100.0, tag=OURS)

    assert result.inventory == pytest.approx(0.0)
    assert result.spread_capture == pytest.approx(1.0)
    assert result.inventory_pnl == pytest.approx(0.0)
    assert result.equity == pytest.approx(1.0)
    assert result.buys == 1 and result.sells == 1
    assert result.maker_fills + result.taker_fills == result.fills


def test_the_components_always_add_up_on_random_ledgers(ledger):
    rng = random.Random(20261114)
    for seed in range(25):
        rows = []
        price = 100.0
        for _ in range(rng.randint(2, 40)):
            price = max(1.0, round(price + rng.uniform(-1.5, 1.5), 2))
            side = rng.choice(["buy", "sell"])
            if rng.random() < 0.45:
                rows.append((side, OURS, "counterparty", price, round(rng.uniform(0.1, 5), 2)))
            elif rng.random() < 0.5:
                rows.append((side, "counterparty", OURS, price, round(rng.uniform(0.1, 5), 2)))
            else:
                rows.append((side, "noise", "noise", price, round(rng.uniform(0.1, 5), 2)))
        if not any(OURS in (row[1], row[2]) for row in rows):
            continue
        path = ledger(rows, name=f"ledger-{seed}.db")
        trades = read(path)
        connection = open_ledger(str(path))
        try:
            mark = latest_price(connection)
        finally:
            connection.close()
        result = attribute(trades, mark=mark, tag=OURS)
        assert result.equity == pytest.approx(result.total, abs=1e-9), f"seed {seed}"
        assert result.as_dict()["residual"] == pytest.approx(0.0, abs=1e-9)


def test_maker_spread_capture_splits_by_role(ledger):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("sell", OURS, "counterparty", 100.5, 1.0),  # taker sell above the ref
            ("sell", "counterparty", OURS, 99.5, 1.0),  # resting bid bought lower
        ]
    )
    result = attribute(read(path), mark=100.0, tag=OURS)
    assert result.taker_spread_capture == pytest.approx(0.5)
    assert result.maker_spread_capture == pytest.approx(1.0)
    assert result.taker_spread_capture + result.maker_spread_capture == pytest.approx(
        result.spread_capture
    )


def test_inventory_path_bounds_are_reported(ledger):
    path = ledger(
        [
            ("sell", OURS, "counterparty", 100.0, 1.0),
            ("sell", OURS, "counterparty", 100.0, 2.0),
            ("buy", OURS, "counterparty", 100.0, 1.0),
        ]
    )
    result = attribute(read(path), mark=100.0, tag=OURS)
    # sells of 1 and 2 take the position to -3, then a buy of 1 leaves it at -2
    assert result.inventory_min == pytest.approx(-3.0)
    assert result.inventory_max == pytest.approx(-1.0)
    assert result.inventory == pytest.approx(-2.0)


def test_an_empty_selection_is_an_error():
    with pytest.raises(AnalysisError):
        attribute([], mark=100.0, tag=OURS)


def test_a_non_positive_mark_is_an_error(ledger):
    path = ledger([("buy", OURS, "x", 100.0, 1.0)])
    with pytest.raises(AnalysisError):
        attribute(read(path), mark=0.0, tag=OURS)


def test_equity_curve_of_nothing_is_empty():
    assert equity_curve([], 100.0) == []


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def test_curve_csv_has_a_header_and_lf_endings(ledger, tmp_path):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("sell", OURS, "counterparty", 100.5, 1.0),
        ]
    )
    result = attribute(read(path), mark=100.0, tag=OURS)
    out = tmp_path / "curve.csv"
    assert write_curve_csv(str(out), result.curve) == 1

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    lines = raw.decode().strip().split("\n")
    assert lines[0].startswith("id,timestamp,price")
    assert len(lines) == 2


def test_writing_an_empty_curve_is_refused(tmp_path):
    with pytest.raises(AnalysisError):
        write_curve_csv(str(tmp_path / "x.csv"), [])


def test_report_mentions_every_component(ledger):
    path = ledger([("buy", "taker", OURS, 100.0, 1.0)])
    text = format_report(attribute(read(path), mark=101.0, tag=OURS))
    for needle in ("spread capture", "inventory PnL", "equity", "residual", "as_mm"):
        assert needle in text


def test_json_dump_has_no_curve_by_default(ledger):
    path = ledger([("buy", "taker", OURS, 100.0, 1.0)])
    result = attribute(read(path), mark=101.0, tag=OURS)
    payload = json.loads(dumps(result))
    assert "curve" not in payload
    assert payload["total"] == pytest.approx(result.total)
    assert "curve" in json.loads(dumps(result, curve=True))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_prints_json_and_exits_zero(ledger, capsys):
    path = ledger(
        [
            ("sell", "noise", "noise", 100.0, 1.0),
            ("sell", OURS, "counterparty", 100.5, 1.0),
        ]
    )
    assert analytics_cli(["--db", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fills"] == 1
    assert payload["spread_capture"] == pytest.approx(0.5)


def test_cli_reports_a_missing_ledger_cleanly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        analytics_cli(["--db", str(tmp_path / "missing.db")])
    assert excinfo.value.code == 2


def test_cli_rejects_a_zero_limit(ledger):
    path = ledger([("buy", "taker", OURS, 100.0, 1.0)])
    with pytest.raises(SystemExit) as excinfo:
        analytics_cli(["--db", str(path), "--limit", "0"])
    assert excinfo.value.code == 2


def test_cli_limit_keeps_the_newest_fills(ledger, capsys):
    path = ledger(
        [
            ("sell", "taker", OURS, 100.0, 1.0),
            ("sell", "taker", OURS, 101.0, 1.0),
            ("sell", "taker", OURS, 102.0, 1.0),
        ]
    )
    assert analytics_cli(["--db", str(path), "--json", "--limit", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fills"] == 2


def test_cli_mark_override_changes_equity(ledger, capsys):
    path = ledger([("buy", OURS, "counterparty", 100.0, 1.0)])
    assert analytics_cli(["--db", str(path), "--json", "--mark", "110"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["equity"] == pytest.approx(10.0)
