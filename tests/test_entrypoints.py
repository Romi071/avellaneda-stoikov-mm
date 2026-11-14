"""Entry points must be importable and side-effect free.

The shipped scripts called ``asyncio.run(main())`` at module scope, so
``import market_maker`` started trading immediately and died with
``ClientConnectorError`` when no exchange was listening.  Nothing could be
imported, reused or tested.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize("module", ["market_maker", "noise_simulator", "analytics", "as_mm"])
def test_importing_a_module_does_not_start_anything(module):
    completed = run("-c", f"import {module}; print('imported')", timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "imported" in completed.stdout
    assert "ClientConnectorError" not in completed.stderr


def test_importing_the_engine_needs_no_running_exchange():
    # Regression: this used to raise ClientConnectorError on port 8000.
    completed = run(
        "-c",
        "import market_maker as m;print(m.Avellaneda_Stoikov().params.gamma)",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1.0"


@pytest.mark.parametrize("script", ["market_maker.py", "noise_simulator.py", "analytics.py"])
def test_cli_help_exits_zero(script):
    completed = run(str(ROOT / script), "--help", timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_visualizer_help_exits_zero():
    # QtGui needs libGL, which a headless box does not have: skip there.
    try:
        import pyqtgraph  # noqa: F401
        from PyQt6 import QtGui  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Qt runtime unavailable: {exc}")
    completed = run(str(ROOT / "depth_visualizer.py"), "--help", timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_analytics_without_a_ledger_fails_loudly(tmp_path):
    missing = tmp_path / "nowhere" / "ledger.db"
    completed = run(str(ROOT / "analytics.py"), "--db", str(missing), timeout=60)
    assert completed.returncode == 2
    assert "not found" in completed.stderr
    assert not missing.exists()


def test_market_maker_rejects_an_invalid_gamma_on_the_command_line():
    completed = run(str(ROOT / "market_maker.py"), "--once", "--gamma", "0", timeout=60)
    assert completed.returncode == 2
    assert "gamma" in completed.stderr


def test_console_script_shims_are_callable():
    completed = run(
        "-c",
        "import market_maker, noise_simulator, analytics;"
        "assert callable(market_maker.cli);"
        "assert callable(noise_simulator.cli);"
        "assert callable(analytics.cli);"
        "print('ok')",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
