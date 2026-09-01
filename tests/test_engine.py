"""Unit tests for backtester and trailing drawdown accounting engine."""

from __future__ import annotations

import pandas as pd
import pytest
from hquant.backtest.engine import (
    Backtester,
    BacktestResult,
    Trade,
    TrailingDrawdownAccount,
)
from hquant.data.loader import generate_synthetic_ohlc
from hquant.features.engineering import atr


def test_account_constructor_validations() -> None:
    """Verify TrailingDrawdownAccount validates input parameters."""
    with pytest.raises(ValueError, match="initial_balance must be positive"):
        TrailingDrawdownAccount(initial_balance=0.0, dd_pct=6.0)

    with pytest.raises(ValueError, match="dd_pct must be positive"):
        TrailingDrawdownAccount(initial_balance=5000.0, dd_pct=-1.0)

    with pytest.raises(ValueError, match="soft_stop_ratio must be in"):
        TrailingDrawdownAccount(initial_balance=5000.0, dd_pct=6.0, soft_stop_ratio=1.5)


def test_backtest_result_metrics() -> None:
    """Verify BacktestResult calculates performance metrics accurately."""
    idx = pd.date_range("2023-01-01", periods=5, freq="1h")
    eq = pd.Series([5000.0, 5100.0, 5050.0, 5200.0, 5250.0], index=idx)
    fl = pd.Series([4700.0, 4800.0, 4800.0, 4900.0, 4950.0], index=idx)

    trade1 = Trade(
        entry_time=idx[0],
        exit_time=idx[1],
        direction=1,
        entry_price=1.0,
        exit_price=1.02,
        size=5000.0,
        stop=0.99,
        target=1.03,
        r_unit=0.01,
        strategy="signal",
        pnl=100.0,
        r_multiple=2.0,
        exit_reason="target",
    )

    res = BacktestResult(
        equity_curve=eq,
        floor_curve=fl,
        trades=[trade1],
        breached=False,
        breach_time=None,
        initial_balance=5000.0,
    )

    m = res.metrics()
    assert m["final_equity"] == pytest.approx(5250.0)
    assert m["total_return_pct"] == pytest.approx(5.0)
    assert m["n_trades"] == 1
    assert m["win_rate"] == 1.0
    assert m["profit_factor"] == float("inf")
    assert not m["breached"]

    df_trades = res.trade_frame
    assert len(df_trades) == 1
    assert df_trades.loc[0, "pnl"] == 100.0

    summary = res.summary()
    assert "Backtest Execution Summary" in summary
    assert "Account survived full sample" in summary


def test_backtest_execution_with_partial_and_trailing() -> None:
    """Verify backtester handles partial profit taking and trailing stops."""
    ohlc = generate_synthetic_ohlc(n_bars=200, seed=99)
    signals = pd.Series(0.0, index=ohlc.index)
    signals.iloc[10] = 1.0  # single buy signal

    bt = Backtester(
        initial_balance=5000.0,
        dd_pct=6.0,
        risk_pct=0.40,
        partial_at_r=1.0,
        partial_fraction=0.50,
        trail_start_r=0.50,
    )

    result = bt.run(ohlc, signals, atr(ohlc, 14))
    assert len(result.equity_curve) > 0
    assert len(result.floor_curve) == len(result.equity_curve)
