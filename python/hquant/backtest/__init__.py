"""Event-driven backtesting and ratcheting drawdown accounting module."""

from __future__ import annotations

from hquant.backtest.engine import (
    Backtester,
    BacktestResult,
    Trade,
    TrailingDrawdownAccount,
)

__all__ = [
    "Trade",
    "BacktestResult",
    "TrailingDrawdownAccount",
    "Backtester",
]
