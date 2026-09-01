"""Event-driven backtesting engine with equity-based trailing drawdown simulation.

Simulates order execution, partial profit realization, ATR-based trailing stops,
and double-capped position sizing under a ratcheting account equity floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Trade:
    """Record of an executed trading position."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float | None
    size: float
    stop: float
    target: float
    r_unit: float
    strategy: str
    pnl: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    commission: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest output series and execution metrics."""

    equity_curve: pd.Series
    floor_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    breached: bool = False
    breach_time: pd.Timestamp | None = None
    initial_balance: float = 0.0

    @property
    def trade_frame(self) -> pd.DataFrame:
        """Convert trade history to a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "stop": t.stop,
                    "target": t.target,
                    "r_unit": t.r_unit,
                    "strategy": t.strategy,
                    "pnl": t.pnl,
                    "r_multiple": t.r_multiple,
                    "exit_reason": t.exit_reason,
                    "commission": t.commission,
                }
                for t in self.trades
            ]
        )

    def metrics(self, periods_per_year: int = 24 * 252) -> dict[str, Any]:
        """Compute institutional performance and risk metrics.

        Parameters
        ----------
        periods_per_year : int, default 24 * 252
            Number of sample periods per year (default H1 bars in 252 trading days).

        Returns
        -------
        dict[str, Any]
            Dictionary of calculated performance metrics.
        """
        closed = [t for t in self.trades if t.exit_time is not None]
        pnls = np.array([t.pnl for t in closed]) if closed else np.empty(0, dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        returns = self.equity_curve.pct_change().dropna()
        n_periods = len(self.equity_curve)

        # Sharpe ratio
        if len(returns) > 1 and float(returns.std(ddof=0)) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
        else:
            sharpe = float("nan")

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and float(downside_returns.std(ddof=0)) > 0:
            sortino = float(
                returns.mean() / downside_returns.std(ddof=0) * np.sqrt(periods_per_year)
            )
        else:
            sortino = float("nan")

        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max.replace(0.0, np.nan)
        max_dd = float(drawdown.min() * 100.0) if len(drawdown) > 0 else 0.0

        # Drawdown duration calculation (bars from peak to recovery)
        is_in_dd = self.equity_curve < running_max
        dd_runs = (~is_in_dd).cumsum()
        max_dd_bars = int(is_in_dd.groupby(dd_runs).sum().max()) if is_in_dd.any() else 0

        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0

        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        final_eq = (
            float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.initial_balance
        )
        total_return_pct = (
            float((final_eq / self.initial_balance - 1.0) * 100.0)
            if self.initial_balance > 0
            else 0.0
        )

        # Annualized return (CAGR)
        if n_periods > 0 and self.initial_balance > 0 and final_eq > 0:
            years = n_periods / periods_per_year
            cagr_pct = float(
                ((final_eq / self.initial_balance) ** (1.0 / max(years, 1e-4)) - 1.0) * 100.0
            )
        else:
            cagr_pct = float("nan")

        calmar = (
            float(abs(cagr_pct / max_dd)) if max_dd < 0 and np.isfinite(cagr_pct) else float("nan")
        )

        return {
            "final_equity": final_eq,
            "total_return_pct": total_return_pct,
            "cagr_pct": cagr_pct,
            "n_trades": len(closed),
            "win_rate": float(len(wins) / len(closed)) if len(closed) > 0 else float("nan"),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "expectancy": float(pnls.mean()) if len(pnls) > 0 else float("nan"),
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_dd,
            "max_drawdown_duration_bars": max_dd_bars,
            "breached": self.breached,
            "breach_time": self.breach_time,
        }

    def summary(self) -> str:
        """Generate human-readable backtest summary."""
        m = self.metrics()
        verdict = (
            f"ACCOUNT BREACHED at {m['breach_time']}"
            if m["breached"]
            else "Account survived full sample"
        )
        return "\n".join(
            [
                "Backtest Execution Summary",
                "=" * 48,
                f"Verdict              : {verdict}",
                f"Initial balance      : {self.initial_balance:,.2f}",
                f"Final equity         : {m['final_equity']:,.2f}",
                f"Total return         : {m['total_return_pct']:+.2f}%",
                f"Trades executed      : {m['n_trades']}",
                f"Win rate             : {m['win_rate']:.3f}",
                f"Expectancy / trade   : {m['expectancy']:+.2f}",
                f"Profit factor        : {m['profit_factor']:.3f}",
                f"Sharpe (annualized)  : {m['sharpe']:.3f}",
                f"Sortino (annualized) : {m['sortino']:.3f}",
                f"Max drawdown         : {m['max_drawdown_pct']:.2f}%",
                f"Max DD duration      : {m['max_drawdown_duration_bars']} bars",
            ]
        )


class TrailingDrawdownAccount:
    """Stateful accounting model for an equity-based trailing drawdown constraint."""

    def __init__(
        self,
        initial_balance: float,
        dd_pct: float,
        soft_stop_ratio: float = 0.60,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {initial_balance}")
        if dd_pct <= 0:
            raise ValueError(f"dd_pct must be positive, got {dd_pct}")
        if not (0.0 < soft_stop_ratio <= 1.0):
            raise ValueError(f"soft_stop_ratio must be in (0, 1], got {soft_stop_ratio}")

        self.initial_balance = initial_balance
        self.dd_budget = initial_balance * (dd_pct / 100.0)
        self.soft_stop_ratio = soft_stop_ratio
        self.equity = initial_balance
        self.peak = initial_balance
        self.halted = False

    @property
    def floor(self) -> float:
        """The ratchet floor level: highest equity peak minus drawdown budget."""
        return self.peak - self.dd_budget

    @property
    def room(self) -> float:
        """Distance between current mark-to-market equity and the breach floor."""
        return self.equity - self.floor

    @property
    def used_ratio(self) -> float:
        """Fraction of total drawdown budget consumed [0.0, 1.0]."""
        if self.dd_budget <= 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (self.room / self.dd_budget)))

    def update(self, equity: float) -> bool:
        """Update mark-to-market equity.

        Returns True if the account breached the floor.
        """
        self.equity = equity
        if equity > self.peak:
            self.peak = equity
        if self.used_ratio >= self.soft_stop_ratio:
            self.halted = True
        return equity <= self.floor


class Backtester:
    """Simulates trading strategy execution with prop-firm risk management."""

    def __init__(
        self,
        initial_balance: float = 5000.0,
        dd_pct: float = 6.0,
        risk_pct: float = 0.40,
        max_room_fraction: float = 0.25,
        cost_per_trade_pct: float = 0.02,
        partial_at_r: float = 1.0,
        partial_fraction: float = 0.50,
        trail_start_r: float = 0.50,
        trail_atr_mult: float = 1.50,
    ) -> None:
        self.initial_balance = initial_balance
        self.dd_pct = dd_pct
        self.risk_pct = risk_pct
        self.max_room_fraction = max_room_fraction
        self.cost_per_trade_pct = cost_per_trade_pct
        self.partial_at_r = partial_at_r
        self.partial_fraction = partial_fraction
        self.trail_start_r = trail_start_r
        self.trail_atr_mult = trail_atr_mult

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        atr_series: pd.Series,
        targets: pd.Series | None = None,
        sl_atr_mult: float = 2.0,
        tp_r_multiple: float = 2.0,
    ) -> BacktestResult:
        """Execute chronological event simulation across OHLC bars.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC DataFrame with DatetimeIndex.
        signals : pd.Series
            Signal series (+1 buy, -1 sell, 0 flat).
        atr_series : pd.Series
            ATR series aligned with df.
        targets : pd.Series | None, optional
            Explicit profit targets; if None, derived from `tp_r_multiple`.
        sl_atr_mult : float, default 2.0
            Stop loss distance as multiple of ATR.
        tp_r_multiple : float, default 2.0
            Take profit distance as multiple of initial R.

        Returns
        -------
        BacktestResult
            Backtest equity curve, floor curve, and trade records.
        """
        account = TrailingDrawdownAccount(self.initial_balance, self.dd_pct)

        equity_history: list[float] = []
        floor_history: list[float] = []
        trades: list[Trade] = []

        open_trade: Trade | None = None
        partial_taken = False
        realised = self.initial_balance

        breached = False
        breach_time: pd.Timestamp | None = None

        for i, ts in enumerate(df.index):
            bar = df.iloc[i]
            atr_now = atr_series.iloc[i]

            # 1. Manage open position
            if open_trade is not None:
                d = open_trade.direction
                exit_price: float | None = None
                reason = ""

                # Pessimistic intrabar evaluation: stop triggers before target
                if (
                    d == 1
                    and bar["low"] <= open_trade.stop
                    or d == -1
                    and bar["high"] >= open_trade.stop
                ):
                    exit_price = float(open_trade.stop)
                    reason = "stop"
                elif (
                    d == 1
                    and bar["high"] >= open_trade.target
                    or d == -1
                    and bar["low"] <= open_trade.target
                ):
                    exit_price = float(open_trade.target)
                    reason = "target"

                if exit_price is None:
                    # Mark unrealized move
                    move = (float(bar["close"]) - open_trade.entry_price) * d
                    r_now = move / open_trade.r_unit if open_trade.r_unit > 0 else 0.0

                    # Partial profit taking: converts floating gain into locked balance
                    if r_now >= self.partial_at_r and not partial_taken:
                        closed_size = open_trade.size * self.partial_fraction
                        gain = move * closed_size
                        realised += gain
                        open_trade.size -= closed_size
                        partial_taken = True

                    # Trailing stop update
                    if r_now >= self.trail_start_r and np.isfinite(atr_now) and atr_now > 0:
                        trail_level = float(bar["close"]) - d * atr_now * self.trail_atr_mult
                        if d == 1:
                            open_trade.stop = max(open_trade.stop, trail_level)
                        else:
                            open_trade.stop = min(open_trade.stop, trail_level)
                else:
                    gain = (exit_price - open_trade.entry_price) * d * open_trade.size
                    cost = abs(open_trade.entry_price * open_trade.size) * (
                        self.cost_per_trade_pct / 100.0
                    )
                    realised += gain - cost
                    open_trade.exit_time = ts
                    open_trade.exit_price = float(exit_price)
                    open_trade.pnl = float(gain - cost)
                    open_trade.r_multiple = float(
                        (exit_price - open_trade.entry_price) * d / open_trade.r_unit
                        if open_trade.r_unit > 0
                        else 0.0
                    )
                    open_trade.exit_reason = reason
                    open_trade.commission = float(cost)
                    trades.append(open_trade)
                    open_trade = None
                    partial_taken = False

            # 2. Mark to market & floor check
            unrealised = 0.0
            if open_trade is not None:
                unrealised = (
                    (float(bar["close"]) - open_trade.entry_price)
                    * open_trade.direction
                    * open_trade.size
                )
            equity = realised + unrealised

            if account.update(equity):
                breached = True
                breach_time = ts
                equity_history.append(equity)
                floor_history.append(account.floor)
                break

            equity_history.append(equity)
            floor_history.append(account.floor)

            # 3. Process new entries
            if open_trade is not None or account.halted:
                continue

            sig = signals.iloc[i]
            if not sig or np.isnan(sig) or sig == 0:
                continue
            if not np.isfinite(atr_now) or atr_now <= 0:
                continue

            d = int(np.sign(sig))
            entry = float(bar["close"])
            stop_dist = float(atr_now * sl_atr_mult)
            if stop_dist <= 0:
                continue

            stop = entry - d * stop_dist

            # Double-cap sizing
            risk_cap = equity * (self.risk_pct / 100.0)
            room_cap = max(0.0, account.room) * self.max_room_fraction
            money_at_risk = min(risk_cap, room_cap)
            if money_at_risk <= 0:
                continue

            size = money_at_risk / stop_dist
            if size <= 0:
                continue

            if targets is not None and np.isfinite(targets.iloc[i]):
                target = float(targets.iloc[i])
                if (d == 1 and target <= entry) or (d == -1 and target >= entry):
                    continue
            else:
                target = entry + d * stop_dist * tp_r_multiple

            open_trade = Trade(
                entry_time=ts,
                exit_time=None,
                direction=d,
                entry_price=entry,
                exit_price=None,
                size=float(size),
                stop=float(stop),
                target=float(target),
                r_unit=float(stop_dist),
                strategy="signal",
            )

        index = df.index[: len(equity_history)]
        return BacktestResult(
            equity_curve=pd.Series(equity_history, index=index, name="equity"),
            floor_curve=pd.Series(floor_history, index=index, name="floor"),
            trades=trades,
            breached=breached,
            breach_time=breach_time,
            initial_balance=self.initial_balance,
        )
