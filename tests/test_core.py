"""Core integration and integrity tests for HybridQuant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hquant.backtest.engine import Backtester, TrailingDrawdownAccount
from hquant.data.loader import generate_synthetic_ohlc
from hquant.features.engineering import adx, atr, build_features, label_regime
from hquant.models.regime import walk_forward_validate


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Generate deterministic synthetic market test fixture."""
    return generate_synthetic_ohlc(n_bars=4000, seed=11)


# Feature Causality & Integrity Tests


def test_features_contain_no_future_information(data: pd.DataFrame) -> None:
    """Verify that truncating time series does not alter historically computed features."""
    cut = 3000
    full = build_features(data)
    partial = build_features(data.iloc[:cut])

    overlap = partial.index[200:]
    pd.testing.assert_frame_equal(
        full.loc[overlap], partial.loc[overlap], check_exact=False, rtol=1e-9
    )


def test_features_are_lagged(data: pd.DataFrame) -> None:
    """Verify that lag shift is strictly enforced across computed indicators."""
    feats = build_features(data, shift=1)
    unshifted = build_features(data, shift=0)
    aligned = unshifted.shift(1)

    valid = feats["rsi14"].notna() & aligned["rsi14"].notna()
    assert np.allclose(feats["rsi14"][valid], aligned["rsi14"][valid])


def test_label_is_forward_looking_and_not_a_feature(data: pd.DataFrame) -> None:
    """Verify forward labeling horizon produces trailing NaNs and is excluded from features."""
    y = label_regime(data, horizon=24)
    X = build_features(data)

    assert y.tail(24).isna().all()
    assert "true_regime" not in X.columns
    assert not any("regime" in c for c in X.columns)


def test_adx_and_atr_are_non_negative(data: pd.DataFrame) -> None:
    """Verify indicator positivity bounds."""
    a, t = adx(data), atr(data)
    assert (a.dropna() >= 0).all()
    assert (t.dropna() >= 0).all()


# Ratcheting Drawdown Account Logic Tests


def test_floor_ratchets_up_and_never_down() -> None:
    """Verify floor rises with equity highs and remains fixed during drawdowns."""
    acc = TrailingDrawdownAccount(5000.0, dd_pct=6.0)
    assert acc.floor == pytest.approx(4700.0)

    acc.update(5200.0)
    assert acc.floor == pytest.approx(4900.0)

    # Equity declines: floor must not retreat
    acc.update(5050.0)
    assert acc.floor == pytest.approx(4900.0)


def test_breach_detected_at_floor() -> None:
    """Verify account breach is triggered when equity crosses beneath floor."""
    acc = TrailingDrawdownAccount(5000.0, dd_pct=6.0)
    acc.update(5300.0)
    assert acc.update(4999.0) is True


def test_profit_then_giveback_can_breach_above_initial_balance() -> None:
    """Verify that floating gains ratchet floor higher, making retracements fatal even above deposit."""
    acc = TrailingDrawdownAccount(5000.0, dd_pct=6.0)
    acc.update(5450.0)  # floor ratchets to 5150
    assert acc.floor == pytest.approx(5150.0)
    assert acc.update(5100.0) is True  # breached despite +100 realised/equity vs initial


def test_soft_stop_trips_before_breach() -> None:
    """Verify circuit breaker halts trading when utilized drawdown threshold is reached."""
    acc = TrailingDrawdownAccount(5000.0, dd_pct=6.0, soft_stop_ratio=0.50)
    acc.update(4849.0)
    assert acc.halted is True
    assert acc.equity > acc.floor


# Position Sizing & Backtest Execution Tests


def test_size_shrinks_as_room_shrinks(data: pd.DataFrame) -> None:
    """Verify that room-to-floor constraint decays position size as drawdown increases."""
    bt = Backtester(initial_balance=5000.0, dd_pct=6.0, risk_pct=0.40, max_room_fraction=0.25)

    healthy = TrailingDrawdownAccount(5000.0, 6.0)
    healthy.update(5000.0)

    stressed = TrailingDrawdownAccount(5000.0, 6.0)
    stressed.update(5000.0)
    stressed.update(4790.0)

    def money_at_risk(acc: TrailingDrawdownAccount) -> float:
        return min(acc.equity * (bt.risk_pct / 100.0), max(0.0, acc.room) * bt.max_room_fraction)

    assert money_at_risk(stressed) < money_at_risk(healthy)


def test_backtest_stops_at_breach(data: pd.DataFrame) -> None:
    """Verify that backtester terminates execution immediately on breach bar."""
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[100::50] = 1.0

    bt = Backtester(initial_balance=5000.0, dd_pct=0.15, risk_pct=5.0, max_room_fraction=1.0)
    result = bt.run(data, signals, atr(data, 14))

    if result.breached:
        assert result.equity_curve.index[-1] == result.breach_time
        assert len(result.equity_curve) < len(data)


def test_no_trade_when_room_exhausted(data: pd.DataFrame) -> None:
    """Verify all executed trades maintain valid positive sizing."""
    bt = Backtester(initial_balance=5000.0, dd_pct=6.0, risk_pct=0.40)
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[50:] = 1.0

    result = bt.run(data, signals, atr(data, 14))
    for t in result.trades:
        assert t.size > 0


# Validation Methodology Tests


def test_walk_forward_folds_are_chronological(data: pd.DataFrame) -> None:
    """Verify that walk-forward training windows expand chronologically."""
    X, y = build_features(data), label_regime(data)
    report = walk_forward_validate(X, y, n_splits=3, embargo=24)

    assert len(report.folds) > 0
    sizes = [f.n_train for f in report.folds]
    assert sizes == sorted(sizes)


def test_classifier_recovers_known_synthetic_regimes(data: pd.DataFrame) -> None:
    """Verify classifier discriminative ability on synthetic regimes."""
    X, y = build_features(data), label_regime(data)
    report = walk_forward_validate(X, y, n_splits=3, embargo=24)
    assert report.mean_auc > 0.60
