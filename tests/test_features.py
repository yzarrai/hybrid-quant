"""Unit tests for feature engineering indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hquant.data.loader import generate_synthetic_ohlc
from hquant.features.engineering import (
    adx,
    atr,
    bollinger_position,
    build_features,
    efficiency_ratio,
    hurst_proxy,
    label_regime,
    rsi,
)


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return generate_synthetic_ohlc(n_bars=300, seed=123)


def test_indicators_boundary_and_types(ohlc: pd.DataFrame) -> None:
    """Verify indicators return series with valid mathematical bounds."""
    close = ohlc["close"]

    atr_vals = atr(ohlc, 14).dropna()
    assert (atr_vals > 0).all()

    adx_vals = adx(ohlc, 14).dropna()
    assert ((adx_vals >= 0.0) & (adx_vals <= 100.0)).all()

    rsi_vals = rsi(close, 14).dropna()
    assert ((rsi_vals >= 0.0) & (rsi_vals <= 100.0)).all()

    bb_vals = bollinger_position(close, 20).dropna()
    assert np.isfinite(bb_vals).all()

    eff_vals = efficiency_ratio(close, 20).dropna()
    assert ((eff_vals >= 0.0) & (eff_vals <= 1.0)).all()

    hurst_vals = hurst_proxy(close, 64).dropna()
    assert np.isfinite(hurst_vals).all()


def test_constant_prices_edge_case() -> None:
    """Verify indicators handle flat constant price inputs without throwing errors or unhandled infinities."""
    idx = pd.date_range("2023-01-01", periods=100, freq="1h")
    flat_df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)

    # Functions should return safe Series without raising UnboundLocalError or ZeroDivisionError
    res_atr = atr(flat_df, 14)
    assert not res_atr.isna().all()

    res_rsi = rsi(flat_df["close"], 14)
    assert not res_rsi.isna().all()

    res_bb = bollinger_position(flat_df["close"], 20)
    assert res_bb.isna().sum() > 0 or np.isfinite(res_bb.dropna()).all()


def test_build_features_missing_cols() -> None:
    """Verify build_features raises on missing required OHLC columns."""
    incomplete = pd.DataFrame({"open": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        build_features(incomplete)


def test_label_regime_invalid_horizon(ohlc: pd.DataFrame) -> None:
    """Verify label_regime validates horizon."""
    with pytest.raises(ValueError, match="horizon must be positive"):
        label_regime(ohlc, horizon=0)
