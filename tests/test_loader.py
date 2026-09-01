"""Unit tests for data loader and synthetic generator."""

from __future__ import annotations

import pandas as pd
import pytest
from hquant.data.loader import generate_synthetic_ohlc, load_mt5_csv


def test_synthetic_generator_shape_and_columns() -> None:
    """Verify synthetic generator generates exact requested dimensions and schema."""
    n_bars = 500
    df = generate_synthetic_ohlc(n_bars=n_bars, seed=42)

    assert len(df) == n_bars
    assert set(df.columns) == {"open", "high", "low", "close", "true_regime"}
    assert (df["high"] >= df["low"]).all()
    assert (df["open"] > 0).all()
    assert (df["close"] > 0).all()
    assert set(df["true_regime"].unique()).issubset({0, 1})


def test_synthetic_generator_invalid_args() -> None:
    """Verify synthetic generator raises on invalid parameters."""
    with pytest.raises(ValueError, match="n_bars must be positive"):
        generate_synthetic_ohlc(n_bars=0)

    with pytest.raises(ValueError, match="start_price must be positive"):
        generate_synthetic_ohlc(start_price=-1.0)

    with pytest.raises(ValueError, match="trend_prob must be in"):
        generate_synthetic_ohlc(trend_prob=1.5)


def test_load_mt5_csv_valid_export(tmp_path: pytest.TempPathFactory) -> None:
    """Verify loading standard MT5 CSV export."""
    csv_file = tmp_path / "EURUSD_H1.csv"  # type: ignore[operator]
    content = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2023.01.02\t00:00:00\t1.0700\t1.0750\t1.0690\t1.0740\t1200\t0\t1\n"
        "2023.01.02\t01:00:00\t1.0740\t1.0760\t1.0720\t1.0730\t1400\t0\t1\n"
    )
    csv_file.write_text(content)

    df = load_mt5_csv(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2023-01-02 00:00:00")
    assert df.loc[df.index[0], "close"] == 1.0740


def test_load_mt5_csv_missing_file() -> None:
    """Verify FileNotFoundError on non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_mt5_csv("non_existent_file_xyz.csv")


def test_load_mt5_csv_empty_file(tmp_path: pytest.TempPathFactory) -> None:
    """Verify ValueError on empty file."""
    empty_file = tmp_path / "empty.csv"  # type: ignore[operator]
    empty_file.write_text("")

    with pytest.raises(ValueError):
        load_mt5_csv(empty_file)
