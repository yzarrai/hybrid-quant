"""Data loading and synthetic market simulation.

Provides utilities for importing MetaTrader 5 history exports and
generating regime-switching synthetic price series for reproducible testing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

OHLC_COLUMNS: list[str] = ["open", "high", "low", "close"]


def load_mt5_csv(
    path: str | Path,
    tz: str | None = None,
    required_columns: Sequence[str] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Load and normalize an MT5 bar export CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file exported from MetaTrader 5 History Center or script.
    tz : str | None, optional
        Timezone identifier (e.g., 'UTC', 'America/New_York') to localize timestamps.
    required_columns : Sequence[str], optional
        Columns required in the output frame. Defaults to ('open', 'high', 'low', 'close').

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame indexed by localized DatetimeIndex, sorted chronologically.

    Raises
    ------
    FileNotFoundError
        If the given file path does not exist.
    ValueError
        If timestamp columns or required price columns are missing or malformed.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"MT5 export file not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"File {file_path} is empty.")

    # Read header first to detect delimiter if necessary
    df = pd.read_csv(file_path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"File {file_path} is empty.")

    # Normalize column names: strip whitespace, angle brackets (<OPEN>), lowercase
    df.columns = [
        str(col).strip().lower().replace("<", "").replace(">", "").replace(" ", "_")
        for col in df.columns
    ]

    # Timestamp reconciliation across MT5 export dialects
    if "date" in df.columns and "time" in df.columns:
        stamp_series = df["date"].astype(str) + " " + df["time"].astype(str)
        timestamps = pd.to_datetime(stamp_series, errors="coerce")
    elif "datetime" in df.columns:
        timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    elif "time" in df.columns:
        timestamps = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        timestamps = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(
            f"No recognizable timestamp column found in {file_path}. Columns: {list(df.columns)}"
        )

    if timestamps.isna().all():
        raise ValueError(f"Failed to parse any valid datetime stamps from {file_path}.")

    df = df.set_index(pd.DatetimeIndex(timestamps))
    # Drop rows with unparseable timestamps
    df = df[df.index.notna()]
    df = df.sort_index()

    # Deduplicate timestamps, keeping last observed bar
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    if tz is not None and isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    rename_map = {
        "tickvol": "volume",
        "tick_volume": "volume",
        "vol": "volume",
        "realvol": "real_volume",
        "real_volume": "real_volume",
    }
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    # Keep only required columns and volume if present
    selected = list(required_columns)
    if "volume" in df.columns and "volume" not in selected:
        selected.append("volume")

    result = df[selected].astype(float)

    # Sanity checks on OHLC relationships
    if (result["open"] <= 0).any() or (result["close"] <= 0).any():
        raise ValueError("Non-positive prices found in dataset.")

    # Bound high and low to prevent data corruption artifacts
    open_arr = result["open"].to_numpy(dtype=float)
    high_arr = result["high"].to_numpy(dtype=float)
    low_arr = result["low"].to_numpy(dtype=float)
    close_arr = result["close"].to_numpy(dtype=float)

    result["high"] = np.maximum(high_arr, np.maximum(open_arr, close_arr))
    result["low"] = np.minimum(low_arr, np.minimum(open_arr, close_arr))

    return result


def generate_synthetic_ohlc(
    n_bars: int = 20_000,
    start: str = "2022-01-03",
    freq: str = "1h",
    seed: int = 7,
    start_price: float = 1.10,
    trend_prob: float = 0.35,
    mean_regime_len: int = 180,
    base_vol: float = 0.0009,
) -> pd.DataFrame:
    """Generate regime-switching geometric price simulation with alternating trending and ranging states.

    State 1 (Trending): Characterized by persistent drift with elevated volatility.
    State 0 (Ranging): Ornstein-Uhlenbeck mean-reverting process around an anchor.

    Parameters
    ----------
    n_bars : int, default 20_000
        Total number of bars to generate.
    start : str, default '2022-01-03'
        Starting datetime string for DatetimeIndex.
    freq : str, default '1h'
        Pandas frequency string.
    seed : int, default 7
        Random seed for full deterministic reproducibility.
    start_price : float, default 1.10
        Initial price level.
    trend_prob : float, default 0.35
        Probability that a regime transition enters a trending state.
    mean_regime_len : int, default 180
        Mean duration of a regime block in bars.
    base_vol : float, default 0.0009
        Baseline standard deviation of single-step returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with open, high, low, close, and true_regime columns.
    """
    if n_bars <= 0:
        raise ValueError(f"n_bars must be positive, got {n_bars}")
    if start_price <= 0:
        raise ValueError(f"start_price must be positive, got {start_price}")
    if not (0.0 <= trend_prob <= 1.0):
        raise ValueError(f"trend_prob must be in [0, 1], got {trend_prob}")
    if mean_regime_len <= 0:
        raise ValueError(f"mean_regime_len must be positive, got {mean_regime_len}")

    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_bars, freq=freq)

    # Generate regime schedule
    regimes = np.zeros(n_bars, dtype=int)
    pos = 0
    while pos < n_bars:
        block_len = max(20, int(rng.exponential(mean_regime_len)))
        current_regime = 1 if rng.random() < trend_prob else 0
        regimes[pos : min(n_bars, pos + block_len)] = current_regime
        pos += block_len

    prices = np.empty(n_bars, dtype=float)
    prices[0] = start_price
    anchor = start_price
    drift = 0.0

    for t in range(1, n_bars):
        if regimes[t] != regimes[t - 1]:
            if regimes[t] == 1:
                direction = float(rng.choice([-1.0, 1.0]))
                drift = direction * rng.uniform(0.0004, 0.0011)
            else:
                anchor = prices[t - 1]
                drift = 0.0

        current_vol = base_vol * (1.35 if regimes[t] == 1 else 1.0)
        shock = rng.normal(0.0, current_vol)

        if regimes[t] == 1:
            step = drift + shock
        else:
            step = 0.02 * (anchor - prices[t - 1]) / prices[t - 1] + shock

        prices[t] = max(1e-6, prices[t - 1] * (1.0 + step))

    close_series = pd.Series(prices, index=index)
    open_series = close_series.shift(1).fillna(start_price)

    close_arr = close_series.to_numpy(dtype=float)
    open_arr = open_series.to_numpy(dtype=float)

    intrabar_noise = np.abs(rng.normal(0.0, base_vol, n_bars)) * close_arr
    high_vals = np.maximum(open_arr, close_arr) + intrabar_noise
    low_vals = np.maximum(1e-6, np.minimum(open_arr, close_arr) - intrabar_noise)

    return pd.DataFrame(
        {
            "open": open_arr,
            "high": high_vals,
            "low": low_vals,
            "close": close_arr,
            "true_regime": regimes,
        },
        index=index,
    )
