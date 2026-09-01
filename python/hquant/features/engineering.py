"""Feature engineering for regime classification and volatility modeling.

Calculates scale-free, leak-free indicator features from OHLC time series.
All feature outputs are lagged to ensure strict causality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range across high, low, and prior close.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.

    Returns
    -------
    pd.Series
        True range series.
    """
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Wilder's Average True Range (ATR).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.
    period : int, default 14
        Smoothing window length.

    Returns
    -------
    pd.Series
        Smoothed ATR series.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return _true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Wilder's Average Directional Index (ADX).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with high, low, close columns.
    period : int, default 14
        Directional movement and ADX smoothing window.

    Returns
    -------
    pd.Series
        ADX values bounded in [0, 100].
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_smooth = _true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()
    tr_safe = tr_smooth.replace(0.0, np.nan)

    plus_di = (
        100.0
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / tr_safe
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / tr_safe
    )

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price series (typically close prices).
    period : int, default 14
        Smoothing window length.

    Returns
    -------
    pd.Series
        RSI values bounded in [0, 100].
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()

    # Handle zero loss condition cleanly without division by zero warning
    rs = gain / loss.replace(0.0, np.nan)
    res = 100.0 - (100.0 / (1.0 + rs))
    fallback = pd.Series(np.where(gain > 0, 100.0, 50.0), index=series.index)
    return res.fillna(fallback)


def bollinger_position(series: pd.Series, period: int = 20, dev: float = 2.0) -> pd.Series:
    """Calculate Bollinger Band relative position score.

    Values around 0.0 represent lower band, 0.5 middle band, 1.0 upper band.

    Parameters
    ----------
    series : pd.Series
        Price series.
    period : int, default 20
        Moving average and rolling standard deviation period.
    dev : float, default 2.0
        Standard deviation multiplier.

    Returns
    -------
    pd.Series
        Dimensionless position within the band envelope.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    mid = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    upper = mid + dev * sd
    lower = mid - dev * sd
    width = (upper - lower).replace(0.0, np.nan)
    return (series - lower) / width


def efficiency_ratio(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Kaufman Efficiency Ratio: net directional change / total price path.

    Approaches 1.0 during strong linear trends; approaches 0.0 in ranging market states.

    Parameters
    ----------
    series : pd.Series
        Price series.
    period : int, default 20
        Lookback window.

    Returns
    -------
    pd.Series
        Efficiency ratio bounded in [0, 1].
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    direction = (series - series.shift(period)).abs()
    path = series.diff().abs().rolling(period).sum()
    return direction / path.replace(0.0, np.nan)


def hurst_proxy(series: pd.Series, period: int = 64) -> pd.Series:
    """Compute rolling variance-ratio proxy for the Hurst exponent.

    Hurst > 0.5 indicates trending/persistence; Hurst < 0.5 indicates mean reversion.

    Parameters
    ----------
    series : pd.Series
        Price series.
    period : int, default 64
        Rolling variance window.

    Returns
    -------
    pd.Series
        Rolling estimated Hurst exponent.
    """
    if period <= 1:
        raise ValueError(f"period must be greater than 1, got {period}")

    r1 = series.pct_change()
    r2 = series.pct_change(2)
    var1 = r1.rolling(period).var(ddof=0)
    var2 = r2.rolling(period).var(ddof=0)
    denom = 2.0 * var1.replace(0.0, np.nan)
    ratio = var2 / denom
    res = 0.5 * np.log2(ratio.clip(lower=1e-9)) + 0.5
    return pd.Series(res, index=series.index)


def build_features(
    df: pd.DataFrame,
    shift: int = 1,
    required_cols: Sequence[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Construct complete scale-free feature matrix for regime classification.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.
    shift : int, default 1
        Lag applied to all features to enforce strict zero lookahead.
    required_cols : Sequence[str]
        Required column names in input DataFrame.

    Returns
    -------
    pd.DataFrame
        Lagged feature matrix with DatetimeIndex matching input.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature construction: {missing}")

    close = df["close"]
    atr14 = atr(df, 14)

    feats = pd.DataFrame(index=df.index)
    feats["adx14"] = adx(df, 14)
    feats["rsi14"] = rsi(close, 14)
    feats["bb_pos"] = bollinger_position(close, 20, 2.0)
    feats["eff_ratio20"] = efficiency_ratio(close, 20)
    feats["eff_ratio50"] = efficiency_ratio(close, 50)
    feats["hurst64"] = hurst_proxy(close, 64)

    # Volatility normalizations
    feats["atr_norm"] = atr14 / close
    atr_rolling_mean = atr14.rolling(50).mean().replace(0.0, np.nan)
    feats["atr_ratio"] = atr14 / atr_rolling_mean

    # Mean deviation in ATR units
    ma50 = close.rolling(50).mean()
    feats["dist_ma50_atr"] = (close - ma50) / atr14.replace(0.0, np.nan)

    # Realized volatility term structure
    returns = close.pct_change()
    feats["rv10"] = returns.rolling(10).std(ddof=0)
    feats["rv50"] = returns.rolling(50).std(ddof=0)
    feats["rv_ratio"] = feats["rv10"] / feats["rv50"].replace(0.0, np.nan)

    if shift > 0:
        return feats.shift(shift)
    return feats


def label_regime(
    df: pd.DataFrame,
    horizon: int = 24,
    threshold: float = 0.60,
) -> pd.Series:
    """Generate forward-looking regime label series based on future efficiency ratio.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with close prices.
    horizon : int, default 24
        Forward window length in bars.
    threshold : float, default 0.60
        Efficiency ratio threshold above which market state is labeled trending (1.0).

    Returns
    -------
    pd.Series
        Binary regime labels (1.0 = trend, 0.0 = range) with NaN for the last `horizon` bars.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    close = df["close"]
    fwd_direction = (close.shift(-horizon) - close).abs()
    fwd_path = close.diff().abs().shift(-1).rolling(horizon).sum().shift(-horizon + 1)
    fwd_er = fwd_direction / fwd_path.replace(0.0, np.nan)
    return (fwd_er > threshold).astype(float).where(fwd_er.notna())
