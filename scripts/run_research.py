"""End-to-end quantitative research execution script.

Executes complete quantitative pipeline:
1. Ingests MT5 CSV export or generates reproducible synthetic market series.
2. Constructs leak-free feature matrix and forward regime classification labels.
3. Conducts walk-forward validation with embargo against majority-class baseline.
4. Simulates and compares three regime gate configurations (None, ADX rule, ML gate)
   under a ratcheting trailing drawdown account model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly from source directory
pkg_root = Path(__file__).resolve().parents[1] / "python"
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from hquant.backtest.engine import Backtester
from hquant.data.loader import generate_synthetic_ohlc, load_mt5_csv
from hquant.features.engineering import (
    atr,
    bollinger_position,
    build_features,
    label_regime,
    rsi,
)
from hquant.models.regime import RegimeClassifier, walk_forward_validate


def mean_reversion_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate raw mean-reversion signals and corresponding target price levels.

    Entry condition: Prior bar closed outside Bollinger band with RSI stretched beyond extremes.
    Target level: Bollinger middle band (20 SMA).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Tuple of (signals series, target prices series).
    """
    close = df["close"]
    bb_pos = bollinger_position(close, 20, 2.0)
    rsi14 = rsi(close, 14)
    middle = close.rolling(20).mean()

    long_entry = (bb_pos.shift(1) < 0.0) & (rsi14.shift(1) < 30.0)
    short_entry = (bb_pos.shift(1) > 1.0) & (rsi14.shift(1) > 70.0)

    signals = pd.Series(0.0, index=df.index)
    signals[long_entry] = 1.0
    signals[short_entry] = -1.0
    return signals, middle


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="HybridQuant quantitative research and regime backtesting pipeline."
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to MT5 exported CSV bar data.")
    parser.add_argument("--balance", type=float, default=5000.0, help="Initial account balance.")
    parser.add_argument(
        "--dd-pct", type=float, default=6.0, help="Maximum trailing drawdown budget percentage."
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.40, help="Base risk per trade as percentage of equity."
    )
    parser.add_argument(
        "--bars", type=int, default=20000, help="Number of bars to generate for synthetic mode."
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for synthetic data generation."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional file path to export backtest results table.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute research pipeline."""
    args = parse_args()

    if args.csv:
        df = load_mt5_csv(args.csv)
        source_desc = f"MT5 CSV: {args.csv}"
    else:
        df = generate_synthetic_ohlc(n_bars=args.bars, seed=args.seed)
        source_desc = f"Synthetic series (seed={args.seed})"

    print("=" * 64)
    print("HYBRIDQUANT QUANTITATIVE RESEARCH RUN")
    print("=" * 64)
    print(f"Data source          : {source_desc}")
    print(f"Bar count            : {len(df):,}")
    print(f"Sample range         : {df.index[0]} -> {df.index[-1]}")
    print(f"Initial balance      : ${args.balance:,.2f}")
    print(f"Drawdown allowance   : {args.dd_pct:.1f}%")
    print()

    # 1. Feature Construction & Target Labeling
    X = build_features(df)
    y = label_regime(df, horizon=24, threshold=0.60)

    print(f"Feature set size     : {len(X.columns)} indicators")
    print(f"Target distribution  : Trend = {y.mean():.1%}, Range = {1.0 - y.mean():.1%}")
    print()

    # 2. Walk-Forward Cross-Validation
    report = walk_forward_validate(X, y, n_splits=5, embargo=24)
    print(report.summary())
    print()

    if report.edge_over_baseline < 0.02:
        print("Model Decision: Classifier does not demonstrate adequate edge over")
        print("baseline. Rule-based ADX filter remains primary recommended gate.")
    else:
        print("Model Decision: Classifier demonstrates positive edge over baseline.")
        print("Proceeding to comparative backtesting to evaluate economic utility.")
    print()

    # 3. Out-of-Sample Regime Gate Evaluation
    frame = pd.concat([X, y.rename("_target")], axis=1).dropna()
    X_clean = frame.drop(columns="_target")
    y_clean = frame["_target"].astype(int)

    split_idx = int(len(X_clean) * 0.60)
    clf = RegimeClassifier().fit(X_clean.iloc[:split_idx], y_clean.iloc[:split_idx])

    oos_index = X_clean.index[split_idx:]
    proba_trend = pd.Series(
        clf.predict_proba_trend(X_clean.iloc[split_idx:]),
        index=oos_index,
        name="proba_trend",
    )

    df_oos = df.loc[oos_index]
    signals_all, targets_all = mean_reversion_signals(df.loc[X_clean.index])
    signals_oos = signals_all.loc[oos_index]
    targets_oos = targets_all.loc[oos_index]
    atr_oos = atr(df.loc[X_clean.index], 14).loc[oos_index]
    adx_oos = X_clean.loc[oos_index, "adx14"]

    gates = {
        "No Gate (Unfiltered)": pd.Series(True, index=oos_index),
        "ADX Filter (< 20.0)": adx_oos < 20.0,
        "ML Gate (p_trend < 0.40)": proba_trend < 0.40,
    }

    print("=" * 64)
    print("OUT-OF-SAMPLE COMPARATIVE BACKTEST (Mean Reversion)")
    print(f"Evaluation window    : {len(oos_index):,} bars ({oos_index[0]} to {oos_index[-1]})")
    print("=" * 64)

    rows: list[dict[str, object]] = []
    for name, gate in gates.items():
        gated_signals = signals_oos.where(gate.reindex(signals_oos.index).fillna(False), 0.0)
        bt = Backtester(
            initial_balance=args.balance,
            dd_pct=args.dd_pct,
            risk_pct=args.risk_pct,
        )
        res = bt.run(df_oos, gated_signals, atr_oos, targets=targets_oos)
        m = res.metrics()
        rows.append(
            {
                "Gate Configuration": name,
                "Trades": m["n_trades"],
                "Win Rate": f"{m['win_rate']:.3f}" if np.isfinite(m["win_rate"]) else "N/A",
                "Profit Factor": f"{m['profit_factor']:.3f}"
                if np.isfinite(m["profit_factor"])
                else "inf",
                "Return (%)": f"{m['total_return_pct']:+.2f}%",
                "Max DD (%)": f"{m['max_drawdown_pct']:.2f}%",
                "Breached": "YES" if m["breached"] else "NO",
            }
        )

    results_df = pd.DataFrame(rows)
    print()
    print(results_df.to_string(index=False))
    print()
    print("Performance Evaluation Note:")
    print("Under a ratcheting equity floor, capital preservation and survival take")
    print("precedence over gross return. Examine breach status prior to return metrics.")

    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults successfully exported to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
