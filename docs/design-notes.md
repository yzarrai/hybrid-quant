# Design Notes

Technical design decisions, mathematical rationales, and alternatives evaluated during development.

## 1. Priority of Drawdown Architecture

Standard algorithmic strategy development typically optimizes alpha signals before adding risk controls. Under funded prop-firm rules with an equity-based trailing drawdown that never freezes, this sequence is inverted: the binding constraint is account survival. A model with high annualized gross return that breaches its ratcheting floor returns zero.

### Static vs. Ratcheting Drawdown Floors

1. **Static Floor Model ($5,000 account, 6% max drawdown):**
   ```
   Floor = $4,700 (fixed permanently)
   ```
2. **Unfrozen Equity Trailing Floor Model:**
   ```
   Floor = max(Equity_observed) - $300
   ```

Under the trailing model, every incremental unrealized high moves the breach boundary upward permanently. Reaching $5,400 in floating equity fixes the floor at $5,100 (strictly above initial capital). A subsequent drawdown of $300 terminates the account despite a positive realized balance.

### Engineering Consequences

- **Primary Mean Reversion Focus**: Rapid profit realization to avoid prolonged open floating exposure.
- **Partial Profit Locking (1.0R)**: Converts floating gains into realized balance, increasing distance to the ratcheting floor.
- **Continuous ATR Trailing Stop (from 0.5R)**: Defends accrued gains tick-by-tick.
- **Tick-by-Tick Position Management**: Prevents intrabar retracements from tripping high-water mark ratchets.
- **Pre-Breach Circuit Breaker (60% Budget Consumption)**: Disables new entries before boundary failure occurs.

---

## 2. Double-Capped Position Sizing

Fixed fractional equity sizing risks a constant percentage of equity:
```
MoneyAtRisk = Equity * RiskPercent
```

When an account is in drawdown, equity remains close to baseline while remaining room to the breach floor decays rapidly. Sizing solely on equity results in excessive risk near the floor.

HybridQuant uses a double-capped structure:
```
MoneyAtRisk = min(
    Equity * RiskPercent,
    RoomToFloor * MaxRoomFraction
)
```

With `MaxRoomFraction = 0.25`, no single trade can risk more than 25% of the remaining drawdown capacity. As the account approaches the floor, lot size automatically decays toward zero.

### Lot Normalization

If computed lot size is lower than broker minimum volume (`SYMBOL_VOLUME_MIN`), `NormalizeLots` returns `0.0` (skipping the trade) rather than rounding up to minimum lot size, preventing inadvertent risk over-allocation on small accounts.

---

## 3. Feature Engineering and Causality

All indicator features are scale-free and strictly lagged:

| Feature Identifier | Description | Mathematical Property |
|---|---|---|
| `eff_ratio20`, `eff_ratio50` | Kaufman Efficiency Ratio | Net displacement divided by total path; bounded in [0, 1] |
| `hurst64` | Variance-Ratio Hurst Proxy | Ratio of multi-step to single-step variance |
| `adx14` | Average Directional Index | Trend strength baseline; bounded in [0, 100] |
| `bb_pos` | Bollinger Band Position | Scale-free relative envelope position |
| `atr_norm`, `atr_ratio` | Normalized Volatility | Volatility relative to price and trailing average |
| `dist_ma50_atr` | Distance from Trend Mean | Standardized ATR distance from 50 SMA |
| `rv10`, `rv50`, `rv_ratio` | Realized Volatility Structure | Multi-horizon return volatility term structure |

### Leakage Enforcement

Features are lagged uniformly by `shift=1` in `build_features`. Truncating data frames produces identical historical feature rows, verifying absence of lookahead peeking.

---

## 4. Target Formulation and Validation

The target label is the forward efficiency ratio over 24 bars thresholded at 0.60, identifying structural persistence rather than noisy single-step price returns.

Validation uses expanding-window `TimeSeriesSplit` with an embargo gap equal to the 24-bar label horizon. The embargo drops training bars whose future labeling window overlaps with the test set, preventing out-of-sample data leakage.
