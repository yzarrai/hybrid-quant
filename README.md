# HybridQuant

[![CI](https://github.com/yzarrai/hybrid-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/yzarrai/hybrid-quant/actions/workflows/ci.yml)

A regime-switching quantitative trading system and research framework for MetaTrader 5, engineered for funded prop accounts with an **equity-based trailing drawdown floor that never freezes**.

## Core Architecture

The system comprises two synchronized layers:

- `mql5/`  Production execution layer in MQL5. Rule-based, deterministic, tick-managed.
- `python/`  Quantitative research layer in Python (`hquant`). Scale-free feature engineering, walk-forward regime classifier with embargo, and event-driven backtester simulating the ratcheting equity floor.

```
hybridquant/
├── .github/
│   └── workflows/ci.yml              # Multi-version CI workflow
├── docs/
│   ├── design-notes.md               # Mathematical rationales & architectural decisions
│   └── publishing.md                 # Publication and staging guide
├── mql5/
│   ├── Experts/
│   │   └── HybridQuantEA.mq5         # Multi-symbol entry point, tick-level trade manager
│   └── Include/HybridQuant/
│       ├── RiskManager.mqh           # Trailing floor accounting & double-capped sizing
│       └── Strategies.mqh            # Signal generators and higher-timeframe regime filter
├── python/
│   └── hquant/                       # Core research package (PEP 561 typed)
│       ├── data/loader.py            # MT5 CSV parser & synthetic regime simulator
│       ├── features/engineering.py   # Leak-free indicator construction
│       ├── models/regime.py          # Gradient-boosted classifier & walk-forward validation
│       └── backtest/engine.py        # Event-driven backtester with floor accounting
├── scripts/
│   └── run_research.py               # Research CLI runner
├── tests/
│   ├── test_core.py                  # End-to-end integration & causality tests
│   ├── test_loader.py                # Ingestion & synthetic simulation tests
│   ├── test_features.py              # Feature bound & integrity tests
│   ├── test_models.py                # Classifier & cross-validation tests
│   └── test_engine.py                # Backtester & accounting tests
├── pyproject.toml                    # PEP 621 package & tool configurations
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Drawdown Accounting and Sizing Mechanics

Under a static drawdown constraint ($5,000 balance, 6% allowance), the account is breached only if equity reaches $4,700.

Under an unfrozen trailing drawdown floor, every new equity high ratchets the floor upward permanently:

$$\text{Floor}_t = \max_{0 \le s \le t} (\text{Equity}_s) - \text{Budget}$$

Floating profit increases the floor level, making subsequent retracements lethal even when net realized balance remains positive.

### Risk Controls

- **Double-Capped Position Sizing**:
  $$\text{Capital at Risk} = \min\left(\text{Equity} \times \text{RiskPct},\; \text{RoomToFloor} \times \text{MaxRoomFraction}\right)$$
  Size automatically decays toward zero as the account approaches the floor.
- **Partial Profit Taking (1.0R)**: Converts floating gains into realized balance, increasing distance to the floor.
- **Continuous ATR Trailing Stop (from 0.5R)**: Defends unrealized gains tick-by-tick.
- **Circuit Breaker**: Halts new entries when 60% of the total drawdown budget is consumed.

---

## Machine Learning Regime Gate

The regime classifier predicts whether the forward 24-bar window is trending or ranging. It does not predict trade direction; directional signals are determined by deterministic rules.

Validation uses expanding-window `TimeSeriesSplit` with an embargo gap equal to the 24-bar labeling horizon:

```python
def test_features_contain_no_future_information(data):
    full = build_features(data)
    partial = build_features(data.iloc[:3000])
    pd.testing.assert_frame_equal(full.loc[overlap], partial.loc[overlap])
```

---

## Quick Start and Verification

### Installation

```bash
git clone https://github.com/yzarrai/hybrid-quant.git
cd hybrid-quant
pip install -r requirements.txt
```

### Research Pipeline Execution

```bash
# Run on deterministic synthetic series
python scripts/run_research.py --bars 20000 --seed 7

# Run on MT5 CSV export
python scripts/run_research.py --csv data/EURUSD_H1.csv
```

### Automated Testing and Linting

```bash
# Run pytest test suite
pytest tests/ -v --tb=short

# Code quality checks
ruff check python/ tests/ scripts/
ruff format --check python/ tests/ scripts/
mypy python/
```

---

## MQL5 Deployment

1. Copy `mql5/Include/HybridQuant/` into your MetaTrader 5 terminal directory: `MQL5/Include/HybridQuant/`.
2. Copy `mql5/Experts/HybridQuantEA.mq5` into `MQL5/Experts/`.
3. Open MetaEditor, compile `HybridQuantEA.mq5`, and verify compilation with zero errors.
4. Run in the MetaTrader 5 Strategy Tester before live deployment.

---

## Limitations and Disclosures

- **No Live Track Record**: Results reflect backtested and simulated data.
- **Synthetic Data**: Synthetic market regimes serve for structural pipeline verification and unit testing.
- **Execution Modeling**: Backtests assume pessimistic intrabar fills and flat execution friction. Live trading is subject to slippage, spread expansion, and latency.
- **Non-Advisory**: This codebase is provided for research and quantitative engineering purposes only.
