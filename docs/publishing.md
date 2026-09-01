# Repository Staging and Publication Guide

Instructions for initializing and staging git history.

## 1. Remote Initialization

Create an empty repository on GitHub without adding template files (README, .gitignore, license).

Recommended repository topics:
`algorithmic-trading`, `mql5`, `metatrader5`, `quantitative-finance`, `machine-learning`, `backtesting`, `risk-management`

## 2. Structured Commit Sequence

```bash
git init
git branch -M main

# 1. Risk Manager
git add .gitignore mql5/Include/HybridQuant/RiskManager.mqh
git commit -m "Add risk manager with ratcheting drawdown floor and double-cap sizing"

# 2. Strategy Engine
git add mql5/Include/HybridQuant/Strategies.mqh
git commit -m "Add regime-gated strategy engine for mean reversion and trend following"

# 3. Expert Advisor
git add mql5/Experts/HybridQuantEA.mq5
git commit -m "Add multi-symbol EA with tick-level position management and trailing stops"

# 4. Python Core Package
git add python/ pyproject.toml requirements.txt
git commit -m "Add hquant Python package for feature engineering, modeling, and backtesting"

# 5. CLI and Verification Suite
git add scripts/ tests/
git commit -m "Add research CLI runner and automated pytest verification suite"

# 6. CI and Documentation
git add .github/ docs/ README.md
git commit -m "Add GitHub Actions CI pipeline and technical architecture documentation"
```

## 3. Remote Push

```bash
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```
