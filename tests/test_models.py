"""Unit tests for regime classification models and cross-validation."""

from __future__ import annotations

import pandas as pd
import pytest
from hquant.data.loader import generate_synthetic_ohlc
from hquant.features.engineering import build_features, label_regime
from hquant.models.regime import RegimeClassifier, walk_forward_validate


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    ohlc = generate_synthetic_ohlc(n_bars=800, seed=42)
    X = build_features(ohlc)
    y = label_regime(ohlc, horizon=12)
    return X, y


def test_regime_classifier_fit_predict(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verify RegimeClassifier fit and inference interface."""
    X, y = dataset
    clean = pd.concat([X, y.rename("_target")], axis=1).dropna()
    X_clean = clean.drop(columns="_target")
    y_clean = clean["_target"]

    clf = RegimeClassifier(n_estimators=20, max_depth=2, random_state=42)
    clf.fit(X_clean, y_clean)

    probas = clf.predict_proba_trend(X_clean)
    assert len(probas) == len(X_clean)
    assert ((probas >= 0.0) & (probas <= 1.0)).all()

    preds = clf.predict(X_clean, threshold=0.5)
    assert len(preds) == len(X_clean)
    assert set(preds).issubset({0, 1})

    imp = clf.importance
    assert len(imp) == len(X_clean.columns)
    assert (imp >= 0.0).all()


def test_walk_forward_validate_report(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verify walk-forward validation execution and summary output."""
    X, y = dataset
    report = walk_forward_validate(X, y, n_splits=3, embargo=12, n_estimators=10)

    assert len(report.folds) > 0
    assert 0.0 <= report.mean_accuracy <= 1.0
    assert 0.0 <= report.mean_auc <= 1.0

    df = report.to_frame()
    assert len(df) == len(report.folds)
    assert "fold" in df.columns
    assert "accuracy" in df.columns

    summary_str = report.summary()
    assert "Walk-Forward Regime Classification Validation" in summary_str
    assert "Mean accuracy" in summary_str


def test_walk_forward_validate_invalid_splits(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verify error raised on invalid n_splits."""
    X, y = dataset
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        walk_forward_validate(X, y, n_splits=1)
