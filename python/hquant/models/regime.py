"""Regime classification and walk-forward cross-validation.

Implements gradient-boosted tree classification on engineered price features
using expanding-window walk-forward splits with strict embargo gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(slots=True)
class FoldResult:
    """Performance metrics for an individual cross-validation fold."""

    fold: int
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    baseline_accuracy: float


@dataclass(slots=True)
class WalkForwardReport:
    """Summary of walk-forward out-of-sample validation."""

    folds: list[FoldResult] = field(default_factory=list)
    feature_importance: pd.Series | None = None

    @property
    def mean_accuracy(self) -> float:
        """Calculate mean test accuracy across all evaluated folds."""
        return float(np.mean([f.accuracy for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_auc(self) -> float:
        """Calculate mean ROC AUC across all evaluated folds."""
        return float(np.mean([f.auc for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_baseline(self) -> float:
        """Calculate mean majority-class baseline accuracy across all folds."""
        return (
            float(np.mean([f.baseline_accuracy for f in self.folds]))
            if self.folds
            else float("nan")
        )

    @property
    def edge_over_baseline(self) -> float:
        """Excess accuracy of the model over the unconditional majority-class baseline."""
        return self.mean_accuracy - self.mean_baseline

    def to_frame(self) -> pd.DataFrame:
        """Convert fold results to a structured pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "fold": f.fold,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "accuracy": f.accuracy,
                    "auc": f.auc,
                    "baseline_accuracy": f.baseline_accuracy,
                }
                for f in self.folds
            ]
        )

    def summary(self) -> str:
        """Generate human-readable tabular validation summary."""
        lines = [
            "Walk-Forward Regime Classification Validation",
            "=" * 48,
            f"Folds evaluated      : {len(self.folds)}",
            f"Mean accuracy        : {self.mean_accuracy:.4f}",
            f"Majority baseline    : {self.mean_baseline:.4f}",
            f"Edge over baseline   : {self.edge_over_baseline:+.4f}",
            f"Mean ROC AUC         : {self.mean_auc:.4f}",
        ]
        if self.feature_importance is not None and not self.feature_importance.empty:
            lines.append("")
            lines.append("Feature Importance (Mean Across Folds)")
            lines.append("-" * 48)
            for name, value in self.feature_importance.head(10).items():
                lines.append(f"  {str(name):<20} {float(value):.4f}")
        return "\n".join(lines)


class RegimeClassifier:
    """Gradient-boosted decision tree classifier for market regime identification."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "random_state": random_state,
        }
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RegimeClassifier:
        """Fit model on feature matrix and target labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names.
        y : pd.Series | np.ndarray
            Binary regime labels.

        Returns
        -------
        RegimeClassifier
            Self instance.
        """
        self.feature_names_ = list(X.columns)
        y_arr = np.asarray(y).ravel()
        self.model.fit(X.values, y_arr)
        return self

    def predict_proba_trend(self, X: pd.DataFrame) -> np.ndarray:
        """Predict out-of-sample probability of a trending regime.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            1D array of estimated trending regime probabilities.
        """
        proba = self.model.predict_proba(X.values)[:, 1]
        return np.asarray(proba, dtype=float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary regime class given a probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float, default 0.5
            Decision threshold for class 1 (trend).

        Returns
        -------
        np.ndarray
            Binary predicted labels (1 = trend, 0 = range).
        """
        return (self.predict_proba_trend(X) >= threshold).astype(int)

    @property
    def importance(self) -> pd.Series:
        """Return Series of Gini feature importances sorted descending."""
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(
            ascending=False
        )


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 24,
    **model_kwargs: Any,
) -> WalkForwardReport:
    """Execute chronological walk-forward validation with an embargo gap.

    Parameters
    ----------
    X : pd.DataFrame
        Lagged feature matrix.
    y : pd.Series
        Forward regime label series.
    n_splits : int, default 5
        Number of expanding-window folds.
    embargo : int, default 24
        Number of bars to purge from the end of each training split to prevent
        overlap with the test set's label lookahead window.
    **model_kwargs
        Keyword arguments forwarded to RegimeClassifier.

    Returns
    -------
    WalkForwardReport
        Aggregated report with individual fold metrics and feature importances.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    frame = pd.concat([X, y.rename("_target")], axis=1).dropna()
    if frame.empty:
        raise ValueError("No valid rows survive after dropping missing values.")

    X_clean = frame.drop(columns="_target")
    y_clean = frame["_target"].astype(int)

    report = WalkForwardReport()
    importances: list[pd.Series] = []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_clean), start=1):
        if embargo > 0:
            train_idx = train_idx[: max(0, len(train_idx) - embargo)]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_train, y_train = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
        X_test, y_test = X_clean.iloc[test_idx], y_clean.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        clf = RegimeClassifier(**model_kwargs).fit(X_train, y_train)
        proba = clf.predict_proba_trend(X_test)
        preds = (proba >= 0.5).astype(int)

        majority = y_train.mode().iloc[0]
        baseline = accuracy_score(y_test, np.full(len(y_test), majority))

        report.folds.append(
            FoldResult(
                fold=fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
                accuracy=float(accuracy_score(y_test, preds)),
                auc=float(roc_auc_score(y_test, proba)),
                baseline_accuracy=float(baseline),
            )
        )
        importances.append(clf.importance)

    if importances:
        report.feature_importance = (
            pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        )
    return report
