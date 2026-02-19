"""
Isolation Forest anomaly detector with feature importance and explanation.

Wraps scikit-learn's IsolationForest with additional capabilities for
threshold-based detection, permutation-based feature importance, and
per-sample anomaly explanations.
"""

import logging
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest-based anomaly detection with explainability.

    Isolates anomalies by building an ensemble of random trees. Anomalous
    points are easier to isolate (shorter average path length) because
    they differ from the majority of the data.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: float = 0.8,
        contamination: float = 0.05,
        random_state: int = 42,
        threshold: Optional[float] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._threshold = threshold

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: Optional[list[str]] = None
        self._training_scores: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        """Fit the Isolation Forest model on training data.

        Args:
            X: Training feature matrix. Can contain both normal and
               anomalous data (unsupervised).

        Returns:
            self for method chaining.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled)

        self._training_scores = self._raw_scores(X_scaled)
        self._is_fitted = True

        if self._threshold is None:
            self._threshold = float(
                np.percentile(self._training_scores, 95)
            )

        logger.info(
            "Isolation Forest fitted on %d samples with %d features. "
            "Threshold: %.6f",
            X_scaled.shape[0], X_scaled.shape[1], self._threshold,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for new data.

        Higher scores indicate greater anomalousness. The raw
        decision_function output is negated so that higher values
        correspond to more anomalous points.

        Args:
            X: Feature matrix.

        Returns:
            Array of anomaly scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        return self._raw_scores(X_scaled)

    def _raw_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute raw anomaly scores from scaled features.

        Negates the decision function so higher = more anomalous.
        """
        raw = self.model.decision_function(X_scaled)
        return -raw

    def detect(
        self, X: pd.DataFrame, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Generate binary anomaly labels.

        Args:
            X: Feature matrix.
            threshold: Override the default detection threshold.

        Returns:
            Array of binary labels (1 = anomaly, 0 = normal).
        """
        scores = self.predict(X)
        cutoff = threshold if threshold is not None else self._threshold
        if cutoff is None:
            raise RuntimeError("No threshold set. Call fit() or set threshold manually.")
        return (scores > cutoff).astype(int)

    def feature_importance(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """Compute permutation-based feature importance.

        Measures how much the anomaly score changes when each feature
        is randomly permuted, indicating its contribution to detection.

        Args:
            X: Feature matrix (should include some anomalies for meaningful results).
            y: Optional true labels for supervised importance evaluation.
            n_repeats: Number of permutation rounds.

        Returns:
            DataFrame with feature importance scores sorted by mean impact.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)

        if y is None:
            labels = self.model.predict(X_scaled)
            y_binary = (labels == -1).astype(int)
        else:
            y_binary = y.values if isinstance(y, pd.Series) else y

        result = permutation_importance(
            self.model,
            X_scaled,
            y_binary,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        feature_labels = self.feature_names or [
            f"feature_{i}" for i in range(X_scaled.shape[1])
        ]

        importance_df = pd.DataFrame({
            "feature": feature_labels,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        return importance_df

    def explain_anomaly(
        self, X: pd.DataFrame, sample_idx: int
    ) -> pd.DataFrame:
        """Explain which features contributed most to a sample's anomaly score.

        Computes a per-feature anomaly contribution by comparing the
        sample's feature values against the training distribution and
        measuring individual feature isolation difficulty.

        Args:
            X: Feature matrix containing the sample.
            sample_idx: Index of the sample to explain.

        Returns:
            DataFrame with per-feature contribution scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)

        sample = X_scaled[sample_idx]
        base_score = self._raw_scores(X_scaled[sample_idx:sample_idx + 1])[0]

        feature_labels = self.feature_names or [
            f"feature_{i}" for i in range(X_scaled.shape[1])
        ]

        contributions = []
        for feat_idx in range(X_scaled.shape[1]):
            perturbed = sample.copy()
            perturbed[feat_idx] = 0.0
            perturbed_score = self._raw_scores(perturbed.reshape(1, -1))[0]
            contribution = base_score - perturbed_score

            z_score = abs(sample[feat_idx])

            contributions.append({
                "feature": feature_labels[feat_idx],
                "value": float(X_array[sample_idx, feat_idx]),
                "scaled_value": float(sample[feat_idx]),
                "z_score": float(z_score),
                "contribution": float(contribution),
                "direction": "high" if sample[feat_idx] > 0 else "low",
            })

        explanation_df = pd.DataFrame(contributions).sort_values(
            "contribution", ascending=False, key=abs
        ).reset_index(drop=True)

        return explanation_df

    def set_threshold(
        self,
        percentile: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        """Manually set or recalibrate the detection threshold.

        Args:
            percentile: Set threshold at this percentile of training scores.
            value: Set threshold to this exact value.
        """
        if value is not None:
            self._threshold = value
        elif percentile is not None and self._training_scores is not None:
            self._threshold = float(np.percentile(self._training_scores, percentile))
        else:
            raise ValueError("Provide either percentile or value")
        logger.info("Threshold set to %.6f", self._threshold)

    @property
    def threshold(self) -> Optional[float]:
        """Current detection threshold."""
        return self._threshold

    def get_params(self) -> dict[str, Any]:
        """Return model parameters as a dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "threshold": self._threshold,
        }

    def save(self, path: str) -> None:
        """Save the fitted model, scaler, and metadata to disk."""
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "threshold": self._threshold,
            "training_scores": self._training_scores,
            "params": self.get_params(),
        }
        joblib.dump(state, path)
        logger.info("Isolation Forest detector saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk."""
        state = joblib.load(path)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self._threshold = state["threshold"]
        self._training_scores = state.get("training_scores")
        self._is_fitted = True

        params = state.get("params", {})
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.max_samples = params.get("max_samples", self.max_samples)
        self.contamination = params.get("contamination", self.contamination)

        logger.info("Isolation Forest detector loaded from %s", path)
