"""
Ensemble anomaly detector combining Isolation Forest and Autoencoder.

Fuses anomaly scores from both methods using configurable weighting
and voting strategies for more robust detection than either model alone.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.autoencoder import AnomalyDetectorAE
from src.isolation_forest_model import IsolationForestDetector

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """Combines Isolation Forest and Autoencoder anomaly scores.

    Provides weighted score fusion, voting-based detection, and threshold
    calibration for the combined model.

    Voting strategies:
    - 'weighted_average': Weighted mean of normalized scores.
    - 'unanimous': Both models must flag an anomaly.
    - 'any': Either model flagging is sufficient.
    """

    VALID_STRATEGIES = {"weighted_average", "unanimous", "any"}

    def __init__(
        self,
        config_path: Optional[str] = None,
        autoencoder_weight: float = 0.6,
        isolation_forest_weight: float = 0.4,
        voting_strategy: str = "weighted_average",
    ):
        self.config = self._load_config(config_path)

        ensemble_config = self.config.get("ensemble", {})
        self.ae_weight = ensemble_config.get("autoencoder_weight", autoencoder_weight)
        self.if_weight = ensemble_config.get("isolation_forest_weight", isolation_forest_weight)
        self.voting_strategy = ensemble_config.get("voting_strategy", voting_strategy)

        if self.voting_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid voting strategy: {self.voting_strategy}. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )

        total_weight = self.ae_weight + self.if_weight
        self.ae_weight /= total_weight
        self.if_weight /= total_weight

        ae_config = self.config.get("autoencoder", {})
        if_config = self.config.get("isolation_forest", {})

        self.autoencoder = AnomalyDetectorAE(
            encoder_layers=ae_config.get("encoder_layers", [128, 64, 32, 16]),
            decoder_layers=ae_config.get("decoder_layers"),
            dropout=ae_config.get("dropout", 0.2),
            learning_rate=ae_config.get("learning_rate", 0.001),
            epochs=ae_config.get("epochs", 100),
            batch_size=ae_config.get("batch_size", 256),
            threshold_percentile=ae_config.get("threshold_percentile", 95),
        )

        self.isolation_forest = IsolationForestDetector(
            n_estimators=if_config.get("n_estimators", 200),
            max_samples=if_config.get("max_samples", 0.8),
            contamination=if_config.get("contamination", 0.05),
            random_state=if_config.get("random_state", 42),
        )

        self.threshold: Optional[float] = None
        self._ae_score_min: Optional[float] = None
        self._ae_score_max: Optional[float] = None
        self._if_score_min: Optional[float] = None
        self._if_score_max: Optional[float] = None
        self._is_fitted = False

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config not found: %s", config_path)
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def fit(
        self,
        X: pd.DataFrame,
        validation_fraction: float = 0.1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train both anomaly detection models.

        Args:
            X: Training feature matrix (numeric columns only).
            validation_fraction: Fraction for autoencoder validation.
            verbose: Whether to log training progress.

        Returns:
            Dictionary with training history and calibration info.
        """
        X_numeric = X.select_dtypes(include=[np.number])

        logger.info("Training Isolation Forest on %d samples...", len(X_numeric))
        self.isolation_forest.fit(X_numeric)

        logger.info("Training Autoencoder on %d samples...", len(X_numeric))
        ae_history = self.autoencoder.fit(
            X_numeric, validation_fraction=validation_fraction, verbose=verbose
        )

        self._calibrate_score_ranges(X_numeric)
        self._is_fitted = True

        return {
            "autoencoder_history": ae_history,
            "ae_threshold": self.autoencoder.threshold,
            "if_threshold": self.isolation_forest.threshold,
            "ensemble_threshold": self.threshold,
        }

    def _calibrate_score_ranges(self, X: pd.DataFrame) -> None:
        """Compute min/max score ranges for normalization."""
        ae_scores = self.autoencoder.predict(X)
        if_scores = self.isolation_forest.predict(X)

        self._ae_score_min = float(np.min(ae_scores))
        self._ae_score_max = float(np.max(ae_scores))
        self._if_score_min = float(np.min(if_scores))
        self._if_score_max = float(np.max(if_scores))

        calibration_scores = self._fuse_scores(ae_scores, if_scores)
        self.threshold = float(np.percentile(calibration_scores, 95))

        logger.info(
            "Score calibration complete. AE range: [%.4f, %.4f], "
            "IF range: [%.4f, %.4f], Ensemble threshold: %.6f",
            self._ae_score_min, self._ae_score_max,
            self._if_score_min, self._if_score_max,
            self.threshold,
        )

    def _normalize_score(
        self, score: np.ndarray, score_min: float, score_max: float
    ) -> np.ndarray:
        """Min-max normalize scores to [0, 1] range."""
        range_val = score_max - score_min
        if range_val < 1e-10:
            return np.zeros_like(score)
        return (score - score_min) / range_val

    def _fuse_scores(
        self, ae_scores: np.ndarray, if_scores: np.ndarray
    ) -> np.ndarray:
        """Combine normalized scores using weighted average."""
        ae_norm = self._normalize_score(
            ae_scores, self._ae_score_min, self._ae_score_max
        )
        if_norm = self._normalize_score(
            if_scores, self._if_score_min, self._if_score_max
        )

        fused = self.ae_weight * ae_norm + self.if_weight * if_norm
        return fused

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute ensemble anomaly scores.

        Args:
            X: Feature matrix.

        Returns:
            Array of fused anomaly scores (higher = more anomalous).
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

        X_numeric = X.select_dtypes(include=[np.number])
        ae_scores = self.autoencoder.predict(X_numeric)
        if_scores = self.isolation_forest.predict(X_numeric)

        return self._fuse_scores(ae_scores, if_scores)

    def predict_detailed(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-model and ensemble scores for each sample.

        Returns a DataFrame with individual model scores, normalized
        scores, and the fused ensemble score.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

        X_numeric = X.select_dtypes(include=[np.number])
        ae_scores = self.autoencoder.predict(X_numeric)
        if_scores = self.isolation_forest.predict(X_numeric)

        ae_norm = self._normalize_score(ae_scores, self._ae_score_min, self._ae_score_max)
        if_norm = self._normalize_score(if_scores, self._if_score_min, self._if_score_max)
        fused = self.ae_weight * ae_norm + self.if_weight * if_norm

        return pd.DataFrame({
            "ae_score": ae_scores,
            "if_score": if_scores,
            "ae_normalized": ae_norm,
            "if_normalized": if_norm,
            "ensemble_score": fused,
            "ae_anomaly": (ae_scores > self.autoencoder.threshold).astype(int),
            "if_anomaly": self.isolation_forest.detect(X_numeric),
        })

    def detect(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Generate binary anomaly labels using the configured voting strategy.

        Args:
            X: Feature matrix.
            threshold: Override ensemble threshold (for weighted_average strategy).

        Returns:
            Array of binary labels (1 = anomaly, 0 = normal).
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

        X_numeric = X.select_dtypes(include=[np.number])

        if self.voting_strategy == "weighted_average":
            scores = self.predict(X_numeric)
            cutoff = threshold if threshold is not None else self.threshold
            return (scores > cutoff).astype(int)

        elif self.voting_strategy == "unanimous":
            ae_labels = self.autoencoder.detect(X_numeric)
            if_labels = self.isolation_forest.detect(X_numeric)
            return ((ae_labels == 1) & (if_labels == 1)).astype(int)

        elif self.voting_strategy == "any":
            ae_labels = self.autoencoder.detect(X_numeric)
            if_labels = self.isolation_forest.detect(X_numeric)
            return ((ae_labels == 1) | (if_labels == 1)).astype(int)

        raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_metric: str = "f1",
    ) -> float:
        """Calibrate the ensemble threshold on a labeled validation set.

        Searches for the threshold that maximizes the specified metric.

        Args:
            X_val: Validation features.
            y_val: True anomaly labels (1 = anomaly).
            target_metric: Metric to optimize ('f1', 'precision', 'recall').

        Returns:
            The optimal threshold value.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        scores = self.predict(X_val)
        y_true = y_val.values if isinstance(y_val, pd.Series) else y_val

        thresholds = np.linspace(
            np.percentile(scores, 80),
            np.percentile(scores, 99),
            100,
        )

        best_threshold = self.threshold
        best_metric_value = 0.0

        for t in thresholds:
            y_pred = (scores > t).astype(int)

            if target_metric == "f1":
                metric_value = f1_score(y_true, y_pred, zero_division=0)
            elif target_metric == "precision":
                metric_value = precision_score(y_true, y_pred, zero_division=0)
            elif target_metric == "recall":
                metric_value = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {target_metric}")

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = float(t)

        self.threshold = best_threshold
        logger.info(
            "Calibrated threshold to %.6f (best %s: %.4f)",
            best_threshold, target_metric, best_metric_value,
        )
        return best_threshold

    def get_model_agreement(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analyze agreement between the two detection models.

        Returns:
            DataFrame with agreement statistics.
        """
        X_numeric = X.select_dtypes(include=[np.number])
        ae_labels = self.autoencoder.detect(X_numeric)
        if_labels = self.isolation_forest.detect(X_numeric)

        both_normal = ((ae_labels == 0) & (if_labels == 0)).sum()
        both_anomaly = ((ae_labels == 1) & (if_labels == 1)).sum()
        ae_only = ((ae_labels == 1) & (if_labels == 0)).sum()
        if_only = ((ae_labels == 0) & (if_labels == 1)).sum()

        total = len(ae_labels)

        return pd.DataFrame({
            "category": ["Both Normal", "Both Anomaly", "AE Only", "IF Only"],
            "count": [both_normal, both_anomaly, ae_only, if_only],
            "percentage": [
                both_normal / total * 100,
                both_anomaly / total * 100,
                ae_only / total * 100,
                if_only / total * 100,
            ],
        })

    def save(self, directory: str) -> None:
        """Save all model artifacts to a directory."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.autoencoder.save(str(save_dir / "autoencoder.joblib"))
        self.isolation_forest.save(str(save_dir / "isolation_forest.joblib"))

        import joblib
        ensemble_state = {
            "ae_weight": self.ae_weight,
            "if_weight": self.if_weight,
            "voting_strategy": self.voting_strategy,
            "threshold": self.threshold,
            "ae_score_min": self._ae_score_min,
            "ae_score_max": self._ae_score_max,
            "if_score_min": self._if_score_min,
            "if_score_max": self._if_score_max,
        }
        joblib.dump(ensemble_state, str(save_dir / "ensemble_state.joblib"))
        logger.info("Ensemble detector saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load all model artifacts from a directory."""
        import joblib

        load_dir = Path(directory)
        self.autoencoder.load(str(load_dir / "autoencoder.joblib"))
        self.isolation_forest.load(str(load_dir / "isolation_forest.joblib"))

        state = joblib.load(str(load_dir / "ensemble_state.joblib"))
        self.ae_weight = state["ae_weight"]
        self.if_weight = state["if_weight"]
        self.voting_strategy = state["voting_strategy"]
        self.threshold = state["threshold"]
        self._ae_score_min = state["ae_score_min"]
        self._ae_score_max = state["ae_score_max"]
        self._if_score_min = state["if_score_min"]
        self._if_score_max = state["if_score_max"]
        self._is_fitted = True

        logger.info("Ensemble detector loaded from %s", directory)
