"""
Deep Autoencoder for anomaly detection.

Implements a configurable encoder-decoder architecture in PyTorch with
batch normalization and dropout. Anomalies are detected via high
reconstruction error on inputs the model has not learned to represent.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class DeepAutoencoder(nn.Module):
    """PyTorch autoencoder with configurable layer architecture.

    Supports batch normalization between layers, dropout for regularization,
    ReLU activations in hidden layers, and sigmoid on the output.

    Args:
        input_dim: Number of input features.
        encoder_layers: List of hidden layer sizes for the encoder.
        decoder_layers: List of hidden layer sizes for the decoder.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_layers: Optional[list[int]] = None,
        decoder_layers: Optional[list[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim

        if encoder_layers is None:
            encoder_layers = [128, 64, 32, 16]
        if decoder_layers is None:
            decoder_layers = list(reversed(encoder_layers))

        self.encoder = self._build_encoder(input_dim, encoder_layers, dropout)
        self.decoder = self._build_decoder(
            encoder_layers[-1], decoder_layers, input_dim, dropout
        )

    def _build_encoder(
        self, input_dim: int, layer_sizes: list[int], dropout: float
    ) -> nn.Sequential:
        """Construct the encoder network."""
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i < len(layer_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        latent_dim: int,
        layer_sizes: list[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        """Construct the decoder network."""
        layers = []
        prev_dim = latent_dim

        for i, hidden_dim in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i < len(layer_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to the latent representation."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size,) with MSE per sample.
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


class AnomalyDetectorAE:
    """Anomaly detection wrapper around the deep autoencoder.

    Handles training, threshold calibration, scoring, and binary
    anomaly detection using reconstruction error.
    """

    def __init__(
        self,
        encoder_layers: Optional[list[int]] = None,
        decoder_layers: Optional[list[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.encoder_layers = encoder_layers or [128, 64, 32, 16]
        self.decoder_layers = decoder_layers or list(reversed(self.encoder_layers))
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[DeepAutoencoder] = None
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self._training_errors: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        validation_fraction: float = 0.1,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the autoencoder on normal data.

        Args:
            X: Training feature matrix (assumed mostly normal data).
            validation_fraction: Fraction held out for validation.
            verbose: Whether to log training progress.

        Returns:
            Dictionary with 'train_loss' and 'val_loss' histories.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.fit_transform(X_array)

        n_val = int(len(X_scaled) * validation_fraction)
        X_train = X_scaled[n_val:]
        X_val = X_scaled[:n_val]

        input_dim = X_scaled.shape[1]
        self.model = DeepAutoencoder(
            input_dim=input_dim,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        train_tensor = torch.FloatTensor(X_train).to(self.device)
        val_tensor = torch.FloatTensor(X_val).to(self.device)

        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 15

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_target in train_loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            self.model.eval()
            with torch.no_grad():
                val_reconstructed = self.model(val_tensor)
                val_loss = criterion(val_reconstructed, val_tensor).item()

            scheduler.step(val_loss)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    if verbose:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f, LR: %.2e",
                    epoch + 1, self.epochs, avg_train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

        self._set_threshold_from_training(X_scaled)
        self._is_fitted = True

        logger.info(
            "Autoencoder training complete. Threshold: %.6f", self.threshold
        )
        return history

    def _set_threshold_from_training(self, X_scaled: np.ndarray) -> None:
        """Set the anomaly threshold from training data reconstruction errors."""
        self.model.eval()
        tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            errors = self.model.reconstruction_error(tensor).cpu().numpy()

        self._training_errors = errors
        self.threshold = float(np.percentile(errors, self.threshold_percentile))

    def set_threshold(self, percentile: Optional[float] = None, value: Optional[float] = None) -> None:
        """Manually set or recalibrate the anomaly threshold.

        Args:
            percentile: Set threshold at this percentile of training errors.
            value: Set threshold to this exact value.
        """
        if value is not None:
            self.threshold = value
        elif percentile is not None and self._training_errors is not None:
            self.threshold = float(np.percentile(self._training_errors, percentile))
        else:
            raise ValueError("Provide either percentile or value")
        logger.info("Threshold set to %.6f", self.threshold)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores (reconstruction error) for input data.

        Args:
            X: Feature matrix.

        Returns:
            Array of anomaly scores (higher = more anomalous).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)

        self.model.eval()
        tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            errors = self.model.reconstruction_error(tensor).cpu().numpy()

        return errors

    def detect(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary anomaly labels based on the threshold.

        Args:
            X: Feature matrix.

        Returns:
            Array of binary labels (1 = anomaly, 0 = normal).
        """
        if self.threshold is None:
            raise RuntimeError("Threshold has not been set")

        scores = self.predict(X)
        return (scores > self.threshold).astype(int)

    def save(self, path: str) -> None:
        """Save model weights, scaler, and threshold to disk."""
        import joblib

        state = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "threshold": self.threshold,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dropout": self.dropout,
            "input_dim": self.model.input_dim,
            "training_errors": self._training_errors,
        }
        joblib.dump(state, path)
        logger.info("Autoencoder detector saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights, scaler, and threshold from disk."""
        import joblib

        state = joblib.load(path)

        self.scaler = state["scaler"]
        self.threshold = state["threshold"]
        self.encoder_layers = state["encoder_layers"]
        self.decoder_layers = state["decoder_layers"]
        self.dropout = state["dropout"]
        self._training_errors = state.get("training_errors")

        self.model = DeepAutoencoder(
            input_dim=state["input_dim"],
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self._is_fitted = True

        logger.info("Autoencoder detector loaded from %s", path)
