"""
FastAPI application for the anomaly detection system.

Provides REST endpoints for scoring transactions, batch detection,
model training, and performance monitoring.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    """Single transaction to score for anomalies."""

    amount: float = Field(..., description="Transaction amount")
    hour_of_day: int = Field(12, ge=0, le=23)
    day_of_week: int = Field(1, ge=0, le=6)
    is_weekend: int = Field(0, ge=0, le=1)
    account_balance: float = Field(5000.0, ge=0)
    avg_monthly_amount: float = Field(2000.0)
    transaction_count_7d: int = Field(5, ge=0)
    transaction_count_30d: int = Field(20, ge=0)
    avg_amount_7d: float = Field(100.0)
    avg_amount_30d: float = Field(100.0)
    days_since_last_txn: float = Field(1.0, ge=0)
    distance_from_home: float = Field(5.0, ge=0)
    is_foreign: int = Field(0, ge=0, le=1)
    risk_score: float = Field(20.0, ge=0, le=100)
    amount_to_balance_ratio: Optional[float] = None
    amount_to_avg_ratio: Optional[float] = None
    velocity_7d: Optional[float] = None
    velocity_30d: Optional[float] = None
    velocity_change: Optional[float] = None
    amount_zscore: Optional[float] = None
    log_amount: Optional[float] = None
    is_high_amount: Optional[int] = None
    is_late_night: Optional[int] = None


class DetectionResponse(BaseModel):
    """Response for a single transaction anomaly detection."""

    request_id: str
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: bool
    ae_score: float
    if_score: float
    processing_time_ms: float


class BatchTransactionInput(BaseModel):
    """Batch of transactions to score."""

    transactions: list[TransactionInput]


class BatchDetectionResponse(BaseModel):
    """Response for batch anomaly detection."""

    request_id: str
    results: list[DetectionResponse]
    total_transactions: int
    total_anomalies: int
    processing_time_ms: float


class TrainingRequest(BaseModel):
    """Request to trigger model training."""

    n_samples: int = Field(10000, ge=1000, le=500000)
    anomaly_rate: float = Field(0.05, ge=0.01, le=0.3)
    epochs: int = Field(50, ge=10, le=500)


class TrainingResponse(BaseModel):
    """Response after model training."""

    request_id: str
    status: str
    training_samples: int
    ae_threshold: Optional[float]
    if_threshold: Optional[float]
    ensemble_threshold: Optional[float]
    training_time_ms: float


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    model_fitted: bool
    ae_threshold: Optional[float]
    if_threshold: Optional[float]
    ensemble_threshold: Optional[float]
    voting_strategy: str
    ae_weight: float
    if_weight: float
    total_scored: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = False


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Anomaly Detection API",
    description=(
        "Anomaly detection system using Isolation Forest and Deep Autoencoders "
        "for scoring financial transactions and insurance claims."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = None
_total_scored = 0


def get_detector():
    """Lazily initialize the ensemble detector singleton."""
    global _detector
    if _detector is None:
        from src.ensemble_detector import EnsembleAnomalyDetector

        config_path = "configs/config.yaml"
        if not Path(config_path).exists():
            config_path = None
        _detector = EnsembleAnomalyDetector(config_path=config_path)
    return _detector


def transaction_to_dataframe(txn: TransactionInput) -> pd.DataFrame:
    """Convert a TransactionInput to a single-row DataFrame."""
    data = txn.model_dump()

    if data.get("amount_to_balance_ratio") is None:
        data["amount_to_balance_ratio"] = data["amount"] / max(data["account_balance"], 1)
    if data.get("amount_to_avg_ratio") is None:
        data["amount_to_avg_ratio"] = data["amount"] / max(data["avg_amount_30d"], 1)
    if data.get("velocity_7d") is None:
        data["velocity_7d"] = data["transaction_count_7d"] / 7.0
    if data.get("velocity_30d") is None:
        data["velocity_30d"] = data["transaction_count_30d"] / 30.0
    if data.get("velocity_change") is None:
        data["velocity_change"] = data.get("velocity_7d", 0) / max(data.get("velocity_30d", 0.01), 0.01)
    if data.get("amount_zscore") is None:
        data["amount_zscore"] = (data["amount"] - data["avg_amount_30d"]) / max(data["avg_amount_30d"] * 0.5, 1)
    if data.get("log_amount") is None:
        data["log_amount"] = np.log1p(data["amount"])
    if data.get("is_high_amount") is None:
        data["is_high_amount"] = int(data["amount_to_avg_ratio"] > 3.0)
    if data.get("is_late_night") is None:
        data["is_late_night"] = int(data["hour_of_day"] >= 23 or data["hour_of_day"] <= 4)

    return pd.DataFrame([data])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(transaction: TransactionInput):
    """Score a single transaction for anomaly detection.

    Returns an anomaly score between 0 and 1, along with individual
    model scores and a binary anomaly decision.
    """
    global _total_scored

    detector = get_detector()
    if not detector._is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained. POST to /train first.",
        )

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    df = transaction_to_dataframe(transaction)
    detailed = detector.predict_detailed(df)

    ensemble_score = float(detailed["ensemble_score"].iloc[0])
    is_anomaly = bool(ensemble_score > detector.threshold)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _total_scored += 1

    return DetectionResponse(
        request_id=request_id,
        anomaly_score=round(ensemble_score, 6),
        is_anomaly=is_anomaly,
        ae_score=round(float(detailed["ae_normalized"].iloc[0]), 6),
        if_score=round(float(detailed["if_normalized"].iloc[0]), 6),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(batch: BatchTransactionInput):
    """Score a batch of transactions for anomaly detection."""
    global _total_scored

    detector = get_detector()
    if not detector._is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained. POST to /train first.",
        )

    if len(batch.transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 transactions per batch.",
        )

    batch_request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    frames = [transaction_to_dataframe(txn) for txn in batch.transactions]
    df = pd.concat(frames, ignore_index=True)

    detailed = detector.predict_detailed(df)

    results = []
    for idx in range(len(df)):
        score = float(detailed["ensemble_score"].iloc[idx])
        results.append(DetectionResponse(
            request_id=f"{batch_request_id}-{idx}",
            anomaly_score=round(score, 6),
            is_anomaly=bool(score > detector.threshold),
            ae_score=round(float(detailed["ae_normalized"].iloc[idx]), 6),
            if_score=round(float(detailed["if_normalized"].iloc[idx]), 6),
            processing_time_ms=0,
        ))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    total_anomalies = sum(1 for r in results if r.is_anomaly)
    _total_scored += len(batch.transactions)

    return BatchDetectionResponse(
        request_id=batch_request_id,
        results=results,
        total_transactions=len(batch.transactions),
        total_anomalies=total_anomalies,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Trigger model training with synthetic data.

    Generates synthetic transaction data, trains both the autoencoder
    and isolation forest, and calibrates the ensemble threshold.
    """
    from src.data_generator import SyntheticDataGenerator

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "Training requested: %d samples, %.2f anomaly rate, %d epochs",
        request.n_samples, request.anomaly_rate, request.epochs,
    )

    generator = SyntheticDataGenerator()
    full_df, X_numeric, y_labels = generator.generate_labeled_dataset(
        n_samples=request.n_samples,
        anomaly_rate=request.anomaly_rate,
    )

    detector = get_detector()
    detector.autoencoder.epochs = request.epochs

    training_info = detector.fit(X_numeric, verbose=True)

    if y_labels.sum() > 0:
        detector.calibrate_threshold(X_numeric, y_labels, target_metric="f1")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return TrainingResponse(
        request_id=request_id,
        status="completed",
        training_samples=request.n_samples,
        ae_threshold=training_info.get("ae_threshold"),
        if_threshold=training_info.get("if_threshold"),
        ensemble_threshold=detector.threshold,
        training_time_ms=round(elapsed_ms, 2),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return current model performance metrics and configuration."""
    detector = get_detector()

    return MetricsResponse(
        model_fitted=detector._is_fitted,
        ae_threshold=detector.autoencoder.threshold if detector._is_fitted else None,
        if_threshold=detector.isolation_forest.threshold if detector._is_fitted else None,
        ensemble_threshold=detector.threshold,
        voting_strategy=detector.voting_strategy,
        ae_weight=detector.ae_weight,
        if_weight=detector.if_weight,
        total_scored=_total_scored,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    detector = get_detector()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=detector._is_fitted,
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
