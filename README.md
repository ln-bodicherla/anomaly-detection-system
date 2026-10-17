# Anomaly Detection System

Unsupervised anomaly detection using Isolation Forest and Deep Autoencoders for identifying novel patterns in financial transaction data and insurance claims. Combines statistical and deep learning approaches for robust detection.

## Overview

Detecting anomalies in financial data requires methods that can identify previously unseen patterns without relying on labeled examples. This system implements two complementary approaches:

- **Deep Autoencoder** -- A PyTorch neural network trained to reconstruct normal transaction patterns. Anomalies produce high reconstruction error because the model has never learned to represent them.
- **Isolation Forest** -- An ensemble of random trees that isolates anomalies by exploiting the fact that outliers require fewer random splits to separate from the rest of the data.

An ensemble detector combines scores from both methods using configurable weighted fusion and voting strategies.

## Architecture

```
Transaction Data
       |
       v
  Feature Scaling
  |              |
  v              v
Autoencoder   Isolation Forest
  |              |
  v              v
Reconstruction  Anomaly
Error Score     Path Score
  |              |
  v              v
  Ensemble Score Fusion
       |
       v
  Anomaly Decision
  (weighted average / voting)
```

## Installation

```bash
git clone https://github.com/ln-bodicherla/anomaly-detection-system.git
cd anomaly-detection-system

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### Train and detect anomalies

```python
from src.data_generator import SyntheticDataGenerator
from src.ensemble_detector import EnsembleAnomalyDetector

# Generate synthetic data with injected anomalies
generator = SyntheticDataGenerator()
data = generator.generate_normal_transactions(n_samples=10000)
data_with_anomalies = generator.inject_anomalies(data, anomaly_rate=0.05)

# Train ensemble detector
detector = EnsembleAnomalyDetector(config_path="configs/config.yaml")
detector.fit(data_with_anomalies.drop(columns=["is_anomaly"]))

# Score and detect
scores = detector.predict(data_with_anomalies.drop(columns=["is_anomaly"]))
labels = detector.detect(data_with_anomalies.drop(columns=["is_anomaly"]))
```

### REST API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Score a single transaction
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"amount": 15000.0, "frequency": 12, "avg_amount": 500.0, ...}'
```

## Detection Methods

### Deep Autoencoder

- Configurable encoder/decoder architecture with symmetric layer sizes
- Batch normalization and dropout for regularization
- Trained on normal data only (semi-supervised)
- Anomaly score = reconstruction error (MSE between input and output)
- Threshold set at the 95th percentile of training reconstruction errors

### Isolation Forest

- Ensemble of 200 randomized isolation trees
- Anomaly score based on average path length to isolation
- Permutation-based feature importance for explaining anomalies
- Configurable contamination parameter for expected anomaly rate

### Ensemble

- Weighted score fusion (default: 60% autoencoder, 40% isolation forest)
- Min-max normalization of individual model scores before fusion
- Voting strategies: weighted average, unanimous, or any-agree
- Threshold calibration on a validation set

## Configuration

All parameters are defined in `configs/config.yaml`:

```yaml
autoencoder:
  encoder_layers: [128, 64, 32, 16]
  dropout: 0.2
  learning_rate: 0.001
  epochs: 100
  threshold_percentile: 95

isolation_forest:
  n_estimators: 200
  contamination: 0.05

ensemble:
  autoencoder_weight: 0.6
  isolation_forest_weight: 0.4
  voting_strategy: "weighted_average"
```

## Project Structure

```
anomaly-detection-system/
├── configs/
│   └── config.yaml
├── src/
│   ├── api.py
│   ├── autoencoder.py
│   ├── data_generator.py
│   ├── ensemble_detector.py
│   └── isolation_forest_model.py
├── requirements.txt
└── README.md
```

## Data and Evaluation

Uses a generated synthetic transaction dataset (`src/data_generator.py`) with injected anomalies for demonstration. No evaluation results are published in this repository; because the anomalies are injected by the generator, scores on this data would not reflect performance on real transactions.

## License

MIT License
