"""
Synthetic data generator for anomaly detection evaluation.

Creates realistic normal financial transaction data and injects
various types of anomalies for model training and testing.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic financial transaction data with configurable anomalies.

    Produces realistic normal transaction patterns including amounts,
    frequencies, merchant categories, and temporal features. Injects
    multiple types of anomalies: amount spikes, frequency bursts,
    unusual patterns, geographic outliers, and velocity changes.
    """

    MERCHANT_CATEGORIES = [
        "grocery", "restaurant", "gas_station", "retail", "online_shopping",
        "travel", "entertainment", "healthcare", "utilities", "insurance",
        "education", "professional_services", "automotive", "home_improvement",
    ]

    TRANSACTION_TYPES = ["purchase", "refund", "transfer", "payment", "withdrawal"]

    CHANNELS = ["in_store", "online", "mobile", "atm", "phone"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_normal_transactions(
        self,
        n_samples: int = 10000,
        n_accounts: int = 500,
    ) -> pd.DataFrame:
        """Generate realistic normal financial transaction data.

        Creates transactions with correlated features mimicking real-world
        spending patterns. Different merchant categories have distinct
        amount distributions and frequency profiles.

        Args:
            n_samples: Number of transactions to generate.
            n_accounts: Number of distinct accounts.

        Returns:
            DataFrame with transaction features.
        """
        accounts = [f"ACCT_{i:06d}" for i in range(n_accounts)]

        account_profiles = {}
        for acct in accounts:
            account_profiles[acct] = {
                "avg_amount": self.rng.lognormal(mean=4.0, sigma=0.8),
                "monthly_frequency": self.rng.poisson(lam=15) + 5,
                "preferred_categories": self.rng.choice(
                    self.MERCHANT_CATEGORIES,
                    size=self.rng.randint(3, 8),
                    replace=False,
                ).tolist(),
                "preferred_channel": self.rng.choice(self.CHANNELS),
                "balance": self.rng.lognormal(mean=8.0, sigma=1.0),
            }

        category_amounts = {
            "grocery": (40, 150),
            "restaurant": (15, 80),
            "gas_station": (25, 70),
            "retail": (20, 200),
            "online_shopping": (10, 300),
            "travel": (100, 2000),
            "entertainment": (10, 100),
            "healthcare": (20, 500),
            "utilities": (50, 300),
            "insurance": (100, 1000),
            "education": (50, 5000),
            "professional_services": (50, 500),
            "automotive": (30, 800),
            "home_improvement": (20, 500),
        }

        records = []
        start_date = pd.Timestamp("2023-01-01")

        for _ in range(n_samples):
            acct = self.rng.choice(accounts)
            profile = account_profiles[acct]

            category = self.rng.choice(profile["preferred_categories"])
            low, high = category_amounts[category]
            amount = self.rng.lognormal(
                mean=np.log((low + high) / 2), sigma=0.5
            )
            amount = np.clip(amount, low * 0.5, high * 2.0)

            day_offset = self.rng.randint(0, 365)
            hour = self._sample_transaction_hour()
            txn_date = start_date + pd.Timedelta(days=int(day_offset), hours=int(hour))

            channel = (
                profile["preferred_channel"]
                if self.rng.random() < 0.6
                else self.rng.choice(self.CHANNELS)
            )

            txn_type = "purchase" if self.rng.random() < 0.85 else self.rng.choice(self.TRANSACTION_TYPES)

            records.append({
                "account_id": acct,
                "transaction_date": txn_date,
                "amount": round(amount, 2),
                "merchant_category": category,
                "transaction_type": txn_type,
                "channel": channel,
                "hour_of_day": hour,
                "day_of_week": txn_date.dayofweek,
                "is_weekend": int(txn_date.dayofweek >= 5),
                "account_balance": round(profile["balance"] + self.rng.normal(0, profile["balance"] * 0.1), 2),
                "avg_monthly_amount": round(profile["avg_amount"] * profile["monthly_frequency"], 2),
                "transaction_count_7d": max(1, self.rng.poisson(profile["monthly_frequency"] / 4)),
                "transaction_count_30d": max(1, self.rng.poisson(profile["monthly_frequency"])),
                "avg_amount_7d": round(profile["avg_amount"] * (1 + self.rng.normal(0, 0.1)), 2),
                "avg_amount_30d": round(profile["avg_amount"] * (1 + self.rng.normal(0, 0.05)), 2),
                "days_since_last_txn": max(0, self.rng.exponential(scale=2.0)),
                "distance_from_home": abs(self.rng.normal(5, 10)),
                "is_foreign": 0,
                "risk_score": max(0, min(100, self.rng.normal(20, 10))),
            })

        df = pd.DataFrame(records)

        df = self._add_derived_features(df)

        logger.info("Generated %d normal transactions for %d accounts", n_samples, n_accounts)
        return df

    def _sample_transaction_hour(self) -> int:
        """Sample a transaction hour weighted toward business and evening hours."""
        hour_weights = [
            0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 0-5 AM
            0.02, 0.04, 0.06, 0.07, 0.08, 0.09,       # 6-11 AM
            0.10, 0.08, 0.07, 0.06, 0.06, 0.07,        # 12-5 PM
            0.08, 0.07, 0.05, 0.04, 0.03, 0.02,        # 6-11 PM
        ]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        return int(self.rng.choice(24, p=hour_weights))

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived numeric features for model consumption."""
        df["amount_to_balance_ratio"] = df["amount"] / df["account_balance"].replace(0, 1)
        df["amount_to_avg_ratio"] = df["amount"] / df["avg_amount_30d"].replace(0, 1)
        df["velocity_7d"] = df["transaction_count_7d"] / 7.0
        df["velocity_30d"] = df["transaction_count_30d"] / 30.0
        df["velocity_change"] = df["velocity_7d"] / df["velocity_30d"].replace(0, 0.01)
        df["amount_zscore"] = (
            (df["amount"] - df["avg_amount_30d"])
            / df.groupby("account_id")["amount"].transform("std").replace(0, 1)
        )
        df["log_amount"] = np.log1p(df["amount"])
        df["is_high_amount"] = (df["amount_to_avg_ratio"] > 3.0).astype(int)
        df["is_late_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 4)).astype(int)

        return df

    def inject_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_rate: float = 0.05,
        anomaly_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Inject various types of anomalies into the dataset.

        Args:
            df: Normal transaction DataFrame.
            anomaly_rate: Fraction of transactions to convert to anomalies.
            anomaly_types: Specific anomaly types to inject. If None,
                all types are used.

        Returns:
            DataFrame with injected anomalies and an 'is_anomaly' column.
        """
        df = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)

        if anomaly_types is None:
            anomaly_types = [
                "amount_spike",
                "frequency_burst",
                "unusual_pattern",
                "velocity_change",
                "foreign_outlier",
            ]

        anomaly_indices = self.rng.choice(len(df), size=n_anomalies, replace=False)
        df["is_anomaly"] = 0

        per_type = max(1, n_anomalies // len(anomaly_types))
        type_indices = {}
        start = 0

        for i, atype in enumerate(anomaly_types):
            if i == len(anomaly_types) - 1:
                type_indices[atype] = anomaly_indices[start:]
            else:
                type_indices[atype] = anomaly_indices[start:start + per_type]
                start += per_type

        for atype, indices in type_indices.items():
            if atype == "amount_spike":
                df = self._inject_amount_spike(df, indices)
            elif atype == "frequency_burst":
                df = self._inject_frequency_burst(df, indices)
            elif atype == "unusual_pattern":
                df = self._inject_unusual_pattern(df, indices)
            elif atype == "velocity_change":
                df = self._inject_velocity_change(df, indices)
            elif atype == "foreign_outlier":
                df = self._inject_foreign_outlier(df, indices)

        df.loc[anomaly_indices, "is_anomaly"] = 1

        n_injected = df["is_anomaly"].sum()
        logger.info(
            "Injected %d anomalies (%.2f%%) across %d types",
            n_injected, n_injected / len(df) * 100, len(anomaly_types),
        )
        return df

    def _inject_amount_spike(
        self, df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """Inject abnormally high transaction amounts."""
        multipliers = self.rng.uniform(5, 50, size=len(indices))
        df.loc[indices, "amount"] = df.loc[indices, "amount"] * multipliers
        df.loc[indices, "amount_to_balance_ratio"] = (
            df.loc[indices, "amount"] / df.loc[indices, "account_balance"].replace(0, 1)
        )
        df.loc[indices, "amount_to_avg_ratio"] = (
            df.loc[indices, "amount"] / df.loc[indices, "avg_amount_30d"].replace(0, 1)
        )
        df.loc[indices, "log_amount"] = np.log1p(df.loc[indices, "amount"])
        df.loc[indices, "is_high_amount"] = 1
        df.loc[indices, "amount_zscore"] = self.rng.uniform(3, 10, size=len(indices))
        return df

    def _inject_frequency_burst(
        self, df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """Inject abnormally high transaction frequencies."""
        df.loc[indices, "transaction_count_7d"] = self.rng.randint(20, 50, size=len(indices))
        df.loc[indices, "transaction_count_30d"] = self.rng.randint(60, 150, size=len(indices))
        df.loc[indices, "velocity_7d"] = df.loc[indices, "transaction_count_7d"] / 7.0
        df.loc[indices, "velocity_30d"] = df.loc[indices, "transaction_count_30d"] / 30.0
        df.loc[indices, "velocity_change"] = (
            df.loc[indices, "velocity_7d"] / df.loc[indices, "velocity_30d"].replace(0, 0.01)
        )
        df.loc[indices, "days_since_last_txn"] = self.rng.uniform(0, 0.1, size=len(indices))
        return df

    def _inject_unusual_pattern(
        self, df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """Inject transactions with unusual time and location patterns."""
        df.loc[indices, "hour_of_day"] = self.rng.choice([0, 1, 2, 3, 4], size=len(indices))
        df.loc[indices, "is_late_night"] = 1
        df.loc[indices, "distance_from_home"] = self.rng.uniform(100, 1000, size=len(indices))
        df.loc[indices, "risk_score"] = self.rng.uniform(60, 100, size=len(indices))
        return df

    def _inject_velocity_change(
        self, df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """Inject sudden velocity changes in transaction patterns."""
        df.loc[indices, "velocity_7d"] = self.rng.uniform(3, 10, size=len(indices))
        df.loc[indices, "velocity_change"] = self.rng.uniform(3, 15, size=len(indices))
        df.loc[indices, "days_since_last_txn"] = self.rng.uniform(0, 0.05, size=len(indices))
        df.loc[indices, "transaction_count_7d"] = (df.loc[indices, "velocity_7d"] * 7).astype(int)
        return df

    def _inject_foreign_outlier(
        self, df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """Inject foreign and geographically anomalous transactions."""
        df.loc[indices, "is_foreign"] = 1
        df.loc[indices, "distance_from_home"] = self.rng.uniform(500, 10000, size=len(indices))
        df.loc[indices, "risk_score"] = self.rng.uniform(50, 95, size=len(indices))
        amounts = df.loc[indices, "avg_amount_30d"] * self.rng.uniform(2, 8, size=len(indices))
        df.loc[indices, "amount"] = amounts
        df.loc[indices, "log_amount"] = np.log1p(amounts)
        df.loc[indices, "amount_to_avg_ratio"] = (
            amounts / df.loc[indices, "avg_amount_30d"].replace(0, 1)
        )
        return df

    def get_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only numeric feature columns suitable for model input.

        Drops identifiers, dates, and the anomaly label column.
        """
        exclude_cols = [
            "account_id", "transaction_date", "merchant_category",
            "transaction_type", "channel", "is_anomaly",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return df[feature_cols].select_dtypes(include=[np.number])

    def generate_labeled_dataset(
        self,
        n_samples: int = 10000,
        anomaly_rate: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Generate a complete labeled dataset ready for model training.

        Returns:
            Tuple of (full_df, X_numeric, y_labels).
        """
        df = self.generate_normal_transactions(n_samples=n_samples)
        df = self.inject_anomalies(df, anomaly_rate=anomaly_rate)

        X = self.get_numeric_features(df)
        y = df["is_anomaly"]

        return df, X, y
