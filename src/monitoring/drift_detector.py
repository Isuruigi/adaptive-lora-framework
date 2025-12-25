"""
Model drift detection for production monitoring.

Features:
- Input distribution monitoring
- Output quality tracking
- Statistical drift detection
- Adaptive routing adjustment
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift.

    Attributes:
        timestamp: When drift was detected.
        drift_type: Type of drift (input, output, performance).
        severity: Severity level (low, medium, high).
        metric_name: Metric that drifted.
        baseline_value: Expected value.
        current_value: Observed value.
        details: Additional details.
    """

    timestamp: datetime
    drift_type: str
    severity: str
    metric_name: str
    baseline_value: float
    current_value: float
    details: Dict[str, Any]


class DriftDetector:
    """Detect distribution drift in model inputs and outputs.

    Example:
        >>> detector = DriftDetector()
        >>> detector.update_baseline(reference_embeddings)
        >>> drift = detector.check_drift(current_embeddings)
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold: float = 0.1,
        min_samples: int = 100
    ):
        """Initialize drift detector.

        Args:
            window_size: Size of sliding window.
            alert_threshold: Threshold for triggering alerts.
            min_samples: Minimum samples before detection.
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.min_samples = min_samples

        # Baseline statistics
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.baseline_samples: int = 0

        # Current window
        self.current_window: Deque[np.ndarray] = deque(maxlen=window_size)

        # History
        self.drift_history: List[DriftAlert] = []

    def update_baseline(
        self,
        embeddings: np.ndarray
    ) -> None:
        """Update baseline distribution.

        Args:
            embeddings: Reference embeddings (n_samples, dim).
        """
        self.baseline_mean = np.mean(embeddings, axis=0)
        self.baseline_std = np.std(embeddings, axis=0) + 1e-10
        self.baseline_samples = len(embeddings)

        logger.info(
            f"Updated baseline with {self.baseline_samples} samples "
            f"(dim={len(self.baseline_mean)})"
        )

    def add_sample(self, embedding: np.ndarray) -> Optional[DriftAlert]:
        """Add a sample and check for drift.

        Args:
            embedding: New embedding vector.

        Returns:
            DriftAlert if drift detected, None otherwise.
        """
        self.current_window.append(embedding)

        # Check if enough samples
        if len(self.current_window) < self.min_samples:
            return None

        # Check drift
        return self.check_drift()

    def check_drift(
        self,
        embeddings: Optional[np.ndarray] = None
    ) -> Optional[DriftAlert]:
        """Check for distribution drift.

        Args:
            embeddings: Optional specific embeddings to check.

        Returns:
            DriftAlert if drift detected.
        """
        if self.baseline_mean is None:
            return None

        if embeddings is None:
            if len(self.current_window) < self.min_samples:
                return None
            embeddings = np.array(list(self.current_window))

        # Calculate current statistics
        current_mean = np.mean(embeddings, axis=0)
        current_std = np.std(embeddings, axis=0) + 1e-10

        # Calculate drift metrics
        mean_drift = self._calculate_mean_drift(current_mean)
        variance_drift = self._calculate_variance_drift(current_std)

        # Combined drift score
        drift_score = 0.7 * mean_drift + 0.3 * variance_drift

        if drift_score > self.alert_threshold:
            severity = self._classify_severity(drift_score)

            alert = DriftAlert(
                timestamp=datetime.utcnow(),
                drift_type="input_distribution",
                severity=severity,
                metric_name="embedding_drift",
                baseline_value=0.0,
                current_value=drift_score,
                details={
                    "mean_drift": float(mean_drift),
                    "variance_drift": float(variance_drift),
                    "sample_count": len(embeddings)
                }
            )

            self.drift_history.append(alert)
            logger.warning(f"Drift detected: {severity} (score={drift_score:.4f})")

            return alert

        return None

    def _calculate_mean_drift(self, current_mean: np.ndarray) -> float:
        """Calculate mean drift using normalized distance."""
        diff = current_mean - self.baseline_mean
        normalized_diff = diff / self.baseline_std
        return float(np.mean(np.abs(normalized_diff)))

    def _calculate_variance_drift(self, current_std: np.ndarray) -> float:
        """Calculate variance drift."""
        ratio = current_std / self.baseline_std
        return float(np.mean(np.abs(np.log(ratio))))

    def _classify_severity(self, drift_score: float) -> str:
        """Classify drift severity."""
        if drift_score > 0.5:
            return "high"
        elif drift_score > 0.3:
            return "medium"
        return "low"

    def get_drift_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get summary of drift events.

        Args:
            time_window: Optional time window to analyze.

        Returns:
            Drift summary.
        """
        alerts = self.drift_history

        if time_window:
            cutoff = datetime.utcnow() - time_window
            alerts = [a for a in alerts if a.timestamp >= cutoff]

        return {
            "total_alerts": len(alerts),
            "by_severity": {
                "high": sum(1 for a in alerts if a.severity == "high"),
                "medium": sum(1 for a in alerts if a.severity == "medium"),
                "low": sum(1 for a in alerts if a.severity == "low")
            },
            "by_type": {
                "input_distribution": sum(1 for a in alerts if a.drift_type == "input_distribution"),
                "output_quality": sum(1 for a in alerts if a.drift_type == "output_quality"),
                "performance": sum(1 for a in alerts if a.drift_type == "performance")
            },
            "recent_alerts": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.drift_type,
                    "severity": a.severity,
                    "metric": a.metric_name,
                    "value": a.current_value
                }
                for a in alerts[-10:]
            ]
        }

    def reset_baseline(self) -> None:
        """Reset baseline statistics."""
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_samples = 0
        self.current_window.clear()


class OutputQualityMonitor:
    """Monitor output quality over time."""

    def __init__(
        self,
        window_size: int = 500,
        degradation_threshold: float = 0.1
    ):
        """Initialize quality monitor.

        Args:
            window_size: Size of sliding window.
            degradation_threshold: Threshold for quality degradation.
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold

        # Quality scores
        self.baseline_scores: Optional[np.ndarray] = None
        self.current_scores: Deque[float] = deque(maxlen=window_size)

        # Per-adapter tracking
        self.adapter_scores: Dict[str, Deque[float]] = {}

    def update_baseline(self, scores: List[float]) -> None:
        """Set baseline quality scores.

        Args:
            scores: Reference quality scores.
        """
        self.baseline_scores = np.array(scores)
        logger.info(
            f"Baseline quality: mean={np.mean(scores):.4f}, "
            f"std={np.std(scores):.4f}"
        )

    def add_score(
        self,
        score: float,
        adapter: Optional[str] = None
    ) -> Optional[DriftAlert]:
        """Add a quality score.

        Args:
            score: Quality score (0-1).
            adapter: Optional adapter name.

        Returns:
            DriftAlert if degradation detected.
        """
        self.current_scores.append(score)

        if adapter:
            if adapter not in self.adapter_scores:
                self.adapter_scores[adapter] = deque(maxlen=self.window_size)
            self.adapter_scores[adapter].append(score)

        # Check for degradation
        return self._check_degradation()

    def _check_degradation(self) -> Optional[DriftAlert]:
        """Check for quality degradation."""
        if self.baseline_scores is None:
            return None

        if len(self.current_scores) < 50:
            return None

        baseline_mean = np.mean(self.baseline_scores)
        current_mean = np.mean(list(self.current_scores))

        degradation = baseline_mean - current_mean

        if degradation > self.degradation_threshold:
            severity = "high" if degradation > 0.3 else "medium" if degradation > 0.2 else "low"

            alert = DriftAlert(
                timestamp=datetime.utcnow(),
                drift_type="output_quality",
                severity=severity,
                metric_name="quality_score",
                baseline_value=float(baseline_mean),
                current_value=float(current_mean),
                details={
                    "degradation": float(degradation),
                    "sample_count": len(self.current_scores)
                }
            )

            logger.warning(
                f"Quality degradation detected: {baseline_mean:.4f} -> {current_mean:.4f}"
            )

            return alert

        return None

    def get_adapter_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare quality across adapters.

        Returns:
            Per-adapter quality statistics.
        """
        result = {}

        for adapter, scores in self.adapter_scores.items():
            scores_list = list(scores)
            if scores_list:
                result[adapter] = {
                    "mean": np.mean(scores_list),
                    "std": np.std(scores_list),
                    "min": min(scores_list),
                    "max": max(scores_list),
                    "count": len(scores_list)
                }

        return result


class AdaptiveRouter:
    """Adjust routing based on drift detection.

    Dynamically adjusts adapter weights based on
    performance monitoring.
    """

    def __init__(
        self,
        adapters: List[str],
        update_interval: int = 100
    ):
        """Initialize adaptive router.

        Args:
            adapters: List of adapter names.
            update_interval: Requests between weight updates.
        """
        self.adapters = adapters
        self.update_interval = update_interval

        # Initial equal weights
        self.weights = {a: 1.0 / len(adapters) for a in adapters}

        # Performance tracking
        self.adapter_performance: Dict[str, Deque[float]] = {
            a: deque(maxlen=100) for a in adapters
        }

        self.request_count = 0

    def record_performance(
        self,
        adapter: str,
        score: float
    ) -> None:
        """Record adapter performance.

        Args:
            adapter: Adapter name.
            score: Performance score (0-1).
        """
        if adapter in self.adapter_performance:
            self.adapter_performance[adapter].append(score)
            self.request_count += 1

            if self.request_count % self.update_interval == 0:
                self._update_weights()

    def _update_weights(self) -> None:
        """Update routing weights based on performance."""
        scores = {}

        for adapter, perf in self.adapter_performance.items():
            if perf:
                scores[adapter] = np.mean(list(perf))
            else:
                scores[adapter] = 0.5

        # Softmax over scores
        total = sum(np.exp(s) for s in scores.values())
        self.weights = {
            a: np.exp(scores[a]) / total
            for a in self.adapters
        }

        logger.info(f"Updated routing weights: {self.weights}")

    def get_routing_weights(self) -> Dict[str, float]:
        """Get current routing weights.

        Returns:
            Adapter weights dictionary.
        """
        return self.weights.copy()

    def select_adapter(self) -> str:
        """Select adapter based on current weights.

        Returns:
            Selected adapter name.
        """
        adapters = list(self.weights.keys())
        weights = list(self.weights.values())

        return np.random.choice(adapters, p=weights)
