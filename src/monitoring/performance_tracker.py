"""
Performance tracking for production monitoring.

Features:
- Request latency tracking
- Throughput monitoring
- Cost estimation
- Resource utilization
- Alerting thresholds
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class RequestMetric:
    """Metrics for a single request.

    Attributes:
        request_id: Unique request identifier.
        adapter: Adapter used.
        latency_ms: Request latency.
        tokens_in: Input tokens.
        tokens_out: Output tokens.
        success: Whether request succeeded.
        timestamp: Request timestamp.
    """

    request_id: str
    adapter: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.latency_ms <= 0:
            return 0.0
        return (self.tokens_out * 1000) / self.latency_ms


class PerformanceTracker:
    """Track and analyze system performance.

    Collects metrics and provides analytics for:
    - Latency distribution
    - Throughput over time
    - Adapter performance comparison
    - Cost estimation

    Example:
        >>> tracker = PerformanceTracker()
        >>> tracker.record_request(request_id, "reasoning", 150.5, 50, 200, True)
        >>> stats = tracker.get_statistics()
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_latency_threshold: float = 5000.0,
        alert_error_rate_threshold: float = 0.1
    ):
        """Initialize performance tracker.

        Args:
            window_size: Number of recent requests to keep.
            alert_latency_threshold: Latency threshold for alerts (ms).
            alert_error_rate_threshold: Error rate threshold for alerts.
        """
        self.window_size = window_size
        self.alert_latency_threshold = alert_latency_threshold
        self.alert_error_rate_threshold = alert_error_rate_threshold

        # Metrics storage
        self.requests: Deque[RequestMetric] = deque(maxlen=window_size)
        self.adapter_metrics: Dict[str, Deque[RequestMetric]] = {}

        # Aggregated stats
        self.total_requests = 0
        self.total_errors = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Setup Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self.prom_requests = Counter(
            "lora_requests_total",
            "Total number of requests",
            ["adapter", "status"]
        )
        self.prom_latency = Histogram(
            "lora_request_latency_seconds",
            "Request latency in seconds",
            ["adapter"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        self.prom_tokens = Counter(
            "lora_tokens_total",
            "Total tokens processed",
            ["direction"]
        )
        self.prom_active_adapters = Gauge(
            "lora_active_adapters",
            "Number of active adapters"
        )

    def record_request(
        self,
        request_id: str,
        adapter: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        success: bool
    ) -> None:
        """Record a request's metrics.

        Args:
            request_id: Unique request identifier.
            adapter: Adapter used.
            latency_ms: Request latency in milliseconds.
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.
            success: Whether request succeeded.
        """
        metric = RequestMetric(
            request_id=request_id,
            adapter=adapter,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            success=success
        )

        # Store metric
        self.requests.append(metric)

        # Per-adapter metrics
        if adapter not in self.adapter_metrics:
            self.adapter_metrics[adapter] = deque(maxlen=self.window_size)
        self.adapter_metrics[adapter].append(metric)

        # Update aggregates
        self.total_requests += 1
        if not success:
            self.total_errors += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            status = "success" if success else "error"
            self.prom_requests.labels(adapter=adapter, status=status).inc()
            self.prom_latency.labels(adapter=adapter).observe(latency_ms / 1000)
            self.prom_tokens.labels(direction="in").inc(tokens_in)
            self.prom_tokens.labels(direction="out").inc(tokens_out)

        # Check alerts
        self._check_alerts(metric)

    def _check_alerts(self, metric: RequestMetric) -> None:
        """Check if metric triggers any alerts."""
        alerts = []

        # Latency alert
        if metric.latency_ms > self.alert_latency_threshold:
            alerts.append({
                "type": "high_latency",
                "value": metric.latency_ms,
                "threshold": self.alert_latency_threshold,
                "request_id": metric.request_id
            })

        # Error rate alert
        recent_requests = list(self.requests)[-100:]
        if recent_requests:
            error_rate = sum(1 for r in recent_requests if not r.success) / len(recent_requests)
            if error_rate > self.alert_error_rate_threshold:
                alerts.append({
                    "type": "high_error_rate",
                    "value": error_rate,
                    "threshold": self.alert_error_rate_threshold
                })

        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback.

        Args:
            callback: Function to call on alert.
        """
        self.alert_callbacks.append(callback)

    def get_statistics(
        self,
        adapter: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance statistics.

        Args:
            adapter: Optional specific adapter to analyze.
            time_window: Optional time window to analyze.

        Returns:
            Dictionary of statistics.
        """
        # Filter metrics
        if adapter:
            metrics = list(self.adapter_metrics.get(adapter, []))
        else:
            metrics = list(self.requests)

        if time_window:
            cutoff = datetime.utcnow() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff]

        if not metrics:
            return {"error": "No metrics available"}

        # Calculate statistics
        latencies = [m.latency_ms for m in metrics]
        successes = [m for m in metrics if m.success]
        tokens_per_sec = [m.tokens_per_second for m in metrics if m.tokens_per_second > 0]

        import numpy as np

        return {
            "total_requests": len(metrics),
            "success_rate": len(successes) / len(metrics),
            "latency": {
                "mean_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": min(latencies),
                "max_ms": max(latencies)
            },
            "throughput": {
                "mean_tokens_per_sec": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "total_tokens_in": sum(m.tokens_in for m in metrics),
                "total_tokens_out": sum(m.tokens_out for m in metrics)
            },
            "adapters": self._get_adapter_breakdown(metrics)
        }

    def _get_adapter_breakdown(
        self,
        metrics: List[RequestMetric]
    ) -> Dict[str, Dict[str, Any]]:
        """Get per-adapter statistics."""
        import numpy as np

        adapter_stats = {}

        adapters = set(m.adapter for m in metrics)
        for adapter in adapters:
            adapter_metrics = [m for m in metrics if m.adapter == adapter]
            latencies = [m.latency_ms for m in adapter_metrics]

            adapter_stats[adapter] = {
                "requests": len(adapter_metrics),
                "success_rate": sum(1 for m in adapter_metrics if m.success) / len(adapter_metrics),
                "mean_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95)
            }

        return adapter_stats

    def get_cost_estimate(
        self,
        cost_per_1k_tokens_in: float = 0.001,
        cost_per_1k_tokens_out: float = 0.002
    ) -> Dict[str, float]:
        """Estimate costs based on token usage.

        Args:
            cost_per_1k_tokens_in: Cost per 1K input tokens.
            cost_per_1k_tokens_out: Cost per 1K output tokens.

        Returns:
            Cost breakdown.
        """
        input_cost = (self.total_tokens_in / 1000) * cost_per_1k_tokens_in
        output_cost = (self.total_tokens_out / 1000) * cost_per_1k_tokens_out

        return {
            "total_cost": input_cost + output_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out
        }

    def export_metrics(self) -> str:
        """Export Prometheus metrics.

        Returns:
            Prometheus metrics text.
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest().decode("utf-8")
        return ""

    def reset(self) -> None:
        """Reset all metrics."""
        self.requests.clear()
        self.adapter_metrics.clear()
        self.total_requests = 0
        self.total_errors = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0


class ResourceMonitor:
    """Monitor system resource utilization."""

    def __init__(self, sample_interval: float = 1.0):
        """Initialize resource monitor.

        Args:
            sample_interval: Sampling interval in seconds.
        """
        self.sample_interval = sample_interval
        self.samples: Deque[Dict[str, Any]] = deque(maxlen=3600)  # 1 hour at 1s

    def sample(self) -> Dict[str, Any]:
        """Take a resource sample.

        Returns:
            Resource utilization metrics.
        """
        import psutil

        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3)
        }

        # GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                sample["gpu_count"] = torch.cuda.device_count()
                sample["gpu_memory"] = []

                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

                    sample["gpu_memory"].append({
                        "device": i,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "total_gb": total,
                        "utilization_percent": (allocated / total) * 100
                    })
        except Exception:
            pass

        self.samples.append(sample)
        return sample

    def get_recent_samples(self, count: int = 60) -> List[Dict[str, Any]]:
        """Get recent samples.

        Args:
            count: Number of samples to return.

        Returns:
            List of samples.
        """
        return list(self.samples)[-count:]


class DriftDetector:
    """Detect data drift and concept drift in requests.

    Monitors:
    - Input distribution changes
    - Output quality changes
    - Performance degradation

    Example:
        >>> detector = DriftDetector()
        >>> detector.add_query("reasoning", "Explain ML")
        >>> drift = detector.detect_drift("reasoning")
    """

    def __init__(
        self,
        reference_window_size: int = 1000,
        detection_window_size: int = 100,
        significance_level: float = 0.05
    ):
        """Initialize drift detector.

        Args:
            reference_window_size: Size of reference window.
            detection_window_size: Size of detection window.
            significance_level: Statistical significance level.
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.significance_level = significance_level

        # Store embeddings for distribution comparison
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embedding_available = True
        except ImportError:
            self.embedding_model = None
            self._embedding_available = False
            logger.warning("sentence-transformers not available for drift detection")

        self.reference_embeddings: Dict[str, Any] = {}
        self.recent_embeddings: Dict[str, deque] = {}
        self.quality_history: Dict[str, deque] = {}

    def add_query(
        self,
        adapter_name: str,
        query: str,
        quality_score: Optional[float] = None
    ) -> None:
        """Add query for drift detection.

        Args:
            adapter_name: Adapter name.
            query: Query text.
            quality_score: Optional quality score.
        """
        if not self._embedding_available:
            return

        # Initialize adapter storage
        if adapter_name not in self.recent_embeddings:
            self.recent_embeddings[adapter_name] = deque(maxlen=self.detection_window_size)
            self.quality_history[adapter_name] = deque(maxlen=self.detection_window_size)

        # Embed query
        import numpy as np
        embedding = self.embedding_model.encode(query)
        self.recent_embeddings[adapter_name].append(embedding)

        if quality_score is not None:
            self.quality_history[adapter_name].append(quality_score)

        # Initialize reference if needed
        if adapter_name not in self.reference_embeddings:
            if len(self.recent_embeddings[adapter_name]) >= self.detection_window_size:
                self.set_reference_distribution(adapter_name)

    def set_reference_distribution(self, adapter_name: str) -> None:
        """Set current distribution as reference.

        Args:
            adapter_name: Adapter name.
        """
        if adapter_name not in self.recent_embeddings:
            return

        import numpy as np
        embeddings = np.array(list(self.recent_embeddings[adapter_name]))
        self.reference_embeddings[adapter_name] = {
            "embeddings": embeddings,
            "quality_mean": np.mean(list(self.quality_history.get(adapter_name, []))),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Set reference distribution for {adapter_name}")

    def detect_drift(self, adapter_name: str) -> Dict[str, Any]:
        """Detect distribution drift.

        Uses Maximum Mean Discrepancy (MMD) for distribution comparison.

        Args:
            adapter_name: Adapter name.

        Returns:
            Drift detection results.
        """
        if adapter_name not in self.reference_embeddings:
            return {"drift_detected": False, "reason": "no_reference"}

        if len(self.recent_embeddings.get(adapter_name, [])) < self.detection_window_size:
            return {"drift_detected": False, "reason": "insufficient_data"}

        import numpy as np
        reference = self.reference_embeddings[adapter_name]["embeddings"]
        recent = np.array(list(self.recent_embeddings[adapter_name]))

        # Compute Maximum Mean Discrepancy (MMD)
        mmd = self._compute_mmd(reference, recent)

        # Threshold (tune based on your data)
        drift_threshold = 0.1
        drift_detected = mmd > drift_threshold

        # Quality drift check
        quality_drift = False
        quality_change = 0.0
        if adapter_name in self.quality_history and len(self.quality_history[adapter_name]) > 10:
            current_quality = np.mean(list(self.quality_history[adapter_name]))
            ref_quality = self.reference_embeddings[adapter_name].get("quality_mean", current_quality)
            quality_change = current_quality - ref_quality
            quality_drift = quality_change < -0.1  # 10% quality drop

        result = {
            "drift_detected": drift_detected or quality_drift,
            "distribution_drift": drift_detected,
            "quality_drift": quality_drift,
            "mmd_score": float(mmd),
            "quality_change": float(quality_change),
            "threshold": drift_threshold
        }

        if drift_detected:
            logger.warning(f"Distribution drift detected for {adapter_name}: MMD={mmd:.4f}")
        if quality_drift:
            logger.warning(f"Quality drift detected for {adapter_name}: change={quality_change:.4f}")

        return result

    def _compute_mmd(self, X: Any, Y: Any) -> float:
        """Compute Maximum Mean Discrepancy.

        Args:
            X: Reference distribution samples.
            Y: Test distribution samples.

        Returns:
            MMD value.
        """
        import numpy as np

        def rbf_kernel(X, Y, gamma=1.0):
            XX = np.sum(X**2, axis=1)[:, np.newaxis]
            YY = np.sum(Y**2, axis=1)[np.newaxis, :]
            XY = X @ Y.T
            distances = XX + YY - 2 * XY
            return np.exp(-gamma * distances)

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        mmd_squared = (
            K_XX.sum() / (len(X) * len(X)) +
            K_YY.sum() / (len(Y) * len(Y)) -
            2 * K_XY.sum() / (len(X) * len(Y))
        )

        return float(np.sqrt(max(0, mmd_squared)))


class DashboardDataProvider:
    """Provide data for monitoring dashboard.

    Aggregates data from PerformanceTracker, ResourceMonitor, and DriftDetector
    for dashboard visualization.

    Example:
        >>> provider = DashboardDataProvider(tracker, monitor, detector)
        >>> data = provider.get_dashboard_data()
    """

    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        resource_monitor: Optional[ResourceMonitor] = None,
        drift_detector: Optional[DriftDetector] = None
    ):
        """Initialize dashboard data provider.

        Args:
            performance_tracker: Performance tracker instance.
            resource_monitor: Optional resource monitor.
            drift_detector: Optional drift detector.
        """
        self.perf_tracker = performance_tracker
        self.resource_monitor = resource_monitor
        self.drift_detector = drift_detector

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for dashboard.

        Returns:
            Aggregated dashboard data.
        """
        # Performance statistics
        perf_stats = self.perf_tracker.get_statistics()

        # Resource utilization
        resource_data = {}
        if self.resource_monitor:
            samples = self.resource_monitor.get_recent_samples(60)
            if samples:
                import numpy as np
                resource_data = {
                    "cpu_avg": np.mean([s["cpu_percent"] for s in samples]),
                    "memory_avg": np.mean([s["memory_percent"] for s in samples]),
                    "samples": samples[-10:]  # Last 10 samples
                }

        # Drift status
        drift_status = {}
        if self.drift_detector:
            for adapter in self.perf_tracker.adapter_metrics.keys():
                drift_status[adapter] = self.drift_detector.detect_drift(adapter)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": perf_stats,
            "resources": resource_data,
            "drift_status": drift_status,
            "cost_estimate": self.perf_tracker.get_cost_estimate(),
            "summary": self._get_summary(perf_stats)
        }

    def _get_summary(self, stats: Dict) -> Dict[str, Any]:
        """Get summary for quick overview.

        Args:
            stats: Performance statistics.

        Returns:
            Summary dict.
        """
        return {
            "total_requests": stats.get("total_requests", 0),
            "success_rate": stats.get("success_rate", 0),
            "avg_latency_ms": stats.get("latency", {}).get("mean_ms", 0),
            "p95_latency_ms": stats.get("latency", {}).get("p95_ms", 0),
            "adapters_active": len(stats.get("adapters", {})),
            "health": "healthy" if stats.get("success_rate", 0) > 0.95 else "degraded"
        }

    def export_to_json(self, filepath: str) -> None:
        """Export dashboard data to JSON file.

        Args:
            filepath: Output file path.
        """
        import json

        data = self.get_dashboard_data()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported dashboard data to {filepath}")
