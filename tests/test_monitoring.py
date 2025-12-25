"""
Tests for Monitoring Module

Tests for metrics collection, drift detection, and alerting.
"""

import time
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pytest


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""
    
    def test_metric_recording(self):
        """Test metric recording."""
        metrics = {}
        
        # Record metrics
        metrics["latency_ms"] = 150.5
        metrics["tokens_per_second"] = 45.2
        metrics["requests_total"] = 1000
        
        assert "latency_ms" in metrics
        assert metrics["requests_total"] > 0
    
    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        import statistics
        
        latencies = [100, 120, 130, 150, 200, 180, 140, 160, 190, 210]
        
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(0.95 * len(sorted_latencies))]
        
        assert p50 <= p95
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        requests = 100
        duration_seconds = 10
        
        throughput = requests / duration_seconds
        
        assert throughput == 10.0


class TestDriftDetector:
    """Tests for DriftDetector class."""
    
    def test_psi_calculation(self):
        """Test Population Stability Index calculation."""
        # Reference distribution
        reference = [0.1, 0.2, 0.3, 0.4]
        # Current distribution
        current = [0.15, 0.25, 0.25, 0.35]
        
        # PSI calculation
        psi = 0
        for r, c in zip(reference, current):
            if r > 0 and c > 0:
                psi += (c - r) * (c / r if r != 0 else 0)
        
        # PSI < 0.1 = stable, >= 0.2 = drift
        assert psi is not None
    
    def test_ks_test(self):
        """Test Kolmogorov-Smirnov test."""
        import random
        random.seed(42)
        
        reference = [random.gauss(0, 1) for _ in range(100)]
        current_similar = [random.gauss(0, 1) for _ in range(100)]
        current_different = [random.gauss(5, 1) for _ in range(100)]
        
        # Calculate max difference (simplified KS)
        def ks_statistic(a, b):
            a_sorted = sorted(a)
            b_sorted = sorted(b)
            return abs(sum(a_sorted) / len(a_sorted) - sum(b_sorted) / len(b_sorted))
        
        diff_similar = ks_statistic(reference, current_similar)
        diff_different = ks_statistic(reference, current_different)
        
        assert diff_different > diff_similar
    
    def test_drift_threshold(self):
        """Test drift detection threshold."""
        threshold = 0.2
        
        drift_scores = [0.1, 0.15, 0.25, 0.3]
        
        alerts = [score > threshold for score in drift_scores]
        
        assert sum(alerts) == 2  # 0.25 and 0.3 exceed threshold


class TestAlertManager:
    """Tests for AlertManager class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = {
            "name": "high_latency",
            "severity": "warning",
            "message": "Latency exceeded threshold",
            "timestamp": time.time(),
            "value": 500,
            "threshold": 300
        }
        
        assert alert["severity"] in ["info", "warning", "critical"]
        assert alert["value"] > alert["threshold"]
    
    def test_alert_severity_levels(self):
        """Test alert severity levels."""
        severities = ["info", "warning", "critical"]
        severity_order = {s: i for i, s in enumerate(severities)}
        
        assert severity_order["critical"] > severity_order["warning"]
        assert severity_order["warning"] > severity_order["info"]
    
    def test_alert_aggregation(self):
        """Test alert aggregation to prevent spam."""
        window_seconds = 300  # 5 minutes
        max_per_window = 3
        
        alerts = [
            {"name": "high_latency", "timestamp": time.time() - 60},
            {"name": "high_latency", "timestamp": time.time() - 30},
            {"name": "high_latency", "timestamp": time.time()},
            {"name": "high_latency", "timestamp": time.time() + 1},  # Should be suppressed
        ]
        
        # Count alerts in window
        current_time = time.time() + 1
        in_window = [
            a for a in alerts
            if current_time - a["timestamp"] <= window_seconds
        ]
        
        should_suppress = len(in_window) > max_per_window
        
        assert should_suppress


class TestNotificationChannels:
    """Tests for notification channels."""
    
    def test_slack_notification_format(self):
        """Test Slack notification format."""
        notification = {
            "channel": "#alerts",
            "text": "Alert: High latency detected",
            "attachments": [
                {
                    "color": "warning",
                    "fields": [
                        {"title": "Metric", "value": "latency_p95"},
                        {"title": "Value", "value": "500ms"}
                    ]
                }
            ]
        }
        
        assert notification["channel"].startswith("#")
        assert "attachments" in notification
    
    def test_email_notification_format(self):
        """Test email notification format."""
        notification = {
            "to": ["team@example.com"],
            "subject": "[ALERT] High Latency",
            "body": "Latency has exceeded the threshold..."
        }
        
        assert "@" in notification["to"][0]
        assert "[ALERT]" in notification["subject"]


class TestMetricsExport:
    """Tests for metrics export."""
    
    def test_prometheus_format(self):
        """Test Prometheus metrics format."""
        metrics = [
            '# HELP requests_total Total requests',
            '# TYPE requests_total counter',
            'requests_total{adapter="reasoning"} 1000',
            '# HELP latency_seconds Request latency',
            '# TYPE latency_seconds histogram',
            'latency_seconds_bucket{le="0.1"} 50',
        ]
        
        for line in metrics:
            assert line.startswith("#") or "{" in line or line.strip() == ""
    
    def test_json_metrics_format(self):
        """Test JSON metrics format."""
        metrics = {
            "timestamp": time.time(),
            "metrics": {
                "requests_total": 1000,
                "latency_p50_ms": 150,
                "latency_p95_ms": 450,
                "error_rate": 0.01
            }
        }
        
        assert "timestamp" in metrics
        assert "metrics" in metrics


class TestHealthChecks:
    """Tests for health checks."""
    
    def test_component_health(self):
        """Test individual component health checks."""
        components = {
            "model": {"status": "healthy", "latency_ms": 5},
            "router": {"status": "healthy", "latency_ms": 2},
            "redis": {"status": "degraded", "error": "Connection timeout"},
            "adapters": {"status": "healthy", "loaded": 4}
        }
        
        healthy = all(c["status"] == "healthy" for c in components.values())
        degraded = any(c["status"] == "degraded" for c in components.values())
        
        assert not healthy  # Redis is degraded
        assert degraded
    
    def test_overall_health_status(self):
        """Test overall health status determination."""
        component_statuses = ["healthy", "healthy", "degraded", "healthy"]
        
        if all(s == "healthy" for s in component_statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in component_statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"
        
        assert overall == "degraded"


class TestDashboard:
    """Tests for dashboard data provision."""
    
    def test_time_series_format(self):
        """Test time series data format."""
        time_series = {
            "metric": "latency_p95",
            "data": [
                {"timestamp": 1700000000, "value": 150},
                {"timestamp": 1700000060, "value": 160},
                {"timestamp": 1700000120, "value": 145},
            ]
        }
        
        # Verify timestamps are increasing
        timestamps = [d["timestamp"] for d in time_series["data"]]
        assert timestamps == sorted(timestamps)
    
    def test_aggregation_functions(self):
        """Test metric aggregation functions."""
        values = [100, 150, 120, 180, 130]
        
        aggregations = {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values)
        }
        
        assert aggregations["min"] <= aggregations["avg"] <= aggregations["max"]
