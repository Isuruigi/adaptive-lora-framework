"""
Dashboard data provider for Grafana integration.

Features:
- Metric aggregation for dashboards
- Time-series data formatting
- Grafana JSON API compatibility
- Real-time metrics streaming
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesPoint:
    """Single data point in time series."""
    
    timestamp: float  # Unix timestamp
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_grafana(self) -> List[Any]:
        """Format for Grafana."""
        return [self.value, int(self.timestamp * 1000)]


@dataclass
class TimeSeries:
    """Time series data."""
    
    name: str
    points: List[TimeSeriesPoint] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add data point."""
        ts = timestamp or time.time()
        self.points.append(TimeSeriesPoint(timestamp=ts, value=value))
        
    def get_range(
        self,
        start: float,
        end: float
    ) -> List[TimeSeriesPoint]:
        """Get points in time range."""
        return [p for p in self.points if start <= p.timestamp <= end]
        
    def to_grafana(self) -> Dict[str, Any]:
        """Format as Grafana time series."""
        return {
            "target": self.name,
            "datapoints": [p.to_grafana() for p in self.points]
        }
        
    def downsample(self, interval_seconds: float) -> "TimeSeries":
        """Downsample to specified interval."""
        if not self.points:
            return TimeSeries(name=self.name, labels=self.labels)
            
        buckets: Dict[int, List[float]] = defaultdict(list)
        
        for point in self.points:
            bucket = int(point.timestamp / interval_seconds)
            buckets[bucket].append(point.value)
            
        new_points = []
        for bucket, values in sorted(buckets.items()):
            avg_value = sum(values) / len(values)
            timestamp = bucket * interval_seconds
            new_points.append(TimeSeriesPoint(timestamp=timestamp, value=avg_value))
            
        return TimeSeries(name=self.name, points=new_points, labels=self.labels)


class MetricStore:
    """In-memory metric store with retention."""
    
    def __init__(
        self,
        retention_hours: int = 24,
        cleanup_interval_minutes: int = 10
    ):
        """Initialize store.
        
        Args:
            retention_hours: How long to keep data.
            cleanup_interval_minutes: Cleanup frequency.
        """
        self.retention = timedelta(hours=retention_hours)
        self.cleanup_interval = cleanup_interval_minutes * 60
        
        self._series: Dict[str, TimeSeries] = {}
        self._last_cleanup = time.time()
        
    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Record metric value.
        
        Args:
            name: Metric name.
            value: Metric value.
            labels: Optional labels.
            timestamp: Optional timestamp.
        """
        key = self._make_key(name, labels or {})
        
        if key not in self._series:
            self._series[key] = TimeSeries(name=name, labels=labels or {})
            
        self._series[key].add_point(value, timestamp)
        
        # Periodic cleanup
        if time.time() - self._last_cleanup > self.cleanup_interval:
            self._cleanup()
            
    def query(
        self,
        name: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[TimeSeries]:
        """Query metrics.
        
        Args:
            name: Metric name (supports wildcards with *).
            start: Start timestamp.
            end: End timestamp.
            labels: Label filters.
            
        Returns:
            List of matching time series.
        """
        end = end or time.time()
        start = start or (end - 3600)  # Default 1 hour
        
        results = []
        
        for key, series in self._series.items():
            if not self._matches(series.name, name):
                continue
                
            if labels and not self._labels_match(series.labels, labels):
                continue
                
            points = series.get_range(start, end)
            if points:
                filtered = TimeSeries(
                    name=series.name,
                    points=points,
                    labels=series.labels
                )
                results.append(filtered)
                
        return results
        
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key from name and labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
        
    def _matches(self, actual: str, pattern: str) -> bool:
        """Check if actual matches pattern."""
        if pattern == "*":
            return True
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(actual, pattern)
        return actual == pattern
        
    def _labels_match(
        self,
        actual: Dict[str, str],
        required: Dict[str, str]
    ) -> bool:
        """Check if labels match."""
        for key, value in required.items():
            if actual.get(key) != value:
                return False
        return True
        
    def _cleanup(self) -> None:
        """Remove old data points."""
        cutoff = time.time() - self.retention.total_seconds()
        
        for series in self._series.values():
            series.points = [p for p in series.points if p.timestamp >= cutoff]
            
        # Remove empty series
        self._series = {k: v for k, v in self._series.items() if v.points}
        
        self._last_cleanup = time.time()
        logger.debug(f"Metric store cleanup complete, {len(self._series)} series retained")


class DashboardProvider:
    """Provide data for Grafana dashboards."""
    
    def __init__(self, metric_store: Optional[MetricStore] = None):
        """Initialize provider.
        
        Args:
            metric_store: Metric storage backend.
        """
        self.store = metric_store or MetricStore()
        
    def query(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle Grafana query request.
        
        Args:
            request: Grafana query request.
            
        Returns:
            Grafana-formatted response.
        """
        results = []
        
        targets = request.get("targets", [])
        range_data = request.get("range", {})
        
        start = self._parse_time(range_data.get("from"))
        end = self._parse_time(range_data.get("to"))
        interval = request.get("intervalMs", 60000) / 1000
        
        for target in targets:
            target_name = target.get("target", "")
            
            series_list = self.store.query(target_name, start, end)
            
            for series in series_list:
                # Downsample if needed
                if interval > 1:
                    series = series.downsample(interval)
                results.append(series.to_grafana())
                
        return results
        
    def search(self, request: Dict[str, Any]) -> List[str]:
        """Handle Grafana search request.
        
        Args:
            request: Search request.
            
        Returns:
            List of available metrics.
        """
        return list(set(s.name for s in self.store._series.values()))
        
    def annotations(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle Grafana annotations request.
        
        Args:
            request: Annotations request.
            
        Returns:
            List of annotations.
        """
        # Placeholder for alert annotations
        return []
        
    def _parse_time(self, time_str: Optional[str]) -> float:
        """Parse Grafana time string."""
        if not time_str:
            return time.time()
            
        try:
            # ISO format
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except ValueError:
            return time.time()
            
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get pre-configured system dashboard.
        
        Returns:
            Grafana dashboard JSON.
        """
        return {
            "title": "Adaptive LoRA System",
            "uid": "adaptive-lora",
            "panels": [
                self._latency_panel(),
                self._throughput_panel(),
                self._quality_panel(),
                self._adapter_usage_panel(),
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "10s"
        }
        
    def _latency_panel(self) -> Dict[str, Any]:
        """Create latency panel definition."""
        return {
            "title": "Request Latency",
            "type": "graph",
            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
            "targets": [
                {"target": "latency_p50", "alias": "P50"},
                {"target": "latency_p95", "alias": "P95"},
                {"target": "latency_p99", "alias": "P99"},
            ],
            "yaxes": [{"format": "ms", "label": "Latency"}]
        }
        
    def _throughput_panel(self) -> Dict[str, Any]:
        """Create throughput panel definition."""
        return {
            "title": "Throughput",
            "type": "graph",
            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
            "targets": [
                {"target": "requests_per_second", "alias": "Requests/sec"},
            ],
            "yaxes": [{"format": "reqps", "label": "Throughput"}]
        }
        
    def _quality_panel(self) -> Dict[str, Any]:
        """Create quality panel definition."""
        return {
            "title": "Response Quality",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 8, "w": 8, "h": 6},
            "targets": [
                {"target": "quality_score", "alias": "Quality"},
            ],
            "options": {
                "minValue": 0,
                "maxValue": 1,
                "thresholds": [
                    {"value": 0, "color": "red"},
                    {"value": 0.6, "color": "yellow"},
                    {"value": 0.8, "color": "green"},
                ]
            }
        }
        
    def _adapter_usage_panel(self) -> Dict[str, Any]:
        """Create adapter usage panel definition."""
        return {
            "title": "Adapter Usage",
            "type": "piechart",
            "gridPos": {"x": 8, "y": 8, "w": 8, "h": 6},
            "targets": [
                {"target": "adapter_usage_*", "alias": "$1"},
            ]
        }


class MetricAggregator:
    """Aggregate metrics for dashboard display."""
    
    def __init__(self, store: MetricStore):
        """Initialize aggregator.
        
        Args:
            store: Metric store.
        """
        self.store = store
        
    def get_summary(
        self,
        metric_name: str,
        window_seconds: float = 300
    ) -> Dict[str, float]:
        """Get metric summary statistics.
        
        Args:
            metric_name: Metric to summarize.
            window_seconds: Time window.
            
        Returns:
            Summary statistics.
        """
        end = time.time()
        start = end - window_seconds
        
        series_list = self.store.query(metric_name, start, end)
        
        if not series_list:
            return {}
            
        all_values = []
        for series in series_list:
            all_values.extend([p.value for p in series.points])
            
        if not all_values:
            return {}
            
        import statistics
        
        return {
            "count": len(all_values),
            "min": min(all_values),
            "max": max(all_values),
            "mean": statistics.mean(all_values),
            "median": statistics.median(all_values),
            "stddev": statistics.stdev(all_values) if len(all_values) > 1 else 0,
        }
        
    def get_percentiles(
        self,
        metric_name: str,
        percentiles: List[float] = [50, 95, 99],
        window_seconds: float = 300
    ) -> Dict[str, float]:
        """Get percentile values.
        
        Args:
            metric_name: Metric name.
            percentiles: Percentiles to compute.
            window_seconds: Time window.
            
        Returns:
            Percentile values.
        """
        end = time.time()
        start = end - window_seconds
        
        series_list = self.store.query(metric_name, start, end)
        
        all_values = []
        for series in series_list:
            all_values.extend([p.value for p in series.points])
            
        if not all_values:
            return {}
            
        all_values.sort()
        n = len(all_values)
        
        result = {}
        for p in percentiles:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            result[f"p{int(p)}"] = all_values[idx]
            
        return result
        
    def rate(
        self,
        metric_name: str,
        window_seconds: float = 60
    ) -> float:
        """Compute rate (change per second).
        
        Args:
            metric_name: Counter metric name.
            window_seconds: Window for rate calculation.
            
        Returns:
            Rate value.
        """
        end = time.time()
        start = end - window_seconds
        
        series_list = self.store.query(metric_name, start, end)
        
        if not series_list or not series_list[0].points:
            return 0.0
            
        points = series_list[0].points
        
        if len(points) < 2:
            return 0.0
            
        first = points[0]
        last = points[-1]
        
        time_diff = last.timestamp - first.timestamp
        value_diff = last.value - first.value
        
        if time_diff <= 0:
            return 0.0
            
        return value_diff / time_diff
