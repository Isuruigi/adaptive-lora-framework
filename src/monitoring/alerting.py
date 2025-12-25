"""
Alerting system for monitoring.

Features:
- Alert conditions and thresholds
- Multiple notification channels
- Alert aggregation and deduplication
- Escalation policies
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class Alert:
    """Alert instance."""
    
    name: str
    severity: AlertSeverity
    message: str
    status: AlertStatus = AlertStatus.FIRING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    @property
    def fingerprint(self) -> str:
        """Unique identifier for this alert."""
        content = f"{self.name}|{sorted(self.labels.items())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @property
    def duration(self) -> timedelta:
        """Alert duration."""
        end = self.resolved_at or datetime.utcnow()
        return end - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "status": self.status.value,
            "labels": self.labels,
            "annotations": self.annotations,
            "value": self.value,
            "threshold": self.threshold,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "fingerprint": self.fingerprint,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    
    name: str
    condition: Callable[[float], bool]
    severity: AlertSeverity
    message_template: str
    for_duration: timedelta = timedelta(seconds=0)  # Must be firing for this long
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # State tracking
    _pending_since: Optional[datetime] = field(default=None, repr=False)
    _firing: bool = field(default=False, repr=False)
    
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate rule against value.
        
        Args:
            value: Metric value to check.
            
        Returns:
            Alert if firing, None otherwise.
        """
        condition_met = self.condition(value)
        now = datetime.utcnow()
        
        if condition_met:
            if self._pending_since is None:
                self._pending_since = now
                
            # Check if condition has been met for required duration
            if now - self._pending_since >= self.for_duration:
                self._firing = True
                
                message = self.message_template.format(
                    value=value,
                    name=self.name
                )
                
                return Alert(
                    name=self.name,
                    severity=self.severity,
                    message=message,
                    labels=self.labels,
                    annotations=self.annotations,
                    value=value
                )
        else:
            self._pending_since = None
            self._firing = False
            
        return None


class NotificationChannel(ABC):
    """Abstract notification channel."""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert notification.
        
        Args:
            alert: Alert to send.
            
        Returns:
            True if sent successfully.
        """
        pass


class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        sender: str = "alerts@example.com",
        recipients: Optional[List[str]] = None
    ):
        """Initialize email channel."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients or []
        
    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            subject = f"[{alert.severity.value.upper()}] {alert.name}"
            body = self._format_email(alert)
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.sendmail(self.sender, self.recipients, msg.as_string())
                
            logger.info(f"Email alert sent for {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
            
    def _format_email(self, alert: Alert) -> str:
        """Format alert as email body."""
        return f"""
Alert: {alert.name}
Severity: {alert.severity.value}
Status: {alert.status.value}

Message: {alert.message}

Value: {alert.value}
Started: {alert.started_at}
Duration: {alert.duration}

Labels: {alert.labels}
"""


class SlackChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None
    ):
        """Initialize Slack channel."""
        self.webhook_url = webhook_url
        self.channel = channel
        
    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            import aiohttp
            
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#FFA500",
                AlertSeverity.ERROR: "#FF6347",
                AlertSeverity.CRITICAL: "#DC143C",
            }.get(alert.severity, "#808080")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"[{alert.severity.value.upper()}] {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Status", "value": alert.status.value, "short": True},
                        {"title": "Value", "value": str(alert.value), "short": True},
                    ],
                    "ts": int(alert.started_at.timestamp())
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
                
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 200
                    
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class PagerDutyChannel(NotificationChannel):
    """PagerDuty notification channel."""
    
    def __init__(
        self,
        routing_key: str,
        severity_mapping: Optional[Dict[AlertSeverity, str]] = None
    ):
        """Initialize PagerDuty channel."""
        self.routing_key = routing_key
        self.severity_mapping = severity_mapping or {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }
        
    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        try:
            import aiohttp
            
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger" if alert.status == AlertStatus.FIRING else "resolve",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.message,
                    "severity": self.severity_mapping.get(alert.severity, "info"),
                    "source": "adaptive-lora-framework",
                    "custom_details": {
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "labels": alert.labels,
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload
                ) as resp:
                    return resp.status == 202
                    
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False


class ConsoleChannel(NotificationChannel):
    """Console/log notification channel."""
    
    async def send(self, alert: Alert) -> bool:
        """Log alert to console."""
        log_level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)
        
        log_level(f"ALERT [{alert.severity.value}] {alert.name}: {alert.message}")
        return True


class AlertManager:
    """Central alert management."""
    
    def __init__(
        self,
        channels: Optional[List[NotificationChannel]] = None,
        check_interval: float = 30.0,
        dedup_window: timedelta = timedelta(minutes=5)
    ):
        """Initialize alert manager.
        
        Args:
            channels: Notification channels.
            check_interval: Seconds between rule evaluations.
            dedup_window: Time window for deduplication.
        """
        self.channels = channels or [ConsoleChannel()]
        self.check_interval = check_interval
        self.dedup_window = dedup_window
        
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self._sent_alerts: Dict[str, datetime] = {}  # fingerprint -> last sent
        self._running = False
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
        
    def add_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self.channels.append(channel)
        
    async def check_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check all rules against current metrics.
        
        Args:
            metrics: Current metric values.
            
        Returns:
            List of firing alerts.
        """
        alerts = []
        
        for rule in self.rules:
            metric_name = rule.labels.get("metric", rule.name)
            
            if metric_name in metrics:
                value = metrics[metric_name]
                alert = rule.evaluate(value)
                
                if alert:
                    alerts.append(alert)
                    
        return alerts
        
    async def process_alerts(self, alerts: List[Alert]) -> None:
        """Process and send alerts.
        
        Args:
            alerts: Alerts to process.
        """
        now = datetime.utcnow()
        
        for alert in alerts:
            fingerprint = alert.fingerprint
            
            # Update or create active alert
            if fingerprint not in self.active_alerts:
                self.active_alerts[fingerprint] = alert
                alert.status = AlertStatus.FIRING
            else:
                # Update existing alert
                existing = self.active_alerts[fingerprint]
                alert.started_at = existing.started_at
                self.active_alerts[fingerprint] = alert
                
            # Check deduplication
            last_sent = self._sent_alerts.get(fingerprint)
            if last_sent and now - last_sent < self.dedup_window:
                continue
                
            # Send to all channels
            await self._send_alert(alert)
            self._sent_alerts[fingerprint] = now
            
    async def _send_alert(self, alert: Alert) -> None:
        """Send alert to all channels."""
        results = await asyncio.gather(
            *[channel.send(alert) for channel in self.channels],
            return_exceptions=True
        )
        
        success = sum(1 for r in results if r is True)
        logger.debug(f"Alert {alert.name} sent to {success}/{len(self.channels)} channels")
        
    async def resolve_alert(self, fingerprint: str) -> Optional[Alert]:
        """Resolve an active alert.
        
        Args:
            fingerprint: Alert fingerprint.
            
        Returns:
            Resolved alert if found.
        """
        if fingerprint in self.active_alerts:
            alert = self.active_alerts.pop(fingerprint)
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            await self._send_alert(alert)
            return alert
            
        return None
        
    def silence_alert(
        self,
        fingerprint: str,
        duration: timedelta = timedelta(hours=1)
    ) -> bool:
        """Silence an alert.
        
        Args:
            fingerprint: Alert fingerprint.
            duration: Silence duration.
            
        Returns:
            True if silenced.
        """
        if fingerprint in self.active_alerts:
            alert = self.active_alerts[fingerprint]
            alert.status = AlertStatus.SILENCED
            # Add silence end time to dedup to prevent sending
            self._sent_alerts[fingerprint] = datetime.utcnow() + duration
            return True
        return False
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
        
    def create_default_rules(self) -> None:
        """Create standard monitoring rules."""
        # Latency alert
        self.add_rule(AlertRule(
            name="high_latency",
            condition=lambda v: v > 1000,  # > 1000ms
            severity=AlertSeverity.WARNING,
            message_template="High latency detected: {value}ms",
            for_duration=timedelta(minutes=2),
            labels={"metric": "latency_p99"}
        ))
        
        # Error rate alert
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda v: v > 0.05,  # > 5%
            severity=AlertSeverity.ERROR,
            message_template="Error rate above threshold: {value:.1%}",
            for_duration=timedelta(minutes=1),
            labels={"metric": "error_rate"}
        ))
        
        # Quality degradation
        self.add_rule(AlertRule(
            name="quality_degradation",
            condition=lambda v: v < 0.7,  # < 0.7 quality score
            severity=AlertSeverity.WARNING,
            message_template="Response quality below threshold: {value:.2f}",
            for_duration=timedelta(minutes=5),
            labels={"metric": "quality_score"}
        ))
        
        # GPU memory alert
        self.add_rule(AlertRule(
            name="high_gpu_memory",
            condition=lambda v: v > 0.9,  # > 90% utilization
            severity=AlertSeverity.WARNING,
            message_template="GPU memory usage high: {value:.1%}",
            labels={"metric": "gpu_memory_utilization"}
        ))
