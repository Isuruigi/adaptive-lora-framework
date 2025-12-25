"""
Sophisticated logging system with multiple handlers and structured output.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- File and console handlers
- Structured JSON logging for production
- Integration with Weights & Biases
- Performance metrics tracking
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging.

    Produces machine-readable log entries suitable for log aggregation
    systems like ELK, Splunk, or CloudWatch.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)

        # Add any other custom attributes
        for key in ["metric", "value", "step", "adapter", "query_id"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Format logs with ANSI colors for console output.

    Makes log output more readable in terminal environments.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors.

        Args:
            record: The log record to format.

        Returns:
            Colored string representation of the log record.
        """
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format message
        message = record.getMessage()

        # Build formatted string
        formatted = (
            f"{self.BOLD}[{timestamp}]{self.RESET} "
            f"{color}{record.levelname:8}{self.RESET} "
            f"[{record.module}:{record.funcName}:{record.lineno}] "
            f"{message}"
        )

        # Add exception if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class StructuredLogger:
    """Production-grade logger with structured output and W&B integration.

    Provides comprehensive logging functionality including:
    - JSON structured logging for production environments
    - Colored console output for development
    - Weights & Biases integration for experiment tracking
    - Metric logging and tracking

    Example:
        >>> logger = StructuredLogger("training", use_wandb=True)
        >>> logger.info("Starting training", epoch=1, lr=2e-4)
        >>> logger.metric("loss", 0.5, step=100)
    """

    def __init__(
        self,
        name: str,
        log_dir: Path = Path("./logs"),
        level: int = logging.INFO,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        structured: bool = True,
        console_output: bool = True
    ):
        """Initialize the structured logger.

        Args:
            name: Logger name (usually module name).
            log_dir: Directory for log files.
            level: Logging level.
            use_wandb: Enable W&B logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity/team name.
            structured: Use JSON formatting for file logs.
            console_output: Enable console output.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.structured = structured

        # Prevent adding handlers multiple times
        if self.logger.handlers:
            return

        # Create log directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler with JSON formatting for production
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(level)

        if structured:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        self.logger.addHandler(file_handler)

        # Console handler with colored output
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(ColoredFormatter())
            self.logger.addHandler(console_handler)

        # Initialize W&B if requested
        if self.use_wandb and wandb_project:
            try:
                if not wandb.run:
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        reinit=True
                    )
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs: Any
    ) -> None:
        """Internal logging method.

        Args:
            level: Logging level.
            message: Log message.
            exc_info: Include exception info.
            **kwargs: Extra fields to include in log.
        """
        record = self.logger.makeRecord(
            self.name,
            level,
            "",
            0,
            message,
            (),
            None
        )
        record.extra = kwargs

        for key, value in kwargs.items():
            setattr(record, key, value)

        self.logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message.
            **kwargs: Extra fields to include.
        """
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message.
            **kwargs: Extra fields to include.
        """
        self._log(logging.INFO, message, **kwargs)

        if self.use_wandb and kwargs:
            try:
                wandb.log({"info": message, **kwargs})
            except Exception:
                pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message.
            **kwargs: Extra fields to include.
        """
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Log message.
            exc_info: Include exception traceback.
            **kwargs: Extra fields to include.
        """
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log critical message.

        Args:
            message: Log message.
            exc_info: Include exception traceback.
            **kwargs: Extra fields to include.
        """
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/iteration number.
            **kwargs: Additional metric metadata.
        """
        self._log(
            logging.INFO,
            f"Metric: {name} = {value}",
            metric=name,
            value=value,
            step=step,
            **kwargs
        )

        if self.use_wandb:
            try:
                log_data = {name: value, **kwargs}
                wandb.log(log_data, step=step)
            except Exception:
                pass

    def metrics(
        self,
        metrics_dict: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric names to values.
            step: Optional step/iteration number.
        """
        for name, value in metrics_dict.items():
            self.metric(name, value, step=step)

    def training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        epoch: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Log training step information.

        Args:
            step: Current training step.
            loss: Training loss.
            lr: Learning rate.
            epoch: Current epoch.
            **kwargs: Additional training metrics.
        """
        message = f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}"
        if epoch is not None:
            message = f"Epoch {epoch} | " + message

        self.info(message, step=step, loss=loss, lr=lr, epoch=epoch, **kwargs)

    def evaluation_results(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        split: str = "eval"
    ) -> None:
        """Log evaluation results.

        Args:
            metrics: Evaluation metrics dictionary.
            step: Current step.
            split: Dataset split (train, eval, test).
        """
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        message = f"[{split.upper()}] {metrics_str}"

        prefixed_metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        self.info(message, step=step, split=split, **prefixed_metrics)


def get_logger(
    name: str,
    level: str = "INFO",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None
) -> StructuredLogger:
    """Get a configured logger instance.

    Convenience function for getting a logger with common settings.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        use_wandb: Enable W&B integration.
        wandb_project: W&B project name.

    Returns:
        Configured StructuredLogger instance.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return StructuredLogger(
        name=name,
        level=level_map.get(level.upper(), logging.INFO),
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )
