"""
General utility functions and decorators for the framework.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

T = TypeVar("T")


@contextmanager
def timer(name: str = "Operation") -> Generator[Dict[str, float], None, None]:
    """Context manager for timing code blocks.

    Example:
        >>> with timer("Training") as t:
        ...     train_model()
        >>> print(f"Training took {t['elapsed']:.2f}s")

    Args:
        name: Name of the operation being timed.

    Yields:
        Dictionary that will contain 'elapsed' time after exiting context.
    """
    result: Dict[str, float] = {}
    start_time = time.perf_counter()

    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start_time
        result["elapsed"] = elapsed
        print(f"{name} completed in {elapsed:.3f}s")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def call_api():
        ...     return requests.get("https://api.example.com")

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        exceptions: Tuple of exception types to catch.

    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise

                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

            raise last_exception  # type: ignore

        return wrapper
    return decorator


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]

    Args:
        lst: List to split.
        chunk_size: Size of each chunk.

    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def hash_dict(d: Dict[str, Any]) -> str:
    """Create a deterministic hash of a dictionary.

    Useful for caching and deduplication.

    Args:
        d: Dictionary to hash.

    Returns:
        SHA256 hash string.
    """
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys.
        sep: Separator between nested keys.

    Returns:
        Flattened dictionary.
    """
    items: List[tuple] = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten a dictionary with dotted keys.

    Example:
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {"a": {"b": 1, "c": 2}}

    Args:
        d: Flattened dictionary.
        sep: Separator used in flattened keys.

    Returns:
        Nested dictionary.
    """
    result: Dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def format_number(num: float, precision: int = 2) -> str:
    """Format a number with K/M/B suffixes.

    Example:
        >>> format_number(1234567)
        '1.23M'

    Args:
        num: Number to format.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{unit}"
        num /= 1000.0

    return f"{num:.{precision}f}T"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time.

    Example:
        >>> format_time(3661)
        '1h 1m 1s'

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory usage information.

    Returns:
        Dictionary with GPU memory statistics.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        device_count = torch.cuda.device_count()
        info = {
            "available": True,
            "device_count": device_count,
            "devices": []
        }

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)

            info["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
            })

        return info

    except ImportError:
        return {"available": False, "error": "torch not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: Only count trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class AverageMeter:
    """Computes and stores the average and current value.

    Useful for tracking training metrics.

    Example:
        >>> losses = AverageMeter("loss")
        >>> losses.update(0.5)
        >>> losses.update(0.3)
        >>> print(losses.avg)
        0.4
    """

    def __init__(self, name: str = "metric"):
        """Initialize meter.

        Args:
            name: Name of the metric being tracked.
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with new value.

        Args:
            val: New value.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
