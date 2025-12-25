"""
GPU utilities for memory management and device handling.

Features:
- GPU memory monitoring
- Device allocation
- Memory optimization
- Multi-GPU support
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Check for GPU availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class GPUInfo:
    """GPU device information."""
    
    device_id: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float
    temperature_c: Optional[float] = None
    
    @property
    def memory_utilization(self) -> float:
        """Memory utilization as fraction."""
        return self.used_memory_gb / self.total_memory_gb if self.total_memory_gb > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "free_memory_gb": round(self.free_memory_gb, 2),
            "used_memory_gb": round(self.used_memory_gb, 2),
            "utilization_percent": round(self.utilization_percent, 1),
            "memory_utilization": round(self.memory_utilization * 100, 1),
            "temperature_c": self.temperature_c,
        }


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device_count() -> int:
    """Get number of available GPUs."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_device(device: Union[str, int, None] = None) -> "torch.device":
    """Get torch device.
    
    Args:
        device: Device specification ('cuda', 'cpu', device id, or None for auto).
        
    Returns:
        torch.device object.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"
        
    return torch.device(device)


def get_gpu_info(device_id: int = 0) -> Optional[GPUInfo]:
    """Get GPU information.
    
    Args:
        device_id: GPU device ID.
        
    Returns:
        GPUInfo or None if not available.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
        
    if device_id >= torch.cuda.device_count():
        return None
        
    try:
        props = torch.cuda.get_device_properties(device_id)
        
        # Memory info
        total = props.total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        free = total - reserved
        
        # Try to get utilization (requires pynvml)
        utilization = 0.0
        temperature = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
        except Exception:
            pass
            
        return GPUInfo(
            device_id=device_id,
            name=props.name,
            total_memory_gb=total,
            free_memory_gb=free,
            used_memory_gb=allocated,
            utilization_percent=utilization,
            temperature_c=temperature
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def get_all_gpu_info() -> List[GPUInfo]:
    """Get info for all GPUs."""
    infos = []
    for i in range(get_device_count()):
        info = get_gpu_info(i)
        if info:
            infos.append(info)
    return infos


def get_best_gpu() -> int:
    """Get GPU with most free memory.
    
    Returns:
        Best GPU device ID, or 0 if none available.
    """
    infos = get_all_gpu_info()
    
    if not infos:
        return 0
        
    best = max(infos, key=lambda x: x.free_memory_gb)
    return best.device_id


def clear_gpu_memory(device_id: Optional[int] = None) -> None:
    """Clear GPU memory cache.
    
    Args:
        device_id: Specific device to clear, or all if None.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return
        
    gc.collect()
    
    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()
        
    logger.debug("GPU memory cache cleared")


def optimize_memory_usage() -> None:
    """Apply memory optimization settings."""
    if not TORCH_AVAILABLE:
        return
        
    # Enable memory efficient attention if available
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # Enable cudnn benchmark for consistent input sizes
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        
    # Set memory fraction limit
    if torch.cuda.is_available():
        fraction = float(os.getenv("CUDA_MEMORY_FRACTION", "0.9"))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass
            
    logger.info("GPU memory optimizations applied")


def estimate_model_memory(
    num_params: int,
    dtype: str = "float16",
    include_gradients: bool = True,
    include_optimizer: bool = True
) -> float:
    """Estimate memory required for model.
    
    Args:
        num_params: Number of model parameters.
        dtype: Data type ('float32', 'float16', 'int8', 'int4').
        include_gradients: Include gradient memory.
        include_optimizer: Include optimizer states (Adam).
        
    Returns:
        Estimated memory in GB.
    """
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    param_bytes = bytes_per_param.get(dtype, 2)
    
    # Base model memory
    memory = num_params * param_bytes
    
    # Gradients (same size as params)
    if include_gradients:
        memory += num_params * param_bytes
        
    # Optimizer states (Adam: 2 states per param)
    if include_optimizer:
        memory += num_params * param_bytes * 2
        
    # Activation memory (rough estimate: 2x params)
    memory += num_params * param_bytes * 0.5
    
    # Convert to GB with 20% buffer
    return (memory / (1024**3)) * 1.2


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None
) -> Tuple[int, int]:
    """Setup distributed training.
    
    Args:
        backend: Distributed backend.
        init_method: Initialization method.
        
    Returns:
        Tuple of (rank, world_size).
    """
    if not TORCH_AVAILABLE:
        return 0, 1
        
    import torch.distributed as dist
    
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
        
    # Check for SLURM or other job scheduler
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
    if world_size > 1:
        init_method = init_method or "env://"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        
    return rank, world_size


class MemoryTracker:
    """Track GPU memory usage over time."""
    
    def __init__(self, device_id: int = 0):
        """Initialize tracker.
        
        Args:
            device_id: GPU device to track.
        """
        self.device_id = device_id
        self.snapshots: List[Dict[str, Any]] = []
        
    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take memory snapshot.
        
        Args:
            label: Optional label for snapshot.
            
        Returns:
            Snapshot data.
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
            
        import time
        
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "allocated_gb": torch.cuda.memory_allocated(self.device_id) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(self.device_id) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device_id) / (1024**3),
        }
        
        self.snapshots.append(snapshot)
        return snapshot
        
    def reset_peak(self) -> None:
        """Reset peak memory stats."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated(self.device_id)
            torch.cuda.reset_max_memory_cached(self.device_id)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get memory tracking summary."""
        if not self.snapshots:
            return {}
            
        allocated = [s["allocated_gb"] for s in self.snapshots]
        
        return {
            "num_snapshots": len(self.snapshots),
            "min_allocated_gb": min(allocated),
            "max_allocated_gb": max(allocated),
            "avg_allocated_gb": sum(allocated) / len(allocated),
            "snapshots": self.snapshots,
        }


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics.
    
    Returns:
        Dictionary of memory stats.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}
        
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }
