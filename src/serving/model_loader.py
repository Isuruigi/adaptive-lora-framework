"""
Model loading utilities for efficient model management.

Features:
- Lazy loading
- Multi-model management
- Memory optimization
- Adapter hot-swapping
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from src.utils.logger import get_logger
from src.utils.helpers import get_gpu_memory_info, format_number

logger = get_logger(__name__)


class ModelLoader:
    """Efficient model loading and management.

    Provides utilities for loading, caching, and managing
    model resources.

    Example:
        >>> loader = ModelLoader()
        >>> model, tokenizer = loader.load_model("meta-llama/Llama-3-8B")
        >>> loader.unload_model("meta-llama/Llama-3-8B")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_cached_models: int = 2,
        default_dtype: str = "bfloat16"
    ):
        """Initialize model loader.

        Args:
            cache_dir: Directory for caching models.
            max_cached_models: Maximum models to keep in memory.
            default_dtype: Default dtype for loading models.
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "adaptive-lora"
        self.max_cached_models = max_cached_models
        self.default_dtype = default_dtype

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Loaded models cache
        self._model_cache: Dict[str, Tuple[PreTrainedModel, Any]] = {}
        self._load_order: List[str] = []

        # Dtype mapping
        self.dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

    def load_model(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        use_flash_attention: bool = True,
        trust_remote_code: bool = False
    ) -> Tuple[PreTrainedModel, Any]:
        """Load a model and tokenizer.

        Args:
            model_name: HuggingFace model ID or local path.
            load_in_4bit: Use 4-bit quantization.
            load_in_8bit: Use 8-bit quantization.
            device_map: Device mapping strategy.
            use_flash_attention: Use flash attention if available.
            trust_remote_code: Trust remote code.

        Returns:
            Tuple of (model, tokenizer).
        """
        # Check cache
        if model_name in self._model_cache:
            logger.info(f"Using cached model: {model_name}")
            return self._model_cache[model_name]

        # Evict old models if needed
        self._evict_if_needed()

        logger.info(f"Loading model: {model_name}")

        # Log memory before loading
        gpu_info = get_gpu_memory_info()
        if gpu_info.get("available"):
            for device in gpu_info.get("devices", []):
                logger.info(
                    f"GPU {device['id']} ({device['name']}): "
                    f"{device['allocated_gb']:.2f}GB / {device['total_memory_gb']:.2f}GB"
                )

        # Quantization config
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype_map[self.default_dtype],
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Determine dtype
        torch_dtype = (
            None if (load_in_4bit or load_in_8bit)
            else self.dtype_map[self.default_dtype]
        )

        # Attention implementation
        attn_impl = None
        if use_flash_attention and torch.cuda.is_available():
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir),
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir),
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {model_name} with {format_number(total_params)} parameters")

        # Cache
        self._model_cache[model_name] = (model, tokenizer)
        self._load_order.append(model_name)

        return model, tokenizer

    def _evict_if_needed(self) -> None:
        """Evict oldest model if cache is full."""
        while len(self._model_cache) >= self.max_cached_models:
            oldest = self._load_order.pop(0)
            self.unload_model(oldest)

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory.

        Args:
            model_name: Model to unload.
        """
        if model_name not in self._model_cache:
            return

        logger.info(f"Unloading model: {model_name}")

        model, tokenizer = self._model_cache[model_name]

        # Delete references
        del model
        del tokenizer
        del self._model_cache[model_name]

        if model_name in self._load_order:
            self._load_order.remove(model_name)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Unloaded {model_name}")

    def get_model(self, model_name: str) -> Optional[Tuple[PreTrainedModel, Any]]:
        """Get cached model if available.

        Args:
            model_name: Model name.

        Returns:
            Model and tokenizer tuple, or None.
        """
        return self._model_cache.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        """Check if model is loaded.

        Args:
            model_name: Model name.

        Returns:
            True if loaded.
        """
        return model_name in self._model_cache

    def list_cached_models(self) -> List[str]:
        """List currently cached models.

        Returns:
            List of model names.
        """
        return list(self._model_cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached models."""
        for model_name in list(self._model_cache.keys()):
            self.unload_model(model_name)


class AdapterLoader:
    """Load and manage LoRA adapters.

    Example:
        >>> loader = AdapterLoader(base_model)
        >>> model = loader.load_adapter("./adapters/reasoning")
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        adapters_dir: Optional[Path] = None
    ):
        """Initialize adapter loader.

        Args:
            base_model: Base model to attach adapters to.
            adapters_dir: Directory containing adapters.
        """
        from peft import PeftModel

        self.base_model = base_model
        self.adapters_dir = Path(adapters_dir) if adapters_dir else None
        self.PeftModel = PeftModel

        self.loaded_adapters: Dict[str, PreTrainedModel] = {}

    def load_adapter(
        self,
        adapter_path: str,
        adapter_name: Optional[str] = None,
        merge: bool = False
    ) -> PreTrainedModel:
        """Load a LoRA adapter.

        Args:
            adapter_path: Path to adapter weights.
            adapter_name: Name for the adapter.
            merge: Merge adapter into base model.

        Returns:
            Model with adapter loaded.
        """
        adapter_path = Path(adapter_path)
        adapter_name = adapter_name or adapter_path.name

        logger.info(f"Loading adapter: {adapter_name} from {adapter_path}")

        # Load with PEFT
        model = self.PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path)
        )

        if merge:
            model = model.merge_and_unload()
            logger.info(f"Merged adapter: {adapter_name}")

        self.loaded_adapters[adapter_name] = model

        return model

    def unload_adapter(self, adapter_name: str) -> None:
        """Unload an adapter.

        Args:
            adapter_name: Adapter name.
        """
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def list_available_adapters(self) -> List[str]:
        """List available adapters in adapters directory.

        Returns:
            List of adapter names.
        """
        if not self.adapters_dir or not self.adapters_dir.exists():
            return []

        adapters = []
        for item in self.adapters_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                adapters.append(item.name)

        return adapters


def get_optimal_config(
    model_name: str,
    available_memory_gb: Optional[float] = None
) -> Dict[str, Any]:
    """Get optimal loading configuration for a model.

    Args:
        model_name: Model name.
        available_memory_gb: Available GPU memory.

    Returns:
        Recommended configuration.
    """
    # Estimate model size (rough heuristics)
    model_sizes = {
        "7b": 14,    # ~14GB in fp16
        "8b": 16,
        "13b": 26,
        "70b": 140,
    }

    model_name_lower = model_name.lower()
    estimated_size = 14  # Default

    for size_key, mem_needed in model_sizes.items():
        if size_key in model_name_lower:
            estimated_size = mem_needed
            break

    # Get available memory
    if available_memory_gb is None:
        gpu_info = get_gpu_memory_info()
        if gpu_info.get("available") and gpu_info.get("devices"):
            available_memory_gb = sum(
                d["total_memory_gb"] - d["allocated_gb"]
                for d in gpu_info["devices"]
            )
        else:
            available_memory_gb = 16  # Default assumption

    # Determine configuration
    if available_memory_gb >= estimated_size:
        # Full precision possible
        return {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "device_map": "auto",
        }
    elif available_memory_gb >= estimated_size / 2:
        # 8-bit quantization
        return {
            "load_in_4bit": False,
            "load_in_8bit": True,
            "device_map": "auto",
        }
    else:
        # 4-bit quantization
        return {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "device_map": "auto",
        }
