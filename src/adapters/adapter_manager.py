"""
Adapter lifecycle management utilities.

Features:
- Adapter loading and switching
- Adapter merging and combining
- Adapter registry and metadata
- Version management
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata for a trained adapter.

    Attributes:
        name: Adapter name.
        base_model: Base model used.
        task_type: Task type (reasoning, code, etc.).
        created_at: Creation timestamp.
        lora_config: LoRA configuration.
        training_config: Training configuration.
        metrics: Evaluation metrics.
        version: Adapter version.
        description: Human-readable description.
    """

    name: str
    base_model: str
    task_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    lora_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    version: str = "1.0.0"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "lora_config": self.lora_config,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "version": self.version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterMetadata":
        """Create from dictionary."""
        return cls(**data)


class AdapterManager:
    """Manage adapter lifecycle, loading, and switching.

    Provides centralized management for multiple LoRA adapters
    including loading, switching, and combining.

    Example:
        >>> manager = AdapterManager(base_model, adapters_dir=Path("./adapters"))
        >>> manager.register_adapter("reasoning", Path("./adapters/reasoning"))
        >>> manager.load_adapter("reasoning")
        >>> manager.switch_adapter("code")
    """

    def __init__(
        self,
        base_model: Optional[PreTrainedModel] = None,
        base_model_name: Optional[str] = None,
        adapters_dir: Optional[Path] = None,
        device_map: str = "auto"
    ):
        """Initialize adapter manager.

        Args:
            base_model: Pre-loaded base model.
            base_model_name: Name of base model to load.
            adapters_dir: Directory containing adapters.
            device_map: Device mapping strategy.
        """
        self.adapters_dir = Path(adapters_dir) if adapters_dir else Path("./adapters")
        self.device_map = device_map
        self.base_model_name = base_model_name

        # State
        self.base_model = base_model
        self.current_adapter: Optional[str] = None
        self.loaded_adapters: Dict[str, Dict[str, Any]] = {}
        self.adapter_registry: Dict[str, AdapterMetadata] = {}

        # Initialize
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        """Load adapter registry from disk."""
        registry_path = self.adapters_dir / "registry.json"

        if registry_path.exists():
            with open(registry_path, "r") as f:
                data = json.load(f)
                for name, metadata in data.items():
                    self.adapter_registry[name] = AdapterMetadata.from_dict(metadata)

            logger.info(f"Loaded {len(self.adapter_registry)} adapters from registry")

    def _save_registry(self) -> None:
        """Save adapter registry to disk."""
        registry_path = self.adapters_dir / "registry.json"

        data = {
            name: metadata.to_dict()
            for name, metadata in self.adapter_registry.items()
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_adapter(
        self,
        name: str,
        adapter_path: Path,
        metadata: Optional[AdapterMetadata] = None,
        copy_files: bool = True
    ) -> None:
        """Register an adapter in the manager.

        Args:
            name: Unique adapter name.
            adapter_path: Path to adapter weights.
            metadata: Optional adapter metadata.
            copy_files: Copy adapter files to managed directory.
        """
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # Copy to managed directory if requested
        if copy_files:
            target_dir = self.adapters_dir / name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(adapter_path, target_dir)
            adapter_path = target_dir

        # Load or create metadata
        if metadata is None:
            # Try to load existing metadata
            config_path = adapter_path / "training_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                metadata = AdapterMetadata(
                    name=name,
                    base_model=config.get("model_name", "unknown"),
                    task_type=name,
                    lora_config=config.get("lora_config", {}),
                    training_config=config.get("training_config", {}),
                )
            else:
                metadata = AdapterMetadata(
                    name=name,
                    base_model="unknown",
                    task_type=name,
                )

        self.adapter_registry[name] = metadata
        self._save_registry()

        logger.info(f"Registered adapter: {name}")

    def load_adapter(
        self,
        name: str,
        merge: bool = False
    ) -> PreTrainedModel:
        """Load an adapter onto the base model.

        Args:
            name: Adapter name.
            merge: Merge adapter weights into base model.

        Returns:
            Model with adapter loaded.

        Raises:
            ValueError: If adapter not found in registry.
        """
        if name not in self.adapter_registry:
            raise ValueError(f"Adapter not registered: {name}")

        adapter_path = self.adapters_dir / name

        if self.base_model is None:
            raise ValueError("Base model not loaded")

        # Load adapter
        model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path)
        )

        if merge:
            model = model.merge_and_unload()

        self.current_adapter = name
        self.loaded_adapters[name] = {"model": model, "merged": merge}

        logger.info(f"Loaded adapter: {name} (merged={merge})")

        return model

    def switch_adapter(self, name: str) -> PreTrainedModel:
        """Switch to a different adapter.

        Args:
            name: Adapter name to switch to.

        Returns:
            Model with new adapter.
        """
        if name not in self.adapter_registry:
            raise ValueError(f"Adapter not registered: {name}")

        # If already loaded, return cached
        if name in self.loaded_adapters and not self.loaded_adapters[name]["merged"]:
            self.current_adapter = name
            return self.loaded_adapters[name]["model"]

        # Load new adapter
        return self.load_adapter(name)

    def unload_adapter(self, name: Optional[str] = None) -> None:
        """Unload an adapter.

        Args:
            name: Adapter to unload. If None, unload current.
        """
        name = name or self.current_adapter

        if name and name in self.loaded_adapters:
            del self.loaded_adapters[name]
            if self.current_adapter == name:
                self.current_adapter = None

            logger.info(f"Unloaded adapter: {name}")

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters.

        Returns:
            List of adapter information.
        """
        return [
            {
                "name": name,
                "loaded": name in self.loaded_adapters,
                "current": name == self.current_adapter,
                **metadata.to_dict()
            }
            for name, metadata in self.adapter_registry.items()
        ]

    def get_adapter_info(self, name: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata.

        Args:
            name: Adapter name.

        Returns:
            Adapter metadata or None.
        """
        return self.adapter_registry.get(name)

    def delete_adapter(self, name: str) -> None:
        """Delete an adapter.

        Args:
            name: Adapter name to delete.
        """
        if name in self.loaded_adapters:
            self.unload_adapter(name)

        if name in self.adapter_registry:
            del self.adapter_registry[name]
            self._save_registry()

        adapter_dir = self.adapters_dir / name
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        logger.info(f"Deleted adapter: {name}")

    def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        output_name: str = "merged"
    ) -> PreTrainedModel:
        """Merge multiple adapters with optional weights.

        Args:
            adapter_names: List of adapter names to merge.
            weights: Optional weights for each adapter.
            output_name: Name for merged adapter.

        Returns:
            Model with merged adapters.
        """
        if weights is None:
            weights = [1.0 / len(adapter_names)] * len(adapter_names)

        if len(weights) != len(adapter_names):
            raise ValueError("Number of weights must match number of adapters")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        logger.info(f"Merging adapters: {adapter_names} with weights {weights}")

        # Load all adapter state dicts
        adapter_states = []
        for name in adapter_names:
            adapter_path = self.adapters_dir / name
            model = PeftModel.from_pretrained(self.base_model, str(adapter_path))
            state_dict = get_peft_model_state_dict(model)
            adapter_states.append(state_dict)

        # Merge state dicts
        merged_state = {}
        for key in adapter_states[0].keys():
            merged_state[key] = sum(
                w * state[key] for w, state in zip(weights, adapter_states)
            )

        # Apply merged state to model
        model = PeftModel.from_pretrained(
            self.base_model,
            str(self.adapters_dir / adapter_names[0])
        )
        set_peft_model_state_dict(model, merged_state)

        # Save merged adapter
        output_path = self.adapters_dir / output_name
        model.save_pretrained(output_path)

        # Register merged adapter
        self.register_adapter(
            output_name,
            output_path,
            metadata=AdapterMetadata(
                name=output_name,
                base_model=self.base_model_name or "unknown",
                task_type="merged",
                description=f"Merged from: {adapter_names} with weights {weights}"
            ),
            copy_files=False
        )

        logger.info(f"Created merged adapter: {output_name}")

        return model

    def export_adapter(
        self,
        name: str,
        output_path: Path,
        include_base_model: bool = False
    ) -> None:
        """Export adapter for deployment.

        Args:
            name: Adapter name.
            output_path: Output directory.
            include_base_model: Include base model weights.
        """
        if name not in self.adapter_registry:
            raise ValueError(f"Adapter not registered: {name}")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        adapter_path = self.adapters_dir / name

        # Copy adapter files
        for item in adapter_path.iterdir():
            if item.is_file():
                shutil.copy(item, output_path / item.name)

        # Add metadata
        metadata = self.adapter_registry[name]
        with open(output_path / "adapter_metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Optionally merge with base model
        if include_base_model:
            model = self.load_adapter(name, merge=True)
            model.save_pretrained(output_path / "merged_model")

        logger.info(f"Exported adapter to {output_path}")
