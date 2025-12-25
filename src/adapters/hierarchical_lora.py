"""
Hierarchical LoRA adapter implementation.

Features:
- Multi-level adapter hierarchy
- Domain-specific sub-adapters
- Weight composition across levels
- Efficient parameter sharing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalLoRAAdapter(nn.Module):
    """Hierarchical LoRA adapter with multiple levels.

    Implements a hierarchical structure where adapters can be
    organized in a tree structure with shared and specialized
    components.

    Architecture:
        - Global adapter: Shared across all tasks
        - Domain adapters: Shared within a domain (e.g., coding, reasoning)
        - Task adapters: Specific to individual tasks

    Example:
        >>> adapter = HierarchicalLoRAAdapter(
        ...     base_model=model,
        ...     hierarchy_config={
        ...         'global': {'r': 8},
        ...         'domains': {
        ...             'coding': {'r': 16, 'tasks': ['python', 'javascript']},
        ...             'reasoning': {'r': 16, 'tasks': ['math', 'logic']}
        ...         }
        ...     }
        ... )
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        hierarchy_config: Dict[str, Any],
        target_modules: List[str] = None,
        composition_mode: str = "add"
    ):
        """Initialize hierarchical adapter.

        Args:
            base_model: Pre-trained base model.
            hierarchy_config: Configuration for hierarchy.
            target_modules: Modules to apply LoRA to.
            composition_mode: How to combine adapter outputs (add, weighted).
        """
        super().__init__()

        self.base_model = base_model
        self.hierarchy_config = hierarchy_config
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.composition_mode = composition_mode

        # Initialize adapters at each level
        self.global_adapter = None
        self.domain_adapters = nn.ModuleDict()
        self.task_adapters = nn.ModuleDict()

        # Composition weights
        self.level_weights = nn.Parameter(torch.ones(3) / 3)

        # Build hierarchy
        self._build_hierarchy()

    def _build_hierarchy(self) -> None:
        """Build adapter hierarchy from config."""
        # Global adapter
        if "global" in self.hierarchy_config:
            global_config = self.hierarchy_config["global"]
            self.global_adapter = self._create_lora_layer(
                global_config.get("r", 8),
                global_config.get("lora_alpha", 16),
                global_config.get("lora_dropout", 0.05)
            )
            logger.info("Created global adapter")

        # Domain adapters
        if "domains" in self.hierarchy_config:
            for domain_name, domain_config in self.hierarchy_config["domains"].items():
                self.domain_adapters[domain_name] = self._create_lora_layer(
                    domain_config.get("r", 16),
                    domain_config.get("lora_alpha", 32),
                    domain_config.get("lora_dropout", 0.05)
                )
                logger.info(f"Created domain adapter: {domain_name}")

                # Task adapters for this domain
                for task_name in domain_config.get("tasks", []):
                    full_name = f"{domain_name}_{task_name}"
                    task_config = domain_config.get("task_config", {})
                    self.task_adapters[full_name] = self._create_lora_layer(
                        task_config.get("r", 16),
                        task_config.get("lora_alpha", 32),
                        task_config.get("lora_dropout", 0.05)
                    )
                    logger.info(f"Created task adapter: {full_name}")

    def _create_lora_layer(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float
    ) -> nn.ModuleDict:
        """Create LoRA A and B matrices for each target module.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: Dropout rate.

        Returns:
            ModuleDict with LoRA parameters.
        """
        layers = nn.ModuleDict()

        # Get hidden size from model config
        hidden_size = self.base_model.config.hidden_size

        for module_name in self.target_modules:
            layers[module_name] = nn.ModuleDict({
                "lora_A": nn.Linear(hidden_size, r, bias=False),
                "lora_B": nn.Linear(r, hidden_size, bias=False),
                "scaling": nn.Parameter(torch.tensor(lora_alpha / r)),
                "dropout": nn.Dropout(lora_dropout)
            })

            # Initialize
            nn.init.kaiming_uniform_(layers[module_name]["lora_A"].weight)
            nn.init.zeros_(layers[module_name]["lora_B"].weight)

        return layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        module_name: str = "q_proj"
    ) -> torch.Tensor:
        """Forward pass with hierarchical composition.

        Args:
            hidden_states: Input hidden states.
            domain: Domain name.
            task: Task name.
            module_name: Target module name.

        Returns:
            LoRA output to add to original.
        """
        outputs = []
        weights = torch.softmax(self.level_weights, dim=0)

        # Global adapter contribution
        if self.global_adapter is not None and module_name in self.global_adapter:
            layer = self.global_adapter[module_name]
            output = layer["lora_B"](
                layer["dropout"](layer["lora_A"](hidden_states))
            ) * layer["scaling"]
            outputs.append(weights[0] * output)

        # Domain adapter contribution
        if domain and domain in self.domain_adapters:
            layer = self.domain_adapters[domain][module_name]
            output = layer["lora_B"](
                layer["dropout"](layer["lora_A"](hidden_states))
            ) * layer["scaling"]
            outputs.append(weights[1] * output)

        # Task adapter contribution
        full_task = f"{domain}_{task}" if domain and task else task
        if full_task and full_task in self.task_adapters:
            layer = self.task_adapters[full_task][module_name]
            output = layer["lora_B"](
                layer["dropout"](layer["lora_A"](hidden_states))
            ) * layer["scaling"]
            outputs.append(weights[2] * output)

        if not outputs:
            return torch.zeros_like(hidden_states)

        if self.composition_mode == "add":
            return sum(outputs)
        else:
            return sum(outputs)

    def get_active_parameters(
        self,
        domain: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[nn.Parameter]:
        """Get parameters active for given domain/task.

        Args:
            domain: Domain name.
            task: Task name.

        Returns:
            List of active parameters.
        """
        params = []

        # Global always active
        if self.global_adapter is not None:
            params.extend(self.global_adapter.parameters())

        # Domain if specified
        if domain and domain in self.domain_adapters:
            params.extend(self.domain_adapters[domain].parameters())

        # Task if specified
        full_task = f"{domain}_{task}" if domain and task else task
        if full_task and full_task in self.task_adapters:
            params.extend(self.task_adapters[full_task].parameters())

        # Level weights
        params.append(self.level_weights)

        return params

    def freeze_level(self, level: str) -> None:
        """Freeze parameters at a specific level.

        Args:
            level: Level to freeze (global, domain, task).
        """
        if level == "global" and self.global_adapter:
            for param in self.global_adapter.parameters():
                param.requires_grad = False
        elif level == "domain":
            for adapter in self.domain_adapters.values():
                for param in adapter.parameters():
                    param.requires_grad = False
        elif level == "task":
            for adapter in self.task_adapters.values():
                for param in adapter.parameters():
                    param.requires_grad = False

        logger.info(f"Froze {level} level parameters")

    def unfreeze_level(self, level: str) -> None:
        """Unfreeze parameters at a specific level.

        Args:
            level: Level to unfreeze.
        """
        if level == "global" and self.global_adapter:
            for param in self.global_adapter.parameters():
                param.requires_grad = True
        elif level == "domain":
            for adapter in self.domain_adapters.values():
                for param in adapter.parameters():
                    param.requires_grad = True
        elif level == "task":
            for adapter in self.task_adapters.values():
                for param in adapter.parameters():
                    param.requires_grad = True

        logger.info(f"Unfroze {level} level parameters")

    def save_hierarchy(self, save_dir: str) -> None:
        """Save hierarchical adapter.

        Args:
            save_dir: Directory to save to.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        state = {
            "global_adapter": self.global_adapter.state_dict() if self.global_adapter else None,
            "domain_adapters": {
                k: v.state_dict() for k, v in self.domain_adapters.items()
            },
            "task_adapters": {
                k: v.state_dict() for k, v in self.task_adapters.items()
            },
            "level_weights": self.level_weights.data,
            "hierarchy_config": self.hierarchy_config,
            "target_modules": self.target_modules,
            "composition_mode": self.composition_mode,
        }

        torch.save(state, f"{save_dir}/hierarchical_adapter.pt")
        logger.info(f"Saved hierarchical adapter to {save_dir}")

    @classmethod
    def load_hierarchy(
        cls,
        load_dir: str,
        base_model: PreTrainedModel
    ) -> "HierarchicalLoRAAdapter":
        """Load hierarchical adapter.

        Args:
            load_dir: Directory to load from.
            base_model: Base model to attach to.

        Returns:
            Loaded hierarchical adapter.
        """
        state = torch.load(f"{load_dir}/hierarchical_adapter.pt")

        adapter = cls(
            base_model=base_model,
            hierarchy_config=state["hierarchy_config"],
            target_modules=state["target_modules"],
            composition_mode=state["composition_mode"]
        )

        # Load weights
        if state["global_adapter"] and adapter.global_adapter:
            adapter.global_adapter.load_state_dict(state["global_adapter"])

        for name, state_dict in state["domain_adapters"].items():
            if name in adapter.domain_adapters:
                adapter.domain_adapters[name].load_state_dict(state_dict)

        for name, state_dict in state["task_adapters"].items():
            if name in adapter.task_adapters:
                adapter.task_adapters[name].load_state_dict(state_dict)

        adapter.level_weights.data = state["level_weights"]

        logger.info(f"Loaded hierarchical adapter from {load_dir}")

        return adapter


class AdapterComposer:
    """Compose multiple adapters dynamically.

    Allows runtime composition of adapters with learned
    or specified weights.
    """

    def __init__(
        self,
        adapters: Dict[str, nn.Module],
        composition_type: str = "weighted_sum"
    ):
        """Initialize composer.

        Args:
            adapters: Dictionary of named adapters.
            composition_type: How to compose (weighted_sum, concat, gated).
        """
        self.adapters = adapters
        self.composition_type = composition_type

        # Initialize weights
        self.weights = nn.Parameter(
            torch.ones(len(adapters)) / len(adapters)
        )

    def compose(
        self,
        hidden_states: torch.Tensor,
        adapter_names: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Compose adapter outputs.

        Args:
            hidden_states: Input hidden states.
            adapter_names: Adapters to use (None = all).
            custom_weights: Custom weights per adapter.

        Returns:
            Composed output.
        """
        if adapter_names is None:
            adapter_names = list(self.adapters.keys())

        outputs = []
        for name in adapter_names:
            if name in self.adapters:
                output = self.adapters[name](hidden_states)
                outputs.append(output)

        if not outputs:
            return hidden_states

        if self.composition_type == "weighted_sum":
            if custom_weights:
                weights = [custom_weights.get(n, 0.0) for n in adapter_names]
            else:
                weights = torch.softmax(self.weights[:len(outputs)], dim=0)
                weights = weights.tolist()

            return sum(w * o for w, o in zip(weights, outputs))

        elif self.composition_type == "mean":
            return sum(outputs) / len(outputs)

        else:
            return sum(outputs)
