"""
Comprehensive configuration management system using Pydantic for validation.

Supports YAML and environment variable configuration with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for base model.

    Attributes:
        model_name: HuggingFace model ID or local path.
        torch_dtype: PyTorch data type for model weights.
        load_in_4bit: Whether to use 4-bit quantization (QLoRA).
        load_in_8bit: Whether to use 8-bit quantization.
        device_map: Device mapping strategy for model parallelism.
        trust_remote_code: Whether to trust remote code in model files.
        max_memory: Maximum memory per device (e.g., {"0": "24GB"}).
        attn_implementation: Attention implementation (flash_attention_2, sdpa, eager).
    """

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    model_name: str = Field(
        default="meta-llama/Llama-3-8B",
        description="HuggingFace model ID or local path"
    )
    torch_dtype: str = Field(
        default="bfloat16",
        description="Model dtype (float16, bfloat16, float32)"
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Use 4-bit quantization for QLoRA"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code in model files"
    )
    max_memory: Optional[Dict[str, str]] = Field(
        default=None,
        description="Maximum memory per device"
    )
    attn_implementation: Optional[str] = Field(
        default="flash_attention_2",
        description="Attention implementation"
    )

    @field_validator("torch_dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if v not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        return v


class LoRAConfig(BaseSettings):
    """Configuration for LoRA adapters.

    Attributes:
        r: LoRA rank (dimension of low-rank matrices).
        lora_alpha: LoRA alpha (scaling factor).
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        bias: Bias training strategy (none, all, lora_only).
        task_type: Task type for PEFT (CAUSAL_LM, SEQ_2_SEQ_LM).
        modules_to_save: Additional modules to train fully.
    """

    model_config = SettingsConfigDict(env_prefix="LORA_")

    r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA rank"
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA alpha (scaling factor)"
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Dropout probability"
    )
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA"
    )
    bias: str = Field(
        default="none",
        description="Bias training strategy"
    )
    task_type: str = Field(
        default="CAUSAL_LM",
        description="Task type"
    )
    modules_to_save: Optional[List[str]] = Field(
        default=None,
        description="Additional modules to train fully"
    )

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        valid_bias = {"none", "all", "lora_only"}
        if v not in valid_bias:
            raise ValueError(f"bias must be one of {valid_bias}")
        return v

    @model_validator(mode="after")
    def validate_alpha_vs_rank(self) -> "LoRAConfig":
        if self.lora_alpha < self.r:
            raise ValueError(
                f"lora_alpha ({self.lora_alpha}) should be >= rank ({self.r}) "
                "for stable training"
            )
        return self


class TrainingConfig(BaseSettings):
    """Training hyperparameters.

    Attributes:
        output_dir: Directory for saving outputs.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Initial learning rate.
        lr_scheduler_type: Learning rate scheduler type.
        warmup_ratio: Warmup ratio for learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        weight_decay: Weight decay coefficient.
        logging_steps: Steps between logging.
        save_steps: Steps between saving checkpoints.
        eval_steps: Steps between evaluations.
        save_total_limit: Maximum number of checkpoints to keep.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        gradient_checkpointing: Use gradient checkpointing to save memory.
        max_seq_length: Maximum sequence length.
        dataloader_num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.
    """

    model_config = SettingsConfigDict(env_prefix="TRAINING_")

    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Output directory"
    )
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs"
    )
    per_device_train_batch_size: int = Field(
        default=4,
        ge=1,
        description="Batch size per device"
    )
    per_device_eval_batch_size: int = Field(
        default=4,
        ge=1,
        description="Eval batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
        description="Gradient accumulation steps"
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        description="Learning rate"
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        description="LR scheduler type"
    )
    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Warmup ratio"
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        description="Maximum gradient norm"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay"
    )
    logging_steps: int = Field(
        default=10,
        ge=1,
        description="Logging interval"
    )
    save_steps: int = Field(
        default=100,
        ge=1,
        description="Checkpoint save interval"
    )
    eval_steps: int = Field(
        default=100,
        ge=1,
        description="Evaluation interval"
    )
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Max checkpoints to keep"
    )
    fp16: bool = Field(
        default=False,
        description="Use FP16"
    )
    bf16: bool = Field(
        default=True,
        description="Use BF16"
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Use gradient checkpointing"
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=32768,
        description="Maximum sequence length"
    )
    dataloader_num_workers: int = Field(
        default=4,
        ge=0,
        description="Dataloader workers"
    )
    seed: int = Field(
        default=42,
        description="Random seed"
    )

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class RouterConfig(BaseSettings):
    """Configuration for router network.

    Attributes:
        encoder_name: Pre-trained encoder model name.
        num_adapters: Number of available adapters.
        num_capabilities: Number of capability dimensions.
        hidden_dim: Hidden dimension size.
        dropout: Dropout probability.
        use_gumbel_softmax: Use Gumbel-Softmax for differentiable routing.
        temperature: Temperature for Gumbel-Softmax.
        min_confidence: Minimum confidence threshold for routing.
        fallback_strategy: Strategy when confidence is low.
    """

    model_config = SettingsConfigDict(env_prefix="ROUTER_")

    encoder_name: str = Field(
        default="prajjwal1/bert-tiny",
        description="Pre-trained encoder model"
    )
    num_adapters: int = Field(
        default=4,
        ge=1,
        description="Number of adapters"
    )
    num_capabilities: int = Field(
        default=6,
        ge=1,
        description="Number of capabilities"
    )
    hidden_dim: int = Field(
        default=256,
        ge=64,
        description="Hidden dimension"
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability"
    )
    use_gumbel_softmax: bool = Field(
        default=True,
        description="Use Gumbel-Softmax"
    )
    temperature: float = Field(
        default=1.0,
        gt=0,
        description="Temperature for routing"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum routing confidence"
    )
    fallback_strategy: str = Field(
        default="uniform",
        description="Fallback strategy (uniform, top_k)"
    )


class ServingConfig(BaseSettings):
    """Configuration for model serving.

    Attributes:
        host: API host address.
        port: API port.
        workers: Number of API workers.
        max_batch_size: Maximum batch size for inference.
        max_concurrent_requests: Maximum concurrent requests.
        timeout: Request timeout in seconds.
        use_vllm: Use vLLM for serving.
        tensor_parallel_size: Tensor parallelism size.
        gpu_memory_utilization: Target GPU memory utilization.
    """

    model_config = SettingsConfigDict(env_prefix="SERVING_")

    host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    port: int = Field(
        default=8000,
        description="API port"
    )
    workers: int = Field(
        default=4,
        ge=1,
        description="API workers"
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Max batch size"
    )
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        description="Max concurrent requests"
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout (seconds)"
    )
    use_vllm: bool = Field(
        default=True,
        description="Use vLLM for serving"
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Tensor parallel size"
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description="GPU memory utilization"
    )


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and logging.

    Attributes:
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team name.
        prometheus_enabled: Enable Prometheus metrics.
        prometheus_port: Prometheus metrics port.
        log_level: Logging level.
        log_dir: Directory for log files.
        structured_logging: Use JSON structured logging.
    """

    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    use_wandb: bool = Field(
        default=True,
        description="Enable W&B"
    )
    wandb_project: str = Field(
        default="adaptive-lora-framework",
        description="W&B project name"
    )
    wandb_entity: Optional[str] = Field(
        default=None,
        description="W&B entity"
    )
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus"
    )
    prometheus_port: int = Field(
        default=9090,
        description="Prometheus port"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Log directory"
    )
    structured_logging: bool = Field(
        default=True,
        description="Use JSON logging"
    )


class SystemConfig:
    """Main configuration aggregator.

    Loads and validates all configuration sections from YAML files
    and environment variables.

    Example:
        >>> config = SystemConfig()
        >>> config = SystemConfig(config_path=Path("configs/default.yaml"))
        >>> print(config.model.model_name)
        'meta-llama/Llama-3-8B'
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        override_dict: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration.

        Args:
            config_path: Optional path to YAML configuration file.
            override_dict: Optional dictionary to override configuration values.
        """
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.router = RouterConfig()
        self.serving = ServingConfig()
        self.monitoring = MonitoringConfig()

        if config_path and config_path.exists():
            self.load_from_yaml(config_path)

        if override_dict:
            self._apply_overrides(override_dict)

    def load_from_yaml(self, path: Path) -> None:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.
        """
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            return

        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        if "lora" in config_dict:
            self.lora = LoRAConfig(**config_dict["lora"])
        if "training" in config_dict:
            self.training = TrainingConfig(**config_dict["training"])
        if "router" in config_dict:
            self.router = RouterConfig(**config_dict["router"])
        if "serving" in config_dict:
            self.serving = ServingConfig(**config_dict["serving"])
        if "monitoring" in config_dict:
            self.monitoring = MonitoringConfig(**config_dict["monitoring"])

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply override dictionary to configuration.

        Args:
            overrides: Dictionary with section.key = value format.
        """
        for key, value in overrides.items():
            if "." in key:
                section, attr = key.split(".", 1)
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    if hasattr(section_config, attr):
                        setattr(section_config, attr, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of all configuration sections.
        """
        return {
            "model": self.model.model_dump(),
            "lora": self.lora.model_dump(),
            "training": self.training.model_dump(),
            "router": self.router.model_dump(),
            "serving": self.serving.model_dump(),
            "monitoring": self.monitoring.model_dump(),
        }

    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        # Convert Path objects to strings for YAML serialization
        for section in config_dict.values():
            for key, value in section.items():
                if isinstance(value, Path):
                    section[key] = str(value)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"SystemConfig(model={self.model.model_name}, lora_r={self.lora.r})"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides: Any
) -> SystemConfig:
    """Convenience function to load configuration.

    Args:
        config_path: Optional path to YAML configuration file.
        **overrides: Keyword arguments to override configuration values.

    Returns:
        Loaded and validated SystemConfig instance.
    """
    if config_path:
        config_path = Path(config_path)

    override_dict = {}
    for key, value in overrides.items():
        override_dict[key] = value

    return SystemConfig(config_path=config_path, override_dict=override_dict)
