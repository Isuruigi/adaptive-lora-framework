"""
Global settings management.

Features:
- Environment variable loading
- Settings singleton
- Configuration validation
- Hot reload support
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    """Global application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = Field(default="adaptive-lora-framework", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Server
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Model
    base_model_name: str = Field(default="meta-llama/Llama-3-8B", description="Base model name")
    model_cache_dir: Path = Field(default=Path("./models/base"), description="Model cache directory")
    adapters_dir: Path = Field(default=Path("./models/adapters"), description="Adapters directory")
    
    # Training
    output_dir: Path = Field(default=Path("./outputs"), description="Training output directory")
    logging_steps: int = Field(default=10, description="Logging interval")
    save_steps: int = Field(default=100, description="Checkpoint save interval")
    eval_steps: int = Field(default=100, description="Evaluation interval")
    
    # GPU
    cuda_visible_devices: Optional[str] = Field(default=None, description="CUDA visible devices")
    use_flash_attention: bool = Field(default=True, description="Use Flash Attention 2")
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    use_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    
    # Router
    router_model_name: str = Field(default="prajjwal1/bert-tiny", description="Router encoder model")
    router_hidden_dim: int = Field(default=256, description="Router hidden dimension")
    router_num_adapters: int = Field(default=4, description="Number of adapters")
    
    # Serving
    vllm_enabled: bool = Field(default=True, description="Enable vLLM inference")
    max_model_len: int = Field(default=4096, description="Maximum sequence length")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Authentication
    auth_enabled: bool = Field(default=True, description="Enable authentication")
    jwt_secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_minutes: int = Field(default=30, description="JWT token expiry")
    
    # External Services
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    wandb_api_key: Optional[str] = Field(default=None, description="Weights & Biases API key")
    wandb_project: str = Field(default="adaptive-lora", description="W&B project name")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_enabled: bool = Field(default=False, description="Enable Redis backend")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json, text)")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # Storage
    storage_backend: str = Field(default="local", description="Storage backend (local, s3)")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default=None, description="S3 region")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v
    
    @field_validator('gpu_memory_utilization')
    @classmethod
    def validate_gpu_memory(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_device_map(self) -> str:
        """Get device map for model loading."""
        if self.cuda_visible_devices:
            return "auto"
        return "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding secrets)."""
        result = {}
        for key in self.model_fields:
            value = getattr(self, key)
            
            # Skip secrets
            if 'key' in key.lower() or 'secret' in key.lower():
                result[key] = "***" if value else None
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
                
        return result


@lru_cache()
def get_settings() -> Settings:
    """Get settings singleton.
    
    Returns:
        Settings instance.
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment.
    
    Returns:
        New Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()


def configure_from_dict(config: Dict[str, Any]) -> Settings:
    """Create settings from dictionary.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Settings instance.
    """
    return Settings(**config)


def setup_environment() -> None:
    """Setup environment based on settings."""
    settings = get_settings()
    
    # Set CUDA devices
    if settings.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
        
    # Set log level
    import logging
    logging.getLogger().setLevel(settings.log_level)
    
    # Create directories
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Environment setup complete for {settings.environment}")


# Convenience accessors
def get_model_settings() -> Dict[str, Any]:
    """Get model-related settings."""
    s = get_settings()
    return {
        "model_name": s.base_model_name,
        "cache_dir": str(s.model_cache_dir),
        "use_4bit": s.use_4bit,
        "use_8bit": s.use_8bit,
        "use_flash_attention": s.use_flash_attention,
    }


def get_api_settings() -> Dict[str, Any]:
    """Get API-related settings."""
    s = get_settings()
    return {
        "host": s.api_host,
        "port": s.api_port,
        "workers": s.api_workers,
        "reload": s.api_reload,
        "debug": s.debug,
    }


def get_training_settings() -> Dict[str, Any]:
    """Get training-related settings."""
    s = get_settings()
    return {
        "output_dir": str(s.output_dir),
        "logging_steps": s.logging_steps,
        "save_steps": s.save_steps,
        "eval_steps": s.eval_steps,
        "wandb_project": s.wandb_project,
    }
