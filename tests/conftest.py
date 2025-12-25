"""
Pytest Configuration and Fixtures

Shared fixtures for all tests in the Adaptive LoRA Framework.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, patch

import pytest
import torch


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Get configs directory."""
    return project_root / "configs"


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device() -> str:
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cpu_device() -> str:
    """Force CPU device."""
    return "cpu"


# =============================================================================
# Mock Model Fixtures
# =============================================================================

@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.model_max_length = 2048
    
    def encode_side_effect(text, *args, **kwargs):
        return {"input_ids": [1] * min(len(text.split()), 512)}
    
    tokenizer.encode.side_effect = encode_side_effect
    tokenizer.__call__ = MagicMock(return_value={
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    })
    
    return tokenizer


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock language model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 4096
    model.config.num_hidden_layers = 32
    model.config.pad_token_id = 0
    
    # Mock forward pass
    model.return_value = MagicMock(
        logits=torch.randn(1, 10, 32000),
        loss=torch.tensor(0.5)
    )
    
    # Mock generate
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    
    return model


@pytest.fixture
def mock_router() -> MagicMock:
    """Create a mock router model."""
    router = MagicMock()
    
    router.return_value = MagicMock(
        adapter_logits=torch.randn(1, 4),
        adapter_weights=torch.softmax(torch.randn(1, 4), dim=-1),
        selected_adapter="reasoning",
        confidence=0.85
    )
    
    return router


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_instruction_data() -> list:
    """Sample instruction-following data."""
    return [
        {
            "instruction": "Explain the concept of machine learning.",
            "input": "",
            "output": "Machine learning is a subset of AI..."
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "input": "n = 5",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        },
        {
            "instruction": "Summarize the following text.",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "A fox jumps over a dog."
        },
    ]


@pytest.fixture
def sample_router_data() -> list:
    """Sample router training data."""
    return [
        {"instruction": "Solve this math problem", "adapter": "reasoning"},
        {"instruction": "Write a function that", "adapter": "code"},
        {"instruction": "Analyze the following data", "adapter": "analysis"},
        {"instruction": "Tell me about", "adapter": "base"},
    ]


@pytest.fixture
def sample_jsonl_file(temp_dir: Path, sample_instruction_data: list) -> Path:
    """Create a sample JSONL file."""
    file_path = temp_dir / "sample.jsonl"
    
    with open(file_path, "w") as f:
        for item in sample_instruction_data:
            f.write(json.dumps(item) + "\n")
    
    return file_path


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def lora_config() -> Dict[str, Any]:
    """Sample LoRA configuration."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }


@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Sample training configuration."""
    return {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100,
        "seed": 42
    }


@pytest.fixture
def router_config() -> Dict[str, Any]:
    """Sample router configuration."""
    return {
        "encoder_name": "prajjwal1/bert-tiny",
        "hidden_dim": 128,
        "num_adapters": 4,
        "adapter_names": ["reasoning", "code", "analysis", "base"],
        "dropout": 0.1
    }


@pytest.fixture
def serving_config() -> Dict[str, Any]:
    """Sample serving configuration."""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "timeout": 60,
        "max_batch_size": 8,
        "cache_enabled": False
    }


# =============================================================================
# API/HTTP Fixtures
# =============================================================================

@pytest.fixture
def mock_request() -> Dict[str, Any]:
    """Sample API request."""
    return {
        "prompt": "Explain quantum computing",
        "max_tokens": 256,
        "temperature": 0.7,
        "adapter": None
    }


@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Sample API response."""
    return {
        "text": "Quantum computing is a type of computation...",
        "adapter_used": "reasoning",
        "tokens_generated": 128,
        "latency_ms": 450.5
    }


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Set mock environment variables for testing."""
    original_env = os.environ.copy()
    
    os.environ.update({
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "WANDB_API_KEY": "test-wandb-key",
        "ENVIRONMENT": "test"
    })
    
    yield
    
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Skip Markers
# =============================================================================

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU"
)

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OpenAI API key"
)

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires Anthropic API key"
)

slow_test = pytest.mark.slow
integration_test = pytest.mark.integration


# =============================================================================
# Utility Functions
# =============================================================================

def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5):
    """Assert two tensors are approximately equal."""
    assert torch.allclose(a, b, rtol=rtol), f"Tensors not equal: {a} vs {b}"


def create_dummy_adapter(path: Path, config: Dict[str, Any]) -> None:
    """Create a dummy adapter for testing."""
    path.mkdir(parents=True, exist_ok=True)
    
    # Create config
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f)
    
    # Create dummy weights
    dummy_state = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 4096),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4096, 16),
    }
    torch.save(dummy_state, path / "adapter_model.bin")
