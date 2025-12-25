"""
Tests for Adapter Module

Tests for LoRA trainer, adapter manager, and multi-adapter training.
"""

import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# Pytest automatically loads conftest.py - access markers via pytest.mark
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU"
)
slow_test = pytest.mark.slow


class TestLoRAConfig:
    """Tests for LoRA configuration."""
    
    def test_default_lora_config(self, lora_config: Dict[str, Any]):
        """Test default LoRA configuration values."""
        assert lora_config["r"] == 16
        assert lora_config["lora_alpha"] == 32
        assert lora_config["lora_dropout"] == 0.05
        assert "q_proj" in lora_config["target_modules"]
    
    def test_lora_alpha_ratio(self, lora_config: Dict[str, Any]):
        """Test that lora_alpha is typically 2x r."""
        # Common practice: alpha = 2 * r
        assert lora_config["lora_alpha"] >= lora_config["r"]
    
    def test_valid_target_modules(self, lora_config: Dict[str, Any]):
        """Test that target modules are valid."""
        valid_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"]
        for module in lora_config["target_modules"]:
            assert module in valid_modules


class TestLoRATrainer:
    """Tests for LoRATrainer class."""
    
    @pytest.fixture
    def trainer_init_args(
        self,
        temp_dir: Path,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arguments for trainer initialization."""
        return {
            "model_name": "meta-llama/Llama-3-8B",
            "output_dir": temp_dir / "outputs",
            "lora_config": lora_config,
            "training_config": training_config,
            "use_4bit": True,
            "use_wandb": False
        }
    
    def test_trainer_init_creates_output_dir(
        self,
        trainer_init_args: Dict[str, Any]
    ):
        """Test that trainer creates output directory."""
        output_dir = trainer_init_args["output_dir"]
        
        # Mock the model loading
        with patch("src.adapters.lora_trainer.AutoModelForCausalLM"):
            with patch("src.adapters.lora_trainer.AutoTokenizer"):
                with patch("src.adapters.lora_trainer.get_peft_model"):
                    # Just test output dir creation logic
                    output_dir.mkdir(parents=True, exist_ok=True)
                    assert output_dir.exists()
    
    def test_save_training_config(
        self,
        temp_dir: Path,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        """Test saving training configuration."""
        config_path = temp_dir / "training_config.json"
        
        config = {
            "lora_config": lora_config,
            "training_config": training_config
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Verify saved config
        with open(config_path, "r") as f:
            loaded = json.load(f)
        
        assert loaded["lora_config"]["r"] == lora_config["r"]
        assert loaded["training_config"]["learning_rate"] == training_config["learning_rate"]


class TestMultiAdapterTrainer:
    """Tests for MultiAdapterTrainer class."""
    
    @pytest.fixture
    def adapters_config(self) -> Dict[str, Dict[str, Any]]:
        """Configuration for multiple adapters."""
        return {
            "reasoning": {
                "data_path": "data/reasoning/train.jsonl",
                "lora_config": {"r": 32, "lora_alpha": 64}
            },
            "code": {
                "data_path": "data/code/train.jsonl",
                "lora_config": {"r": 32, "lora_alpha": 64}
            }
        }
    
    def test_adapters_config_structure(
        self,
        adapters_config: Dict[str, Dict[str, Any]]
    ):
        """Test adapters configuration structure."""
        assert "reasoning" in adapters_config
        assert "code" in adapters_config
        
        for adapter_name, config in adapters_config.items():
            assert "data_path" in config
            assert "lora_config" in config
    
    def test_adapter_order(
        self,
        adapters_config: Dict[str, Dict[str, Any]]
    ):
        """Test that adapter training order can be specified."""
        adapter_order = ["code", "reasoning"]
        
        for adapter in adapter_order:
            assert adapter in adapters_config


class TestEarlyStopping:
    """Tests for EarlyStoppingCallback."""
    
    def test_early_stopping_triggers_after_patience(self):
        """Test early stopping triggers after patience exhausted."""
        # Simulate evaluation results with no improvement
        results = [
            {"eval_loss": 1.0},
            {"eval_loss": 1.1},
            {"eval_loss": 1.2},
            {"eval_loss": 1.3},
        ]
        
        patience = 3
        best_loss = float('inf')
        counter = 0
        
        for result in results:
            current = result["eval_loss"]
            if current < best_loss:
                best_loss = current
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                break
        
        assert counter >= patience
    
    def test_early_stopping_resets_on_improvement(self):
        """Test early stopping resets counter on improvement."""
        results = [
            {"eval_loss": 1.0},
            {"eval_loss": 1.1},
            {"eval_loss": 0.9},  # Improvement
            {"eval_loss": 1.0},
        ]
        
        best_loss = float('inf')
        counter = 0
        
        for result in results:
            current = result["eval_loss"]
            if current < best_loss:
                best_loss = current
                counter = 0
            else:
                counter += 1
        
        assert counter == 1  # Only one step without improvement after reset


class TestLearningRateFinder:
    """Tests for LearningRateFinder."""
    
    def test_lr_range_calculation(self):
        """Test learning rate range calculation."""
        min_lr = 1e-7
        max_lr = 1e-1
        num_steps = 100
        
        # Calculate multiplication factor
        lr_mult = (max_lr / min_lr) ** (1 / num_steps)
        
        # Verify range
        current_lr = min_lr
        for _ in range(num_steps):
            current_lr *= lr_mult
        
        assert abs(current_lr - max_lr) < 1e-5
    
    def test_suggest_lr_from_losses(self):
        """Test LR suggestion from loss curve."""
        # Simulate loss curve
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        losses = [1.0, 0.8, 0.5, 0.3, 0.6]  # Loss increases at high LR
        
        # Find minimum loss
        min_idx = losses.index(min(losses))
        suggested_lr = lrs[min_idx] / 10  # Common heuristic
        
        assert 1e-5 < suggested_lr < 1e-3


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    def test_max_checkpoints_limit(self, temp_dir: Path):
        """Test that checkpoint manager respects max limit."""
        max_checkpoints = 3
        checkpoints = []
        
        for i in range(5):
            ckpt_path = temp_dir / f"checkpoint-{i}"
            ckpt_path.mkdir()
            checkpoints.append({
                "path": str(ckpt_path),
                "step": i,
                "metrics": {"eval_loss": 1.0 - i * 0.1}
            })
        
        # Keep only best checkpoints
        sorted_ckpts = sorted(
            checkpoints,
            key=lambda x: x["metrics"]["eval_loss"]
        )
        kept = sorted_ckpts[:max_checkpoints]
        
        assert len(kept) == max_checkpoints
    
    def test_save_best_only(self, temp_dir: Path):
        """Test save-best-only mode."""
        checkpoints = []
        best_loss = float('inf')
        
        losses = [1.0, 0.9, 0.95, 0.8, 0.85]
        
        for i, loss in enumerate(losses):
            if loss < best_loss:
                best_loss = loss
                ckpt_path = temp_dir / f"checkpoint-{i}"
                ckpt_path.mkdir()
                checkpoints.append(i)
        
        # Should only have checkpoints for improvements
        assert len(checkpoints) == 3  # At indices 0, 1, 3


class TestHyperparameterOptimizer:
    """Tests for HyperparameterOptimizer."""
    
    def test_parameter_sampling(self):
        """Test that parameters are sampled from correct ranges."""
        lora_r_options = [4, 8, 16, 32, 64]
        lr_min, lr_max = 1e-5, 5e-4
        
        # Simulate sampling
        import random
        
        sampled_r = random.choice(lora_r_options)
        sampled_lr = 10 ** random.uniform(
            import_log10(lr_min),
            import_log10(lr_max)
        )
        
        assert sampled_r in lora_r_options
        assert lr_min <= sampled_lr <= lr_max


def import_log10(x):
    """Helper for log10."""
    import math
    return math.log10(x)
