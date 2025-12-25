"""
Tests for router module.
"""

import pytest
import torch

from src.router.router_model import (
    AdapterRouter,
    RouterOutput,
    HierarchicalRouter,
    DynamicRouter,
    RouterEnsemble,
    create_router,
)


class TestRouterOutput:
    """Tests for RouterOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = RouterOutput(
            adapter_weights=torch.tensor([[0.7, 0.2, 0.1]]),
            complexity_score=torch.tensor([[0.1, 0.7, 0.2]]),
            capabilities=torch.tensor([[0.5, 0.8, 0.3, 0.9, 0.2, 0.6]]),
            confidence=torch.tensor([0.85])
        )
        
        result = output.to_dict()
        
        assert "adapter_weights" in result
        assert "complexity_score" in result
        assert "capabilities" in result
        assert "confidence" in result

    def test_get_top_adapters(self):
        """Test getting top adapters above threshold."""
        output = RouterOutput(
            adapter_weights=torch.tensor([[0.7, 0.2, 0.05, 0.05]]),
            complexity_score=torch.tensor([[0.0, 1.0, 0.0]]),
            capabilities=torch.tensor([[0.5] * 6]),
            confidence=torch.tensor([0.9])
        )
        
        top_adapters = output.get_top_adapters(threshold=0.1)
        
        assert len(top_adapters) == 2
        assert top_adapters[0][0] == 0  # Index of highest weight
        assert top_adapters[0][1] == pytest.approx(0.7, rel=1e-5)


class TestAdapterRouter:
    """Tests for AdapterRouter model."""

    @pytest.fixture
    def router(self):
        """Create router for testing."""
        return AdapterRouter(
            encoder_name="prajjwal1/bert-tiny",
            num_adapters=4,
            num_capabilities=6,
            hidden_dim=64,  # Smaller for testing
            dropout=0.1
        )

    @pytest.fixture
    def sample_input(self, router):
        """Create sample input tensors."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        encoding = tokenizer(
            "What is machine learning?",
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        )
        
        return encoding["input_ids"], encoding["attention_mask"]

    def test_forward_pass(self, router, sample_input):
        """Test forward pass of router."""
        input_ids, attention_mask = sample_input
        
        output = router(input_ids, attention_mask)
        
        assert isinstance(output, RouterOutput)
        assert output.adapter_weights.shape == (1, 4)
        assert output.complexity_score.shape == (1, 3)
        assert output.capabilities.shape == (1, 6)
        assert output.confidence.shape == (1,)

    def test_adapter_weights_sum_to_one(self, router, sample_input):
        """Test that adapter weights sum to approximately 1."""
        input_ids, attention_mask = sample_input
        
        output = router(input_ids, attention_mask)
        weights_sum = output.adapter_weights.sum(dim=-1)
        
        assert weights_sum.item() == pytest.approx(1.0, rel=1e-4)

    def test_hard_routing(self, router, sample_input):
        """Test hard routing mode."""
        input_ids, attention_mask = sample_input
        
        output = router(input_ids, attention_mask, hard=True)
        
        # Hard routing should have one-hot weights
        assert output.adapter_weights.max() == 1.0
        assert output.adapter_weights.sum() == 1.0

    def test_return_logits(self, router, sample_input):
        """Test returning raw logits."""
        input_ids, attention_mask = sample_input
        
        result = router(input_ids, attention_mask, return_logits=True)
        
        assert isinstance(result, dict)
        assert "weight_logits" in result
        assert "complexity_logits" in result

    def test_freeze_unfreeze_encoder(self, router):
        """Test freezing and unfreezing encoder."""
        # Initially frozen
        for param in router.encoder.parameters():
            assert not param.requires_grad
        
        # Unfreeze
        router.unfreeze_encoder()
        for param in router.encoder.parameters():
            assert param.requires_grad
        
        # Freeze again
        router.freeze_encoder()
        for param in router.encoder.parameters():
            assert not param.requires_grad

    def test_routing_decision(self, router, sample_input):
        """Test human-readable routing decision."""
        input_ids, attention_mask = sample_input
        
        decision = router.get_routing_decision(
            input_ids,
            attention_mask,
            adapter_names=["reasoning", "code", "analysis", "creative"]
        )
        
        assert "selected_adapters" in decision
        assert "complexity" in decision
        assert "confidence" in decision
        assert "predicted_complexity" in decision


class TestDynamicRouter:
    """Tests for DynamicRouter."""

    def test_fallback_uniform(self):
        """Test uniform fallback strategy."""
        base_router = AdapterRouter(
            encoder_name="prajjwal1/bert-tiny",
            num_adapters=4
        )
        
        dynamic_router = DynamicRouter(
            base_router=base_router,
            min_confidence=1.0,  # Always trigger fallback
            fallback_strategy="uniform"
        )
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        encoding = tokenizer("Test query", return_tensors="pt", padding="max_length", max_length=64)
        
        output = dynamic_router(encoding["input_ids"], encoding["attention_mask"])
        
        # Uniform weights
        expected_weight = 1.0 / 4
        for i in range(4):
            assert output.adapter_weights[0, i].item() == pytest.approx(expected_weight, rel=1e-4)


class TestCreateRouter:
    """Tests for create_router factory function."""

    def test_create_with_config(self):
        """Test creating router from config."""
        config = {
            "encoder_name": "prajjwal1/bert-tiny",
            "num_adapters": 4,
            "hidden_dim": 128
        }
        
        router = create_router(config, device="cpu")
        
        assert isinstance(router, AdapterRouter)
        assert router.num_adapters == 4
