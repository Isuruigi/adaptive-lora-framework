"""
Neural router for dynamic adapter selection.

Features:
- Lightweight BERT-based encoder
- Multi-head prediction (complexity, capability, weights)
- Gumbel-Softmax for differentiable routing
- Hierarchical and ensemble routing options
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RouterOutput:
    """Output from router network.

    Attributes:
        adapter_weights: Weights for each adapter (batch_size, num_adapters).
        complexity_score: Complexity prediction (batch_size, 3).
        capabilities: Capability scores (batch_size, num_capabilities).
        confidence: Routing confidence (batch_size,).
    """

    adapter_weights: torch.Tensor
    complexity_score: torch.Tensor
    capabilities: torch.Tensor
    confidence: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_weights": self.adapter_weights.cpu().numpy().tolist(),
            "complexity_score": self.complexity_score.cpu().numpy().tolist(),
            "capabilities": self.capabilities.cpu().numpy().tolist(),
            "confidence": self.confidence.cpu().numpy().tolist()
        }

    def get_top_adapters(self, threshold: float = 0.1) -> List[Tuple[int, float]]:
        """Get adapters above threshold.

        Args:
            threshold: Minimum weight threshold.

        Returns:
            List of (adapter_idx, weight) tuples.
        """
        weights = self.adapter_weights[0]  # First batch item
        indices = torch.where(weights > threshold)[0]

        return [
            (int(idx), float(weights[idx]))
            for idx in indices
        ]


class AdapterRouter(nn.Module):
    """Neural router for dynamic adapter selection.

    Architecture:
    - Encoder: Pre-trained language model (BERT-tiny/DistilBERT)
    - Complexity Head: Predicts query complexity (easy/medium/hard)
    - Capability Head: Predicts required capabilities
    - Weight Head: Predicts adapter mixing weights
    - Confidence Head: Estimates routing confidence

    Example:
        >>> router = AdapterRouter(num_adapters=4)
        >>> output = router(input_ids, attention_mask)
        >>> print(output.adapter_weights)
    """

    def __init__(
        self,
        encoder_name: str = "prajjwal1/bert-tiny",
        num_adapters: int = 4,
        num_capabilities: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_gumbel_softmax: bool = True,
        temperature: float = 1.0
    ):
        """Initialize router.

        Args:
            encoder_name: Pre-trained encoder model name.
            num_adapters: Number of available adapters.
            num_capabilities: Number of capability dimensions.
            hidden_dim: Hidden dimension size.
            dropout: Dropout probability.
            use_gumbel_softmax: Use Gumbel-Softmax for routing.
            temperature: Temperature for Gumbel-Softmax.
        """
        super().__init__()

        self.num_adapters = num_adapters
        self.num_capabilities = num_capabilities
        self.use_gumbel_softmax = use_gumbel_softmax
        self.temperature = temperature

        # Encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size

        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Projection layer
        self.projection = nn.Linear(encoder_dim, hidden_dim)

        # Complexity head (easy, medium, hard)
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Capability head (multi-label)
        self.capability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_capabilities)
        )

        # Weight head (adapter selection)
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_adapters)
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logits: bool = False,
        hard: bool = False
    ) -> RouterOutput:
        """Forward pass.

        Args:
            input_ids: Tokenized input (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            return_logits: Return raw logits instead of probabilities.
            hard: Use hard routing (argmax).

        Returns:
            RouterOutput with predictions.
        """
        # Encode query
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        # Project
        hidden = F.relu(self.projection(cls_embedding))

        # Predict complexity
        complexity_logits = self.complexity_head(hidden)
        complexity_probs = F.softmax(complexity_logits, dim=-1)

        # Predict capabilities
        capability_logits = self.capability_head(hidden)
        capability_probs = torch.sigmoid(capability_logits)

        # Predict adapter weights
        weight_logits = self.weight_head(hidden)

        if hard:
            # Hard routing - select single adapter
            adapter_weights = F.one_hot(
                torch.argmax(weight_logits, dim=-1),
                num_classes=self.num_adapters
            ).float()
        elif self.use_gumbel_softmax and self.training:
            # Gumbel-Softmax for differentiable routing
            adapter_weights = F.gumbel_softmax(
                weight_logits,
                tau=self.temperature,
                hard=False,
                dim=-1
            )
        else:
            # Soft routing
            adapter_weights = F.softmax(weight_logits, dim=-1)

        # Predict confidence
        confidence = self.confidence_head(hidden).squeeze(-1)

        if return_logits:
            return {
                "weight_logits": weight_logits,
                "complexity_logits": complexity_logits,
                "capability_logits": capability_logits
            }

        return RouterOutput(
            adapter_weights=adapter_weights,
            complexity_score=complexity_probs,
            capabilities=capability_probs,
            confidence=confidence
        )

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def get_routing_decision(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.1,
        adapter_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get human-readable routing decision.

        Args:
            input_ids: Tokenized input.
            attention_mask: Attention mask.
            threshold: Minimum weight to include adapter.
            adapter_names: Optional names for adapters.

        Returns:
            Dictionary with routing decision.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

        # Get top adapters
        weights = output.adapter_weights[0]
        top_indices = torch.where(weights > threshold)[0]

        adapter_names = adapter_names or [f"adapter_{i}" for i in range(self.num_adapters)]

        routing_decision = {
            "selected_adapters": [
                {
                    "name": adapter_names[int(idx)],
                    "index": int(idx),
                    "weight": float(weights[idx])
                }
                for idx in top_indices
            ],
            "complexity": {
                "easy": float(output.complexity_score[0, 0]),
                "medium": float(output.complexity_score[0, 1]),
                "hard": float(output.complexity_score[0, 2])
            },
            "predicted_complexity": ["easy", "medium", "hard"][
                output.complexity_score[0].argmax().item()
            ],
            "confidence": float(output.confidence[0]),
            "all_weights": weights.cpu().numpy().tolist()
        }

        return routing_decision


class HierarchicalRouter(nn.Module):
    """Two-stage hierarchical router.

    Stage 1: Select domain/category
    Stage 2: Select specific adapters within domain
    """

    def __init__(
        self,
        encoder_name: str = "prajjwal1/bert-tiny",
        num_domains: int = 3,
        adapters_per_domain: Optional[Dict[int, int]] = None,
        hidden_dim: int = 256
    ):
        """Initialize hierarchical router.

        Args:
            encoder_name: Pre-trained encoder model.
            num_domains: Number of domains.
            adapters_per_domain: Adapters per domain.
            hidden_dim: Hidden dimension.
        """
        super().__init__()

        self.num_domains = num_domains
        self.adapters_per_domain = adapters_per_domain or {
            i: 2 for i in range(num_domains)
        }

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size

        self.projection = nn.Linear(encoder_dim, hidden_dim)

        # Domain router (coarse)
        self.domain_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_domains)
        )

        # Adapter routers (fine) - one per domain
        self.adapter_routers = nn.ModuleDict({
            str(domain_id): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_adapters)
            )
            for domain_id, num_adapters in self.adapters_per_domain.items()
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward pass.

        Returns:
            - Domain weights (batch_size, num_domains)
            - Adapter weights per domain
        """
        # Encode
        encoder_output = self.encoder(input_ids, attention_mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        hidden = F.relu(self.projection(cls_embedding))

        # Route to domain
        domain_logits = self.domain_router(hidden)
        domain_weights = F.softmax(domain_logits, dim=-1)

        # Route within each domain
        adapter_weights = {}
        for domain_id, router in self.adapter_routers.items():
            logits = router(hidden)
            weights = F.softmax(logits, dim=-1)
            adapter_weights[int(domain_id)] = weights

        return domain_weights, adapter_weights


class DynamicRouter(nn.Module):
    """Router with adaptive behavior based on query characteristics."""

    def __init__(
        self,
        base_router: AdapterRouter,
        min_confidence: float = 0.5,
        fallback_strategy: str = "uniform"
    ):
        """Initialize dynamic router.

        Args:
            base_router: Base router model.
            min_confidence: Minimum confidence threshold.
            fallback_strategy: Strategy when confidence is low.
        """
        super().__init__()

        self.base_router = base_router
        self.min_confidence = min_confidence
        self.fallback_strategy = fallback_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> RouterOutput:
        """Forward with adaptive routing."""
        output = self.base_router(input_ids, attention_mask)

        # Check confidence
        if output.confidence.mean() < self.min_confidence:
            if self.fallback_strategy == "uniform":
                # Uniform distribution
                output.adapter_weights = torch.ones_like(output.adapter_weights)
                output.adapter_weights = output.adapter_weights / output.adapter_weights.sum(
                    dim=-1, keepdim=True
                )
            elif self.fallback_strategy == "top_k":
                # Select top-2 uniformly
                k = 2
                top_k_indices = torch.topk(output.adapter_weights, k, dim=-1).indices
                new_weights = torch.zeros_like(output.adapter_weights)
                new_weights.scatter_(1, top_k_indices, 1.0 / k)
                output.adapter_weights = new_weights

        return output


class RouterEnsemble(nn.Module):
    """Ensemble of multiple routers for robust routing."""

    def __init__(
        self,
        routers: List[AdapterRouter],
        aggregation: str = "weighted"
    ):
        """Initialize ensemble.

        Args:
            routers: List of router models.
            aggregation: Aggregation method (average, weighted).
        """
        super().__init__()

        self.routers = nn.ModuleList(routers)
        self.aggregation = aggregation

        if aggregation == "weighted":
            self.router_weights = nn.Parameter(torch.ones(len(routers)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> RouterOutput:
        """Aggregate predictions from multiple routers."""
        outputs = [
            router(input_ids, attention_mask)
            for router in self.routers
        ]

        if self.aggregation == "average":
            adapter_weights = torch.stack(
                [o.adapter_weights for o in outputs]
            ).mean(dim=0)
            complexity_score = torch.stack(
                [o.complexity_score for o in outputs]
            ).mean(dim=0)
            capabilities = torch.stack(
                [o.capabilities for o in outputs]
            ).mean(dim=0)
            confidence = torch.stack(
                [o.confidence for o in outputs]
            ).mean(dim=0)

        elif self.aggregation == "weighted":
            weights = F.softmax(self.router_weights, dim=0)
            adapter_weights = sum(
                w * o.adapter_weights for w, o in zip(weights, outputs)
            )
            complexity_score = sum(
                w * o.complexity_score for w, o in zip(weights, outputs)
            )
            capabilities = sum(
                w * o.capabilities for w, o in zip(weights, outputs)
            )
            confidence = sum(
                w * o.confidence for w, o in zip(weights, outputs)
            )

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return RouterOutput(
            adapter_weights=adapter_weights,
            complexity_score=complexity_score,
            capabilities=capabilities,
            confidence=confidence
        )


def create_router(
    config: Dict[str, Any],
    device: str = "cuda"
) -> AdapterRouter:
    """Create router from configuration.

    Args:
        config: Router configuration.
        device: Target device.

    Returns:
        Configured router.
    """
    router = AdapterRouter(
        encoder_name=config.get("encoder_name", "prajjwal1/bert-tiny"),
        num_adapters=config.get("num_adapters", 4),
        num_capabilities=config.get("num_capabilities", 6),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
        use_gumbel_softmax=config.get("use_gumbel_softmax", True),
        temperature=config.get("temperature", 1.0)
    )

    return router.to(device)
