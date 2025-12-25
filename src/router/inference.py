"""
Router inference utilities for production use.

Features:
- Efficient batch inference
- Caching for repeated queries
- Integration with adapters
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from src.router.router_model import AdapterRouter, RouterOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RouterInference:
    """Production inference wrapper for router.

    Provides efficient inference with caching and batching.

    Example:
        >>> inference = RouterInference.from_pretrained("./router_checkpoint")
        >>> result = inference.route("What is machine learning?")
        >>> print(result["selected_adapter"])
    """

    def __init__(
        self,
        router: AdapterRouter,
        tokenizer,
        adapter_names: Optional[List[str]] = None,
        device: str = "cuda",
        cache_size: int = 1000
    ):
        """Initialize inference.

        Args:
            router: Trained router model.
            tokenizer: Tokenizer for encoding.
            adapter_names: Names for adapters.
            device: Target device.
            cache_size: Size of result cache.
        """
        self.router = router
        self.tokenizer = tokenizer
        self.device = device
        self.cache_size = cache_size

        self.router.to(device)
        self.router.eval()

        # Default adapter names
        self.adapter_names = adapter_names or [
            f"adapter_{i}" for i in range(router.num_adapters)
        ]

        # Simple cache
        self._cache: Dict[str, Dict] = {}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        **kwargs
    ) -> "RouterInference":
        """Load from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
            device: Target device.
            **kwargs: Additional arguments.

        Returns:
            RouterInference instance.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path / "router.pt",
            map_location=device
        )

        # Load config
        import json
        with open(checkpoint_path / "config.json", "r") as f:
            config = json.load(f)

        # Create router
        router = AdapterRouter(
            encoder_name=config.get("encoder_name", "prajjwal1/bert-tiny"),
            num_adapters=config.get("num_adapters", 4),
            num_capabilities=config.get("num_capabilities", 6),
            hidden_dim=config.get("hidden_dim", 256)
        )

        router.load_state_dict(checkpoint["model_state_dict"])

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.get("encoder_name", "prajjwal1/bert-tiny")
        )

        return cls(
            router=router,
            tokenizer=tokenizer,
            adapter_names=config.get("adapter_names"),
            device=device,
            **kwargs
        )

    @torch.no_grad()
    def route(
        self,
        query: str,
        use_cache: bool = True,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Route a single query.

        Args:
            query: Input query.
            use_cache: Use result caching.
            threshold: Minimum weight threshold.

        Returns:
            Routing decision dictionary.
        """
        # Check cache
        if use_cache and query in self._cache:
            return self._cache[query]

        # Tokenize
        encoding = self.tokenizer(
            query,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Route
        output = self.router(input_ids, attention_mask)

        # Build result
        result = self._build_result(output, threshold)

        # Cache
        if use_cache:
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[query] = result

        return result

    @torch.no_grad()
    def route_batch(
        self,
        queries: List[str],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Route a batch of queries.

        Args:
            queries: List of input queries.
            threshold: Minimum weight threshold.

        Returns:
            List of routing decisions.
        """
        # Tokenize batch
        encoding = self.tokenizer(
            queries,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Route
        output = self.router(input_ids, attention_mask)

        # Build results
        results = []
        batch_size = input_ids.size(0)

        for i in range(batch_size):
            batch_output = RouterOutput(
                adapter_weights=output.adapter_weights[i:i+1],
                complexity_score=output.complexity_score[i:i+1],
                capabilities=output.capabilities[i:i+1],
                confidence=output.confidence[i:i+1]
            )
            result = self._build_result(batch_output, threshold)
            results.append(result)

        return results

    def _build_result(
        self,
        output: RouterOutput,
        threshold: float
    ) -> Dict[str, Any]:
        """Build result dictionary from router output."""
        weights = output.adapter_weights[0]
        complexity = output.complexity_score[0]

        # Get selected adapters
        top_indices = torch.where(weights > threshold)[0]

        selected_adapters = [
            {
                "name": self.adapter_names[int(idx)],
                "index": int(idx),
                "weight": float(weights[idx])
            }
            for idx in top_indices
        ]

        # Sort by weight
        selected_adapters.sort(key=lambda x: x["weight"], reverse=True)

        # Primary adapter
        primary_idx = weights.argmax().item()

        return {
            "selected_adapters": selected_adapters,
            "primary_adapter": self.adapter_names[primary_idx],
            "primary_weight": float(weights[primary_idx]),
            "complexity": {
                "easy": float(complexity[0]),
                "medium": float(complexity[1]),
                "hard": float(complexity[2]),
                "predicted": ["easy", "medium", "hard"][complexity.argmax().item()]
            },
            "confidence": float(output.confidence[0]),
            "all_weights": {
                name: float(weights[i])
                for i, name in enumerate(self.adapter_names)
            }
        }

    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()

    def get_adapter_for_query(self, query: str) -> str:
        """Get primary adapter name for query.

        Args:
            query: Input query.

        Returns:
            Adapter name.
        """
        result = self.route(query)
        return result["primary_adapter"]

    def get_weights_for_query(self, query: str) -> Dict[str, float]:
        """Get adapter weights for query.

        Args:
            query: Input query.

        Returns:
            Dictionary of adapter name to weight.
        """
        result = self.route(query)
        return result["all_weights"]


class RouterWithAdapters:
    """Combined router and adapter inference.

    Routes queries and applies appropriate adapters.
    """

    def __init__(
        self,
        router_inference: RouterInference,
        adapter_models: Dict[str, Any],
        base_model: Any,
        tokenizer: Any
    ):
        """Initialize.

        Args:
            router_inference: Router inference instance.
            adapter_models: Dictionary of adapter name to model.
            base_model: Base model for generation.
            tokenizer: Tokenizer for generation.
        """
        self.router = router_inference
        self.adapter_models = adapter_models
        self.base_model = base_model
        self.tokenizer = tokenizer

    def generate(
        self,
        query: str,
        max_new_tokens: int = 256,
        use_routing: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Generate response with automatic adapter selection.

        Args:
            query: Input query.
            max_new_tokens: Maximum tokens to generate.
            use_routing: Use router for adapter selection.
            **generation_kwargs: Additional generation arguments.

        Returns:
            Dictionary with response and metadata.
        """
        # Route query
        if use_routing:
            routing_result = self.router.route(query)
            adapter_name = routing_result["primary_adapter"]
        else:
            adapter_name = list(self.adapter_models.keys())[0]
            routing_result = None

        # Get adapter model
        model = self.adapter_models.get(adapter_name, self.base_model)

        # Generate
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return {
            "response": response,
            "adapter_used": adapter_name,
            "routing_result": routing_result
        }
