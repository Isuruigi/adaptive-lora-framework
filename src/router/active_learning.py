"""
Active learning strategies for router training.

Features:
- Uncertainty sampling (entropy, margin, least confidence)
- Diversity sampling (clustering, coreset)
- Hybrid strategies (uncertainty + diversity)
- Query batch selection
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuerySample:
    """Sample from the unlabeled pool."""
    
    text: str
    embedding: Optional[np.ndarray] = None
    uncertainty_score: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""
    
    @abstractmethod
    def score(
        self,
        model: torch.nn.Module,
        samples: List[QuerySample],
        tokenizer,
        device: str = "cuda"
    ) -> List[float]:
        """Score samples for selection.
        
        Args:
            model: Router model.
            samples: Unlabeled samples.
            tokenizer: Tokenizer for encoding.
            device: Device for inference.
            
        Returns:
            List of scores (higher = more informative).
        """
        pass


class UncertaintySampling(SamplingStrategy):
    """Uncertainty-based sampling strategies."""
    
    def __init__(self, strategy: str = "entropy"):
        """Initialize uncertainty sampler.
        
        Args:
            strategy: One of 'entropy', 'margin', 'least_confidence'.
        """
        self.strategy = strategy
        
    def score(
        self,
        model: torch.nn.Module,
        samples: List[QuerySample],
        tokenizer,
        device: str = "cuda"
    ) -> List[float]:
        """Compute uncertainty scores."""
        model.eval()
        scores = []
        
        with torch.no_grad():
            for sample in samples:
                # Tokenize
                inputs = tokenizer(
                    sample.text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions
                output = model(**inputs)
                probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
                
                # Compute uncertainty
                if self.strategy == "entropy":
                    score = self._entropy(probs)
                elif self.strategy == "margin":
                    score = self._margin(probs)
                elif self.strategy == "least_confidence":
                    score = self._least_confidence(probs)
                else:
                    score = self._entropy(probs)
                    
                scores.append(score)
                sample.uncertainty_score = score
                
        return scores
    
    def _entropy(self, probs: torch.Tensor) -> float:
        """Compute entropy of predictions."""
        probs = probs.squeeze()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()
    
    def _margin(self, probs: torch.Tensor) -> float:
        """Compute 1 - margin between top 2 predictions."""
        probs = probs.squeeze()
        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[0] - sorted_probs[1]
        return 1.0 - margin.item()
    
    def _least_confidence(self, probs: torch.Tensor) -> float:
        """Compute 1 - max probability."""
        return 1.0 - torch.max(probs).item()


class DiversitySampling(SamplingStrategy):
    """Diversity-based sampling strategies."""
    
    def __init__(self, strategy: str = "clustering", n_clusters: int = 10):
        """Initialize diversity sampler.
        
        Args:
            strategy: One of 'clustering', 'coreset'.
            n_clusters: Number of clusters for clustering strategy.
        """
        self.strategy = strategy
        self.n_clusters = n_clusters
        
    def score(
        self,
        model: torch.nn.Module,
        samples: List[QuerySample],
        tokenizer,
        device: str = "cuda"
    ) -> List[float]:
        """Compute diversity scores based on embeddings."""
        embeddings = self._get_embeddings(model, samples, tokenizer, device)
        
        if self.strategy == "clustering":
            return self._cluster_diversity(samples, embeddings)
        elif self.strategy == "coreset":
            return self._coreset_diversity(samples, embeddings)
        else:
            return self._cluster_diversity(samples, embeddings)
    
    def _get_embeddings(
        self,
        model: torch.nn.Module,
        samples: List[QuerySample],
        tokenizer,
        device: str
    ) -> np.ndarray:
        """Extract embeddings from model."""
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for sample in samples:
                inputs = tokenizer(
                    sample.text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get encoder hidden states
                if hasattr(model, 'encoder'):
                    outputs = model.encoder(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    embedding = np.random.randn(1, 256)
                    
                embeddings.append(embedding.flatten())
                sample.embedding = embedding.flatten()
                
        return np.array(embeddings)
    
    def _cluster_diversity(
        self,
        samples: List[QuerySample],
        embeddings: np.ndarray
    ) -> List[float]:
        """Score samples by distance to cluster centers."""
        n_samples = len(samples)
        n_clusters = min(self.n_clusters, n_samples)
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        # Score by distance to cluster center
        scores = []
        for i, (sample, label) in enumerate(zip(samples, cluster_labels)):
            distance = np.linalg.norm(embeddings[i] - cluster_centers[label])
            scores.append(distance)
            sample.diversity_score = distance
            
        # Normalize scores
        max_score = max(scores) if scores else 1.0
        scores = [s / max_score for s in scores]
        
        return scores
    
    def _coreset_diversity(
        self,
        samples: List[QuerySample],
        embeddings: np.ndarray
    ) -> List[float]:
        """Score samples by distance to nearest labeled sample."""
        n = len(samples)
        scores = []
        
        # Simple greedy coreset approximation
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    min_dist = min(min_dist, dist)
            scores.append(min_dist)
            samples[i].diversity_score = min_dist
            
        return scores


class HybridSampling(SamplingStrategy):
    """Combine uncertainty and diversity sampling."""
    
    def __init__(
        self,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        uncertainty_strategy: str = "entropy",
        diversity_strategy: str = "clustering"
    ):
        """Initialize hybrid sampler.
        
        Args:
            uncertainty_weight: Weight for uncertainty scores.
            diversity_weight: Weight for diversity scores.
            uncertainty_strategy: Strategy for uncertainty.
            diversity_strategy: Strategy for diversity.
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_sampler = UncertaintySampling(uncertainty_strategy)
        self.diversity_sampler = DiversitySampling(diversity_strategy)
        
    def score(
        self,
        model: torch.nn.Module,
        samples: List[QuerySample],
        tokenizer,
        device: str = "cuda"
    ) -> List[float]:
        """Compute hybrid scores."""
        # Get individual scores
        uncertainty_scores = self.uncertainty_sampler.score(model, samples, tokenizer, device)
        diversity_scores = self.diversity_sampler.score(model, samples, tokenizer, device)
        
        # Normalize
        u_max = max(uncertainty_scores) if max(uncertainty_scores) > 0 else 1.0
        d_max = max(diversity_scores) if max(diversity_scores) > 0 else 1.0
        
        uncertainty_scores = [s / u_max for s in uncertainty_scores]
        diversity_scores = [s / d_max for s in diversity_scores]
        
        # Combine
        combined = []
        for i, sample in enumerate(samples):
            score = (
                self.uncertainty_weight * uncertainty_scores[i] +
                self.diversity_weight * diversity_scores[i]
            )
            combined.append(score)
            sample.combined_score = score
            
        return combined


class ActiveLearningSelector:
    """Select samples from unlabeled pool for labeling."""
    
    def __init__(
        self,
        strategy: Union[str, SamplingStrategy] = "hybrid",
        batch_size: int = 100,
        uncertainty_strategy: str = "entropy",
        diversity_strategy: str = "clustering"
    ):
        """Initialize selector.
        
        Args:
            strategy: Sampling strategy or name.
            batch_size: Number of samples to select.
            uncertainty_strategy: Strategy for uncertainty sampling.
            diversity_strategy: Strategy for diversity sampling.
        """
        self.batch_size = batch_size
        
        if isinstance(strategy, SamplingStrategy):
            self.sampler = strategy
        elif strategy == "uncertainty":
            self.sampler = UncertaintySampling(uncertainty_strategy)
        elif strategy == "diversity":
            self.sampler = DiversitySampling(diversity_strategy)
        elif strategy == "hybrid":
            self.sampler = HybridSampling(
                uncertainty_strategy=uncertainty_strategy,
                diversity_strategy=diversity_strategy
            )
        else:
            self.sampler = UncertaintySampling("entropy")
            
    def select(
        self,
        model: torch.nn.Module,
        unlabeled_pool: List[str],
        tokenizer,
        device: str = "cuda",
        return_scores: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """Select samples from unlabeled pool.
        
        Args:
            model: Router model.
            unlabeled_pool: List of unlabeled queries.
            tokenizer: Tokenizer.
            device: Device for inference.
            return_scores: Whether to return scores.
            
        Returns:
            Selected queries (and optionally their scores).
        """
        samples = [QuerySample(text=text) for text in unlabeled_pool]
        
        scores = self.sampler.score(model, samples, tokenizer, device)
        
        # Sort by score (descending)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top samples
        selected_indices = [idx for idx, _ in indexed_scores[:self.batch_size]]
        selected_texts = [unlabeled_pool[idx] for idx in selected_indices]
        selected_scores = [scores[idx] for idx in selected_indices]
        
        logger.info(
            f"Selected {len(selected_texts)} samples. "
            f"Score range: [{min(selected_scores):.4f}, {max(selected_scores):.4f}]"
        )
        
        if return_scores:
            return selected_texts, selected_scores
        return selected_texts
    
    def batch_select(
        self,
        model: torch.nn.Module,
        unlabeled_pool: List[str],
        tokenizer,
        device: str = "cuda",
        num_batches: int = 5
    ) -> List[List[str]]:
        """Select multiple batches progressively.
        
        Useful for iterative active learning where the model
        is updated between batches.
        
        Args:
            model: Router model.
            unlabeled_pool: List of unlabeled queries.
            tokenizer: Tokenizer.
            device: Device.
            num_batches: Number of batches to select.
            
        Returns:
            List of batches, each containing selected queries.
        """
        remaining = unlabeled_pool.copy()
        batches = []
        
        for i in range(num_batches):
            if len(remaining) < self.batch_size:
                batches.append(remaining)
                break
                
            selected = self.select(model, remaining, tokenizer, device)
            batches.append(selected)
            
            # Remove selected from pool
            selected_set = set(selected)
            remaining = [q for q in remaining if q not in selected_set]
            
            logger.info(f"Batch {i+1}: selected {len(selected)}, remaining pool: {len(remaining)}")
            
        return batches


class QueryByCommittee:
    """Query-by-committee active learning."""
    
    def __init__(self, committee_size: int = 5):
        """Initialize QBC.
        
        Args:
            committee_size: Number of models in committee.
        """
        self.committee_size = committee_size
        self.committee: List[torch.nn.Module] = []
        
    def add_model(self, model: torch.nn.Module) -> None:
        """Add model to committee."""
        self.committee.append(model)
        
    def score_disagreement(
        self,
        queries: List[str],
        tokenizer,
        device: str = "cuda"
    ) -> List[float]:
        """Score queries by committee disagreement.
        
        Args:
            queries: List of queries.
            tokenizer: Tokenizer.
            device: Device.
            
        Returns:
            Disagreement scores (higher = more disagreement).
        """
        if len(self.committee) < 2:
            raise ValueError("Need at least 2 models in committee")
            
        scores = []
        
        for query in queries:
            inputs = tokenizer(
                query,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions from all committee members
            predictions = []
            for model in self.committee:
                model.eval()
                with torch.no_grad():
                    output = model(**inputs)
                    probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
                    predictions.append(probs.cpu().numpy())
                    
            # Compute vote entropy
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            vote_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
            
            scores.append(vote_entropy)
            
        return scores
