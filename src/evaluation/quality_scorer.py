"""
Quality scoring for model outputs.

Provides customizable scoring rubrics and aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityDimension:
    """A single quality dimension for scoring.

    Attributes:
        name: Dimension name.
        weight: Weight in overall score.
        scorer: Scoring function.
        threshold: Minimum acceptable score.
    """

    name: str
    weight: float
    scorer: Callable[[str, str], float]
    threshold: float = 0.0


class QualityScorer:
    """Flexible quality scorer with customizable rubrics.

    Example:
        >>> scorer = QualityScorer()
        >>> scorer.add_dimension("length", 0.3, length_scorer)
        >>> score = scorer.score(query, response)
    """

    def __init__(
        self,
        dimensions: Optional[List[QualityDimension]] = None
    ):
        """Initialize scorer.

        Args:
            dimensions: Initial quality dimensions.
        """
        self.dimensions = dimensions or []
        self._normalize_weights()

    def add_dimension(
        self,
        name: str,
        weight: float,
        scorer: Callable[[str, str], float],
        threshold: float = 0.0
    ) -> None:
        """Add a quality dimension.

        Args:
            name: Dimension name.
            weight: Weight in overall score.
            scorer: Scoring function.
            threshold: Minimum acceptable score.
        """
        self.dimensions.append(QualityDimension(
            name=name,
            weight=weight,
            scorer=scorer,
            threshold=threshold
        ))
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize dimension weights to sum to 1."""
        total = sum(d.weight for d in self.dimensions)
        if total > 0:
            for d in self.dimensions:
                d.weight = d.weight / total

    def score(
        self,
        query: str,
        response: str,
        return_breakdown: bool = False
    ) -> Dict[str, Any]:
        """Score a response.

        Args:
            query: Input query.
            response: Model response.
            return_breakdown: Include per-dimension scores.

        Returns:
            Score dictionary.
        """
        dimension_scores = {}
        weighted_sum = 0.0
        below_threshold = []

        for dim in self.dimensions:
            try:
                score = dim.scorer(query, response)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except Exception as e:
                logger.warning(f"Scorer {dim.name} failed: {e}")
                score = 0.5

            dimension_scores[dim.name] = score
            weighted_sum += dim.weight * score

            if score < dim.threshold:
                below_threshold.append(dim.name)

        result = {
            "overall_score": weighted_sum,
            "passed": len(below_threshold) == 0
        }

        if return_breakdown:
            result["dimension_scores"] = dimension_scores
            result["below_threshold"] = below_threshold

        return result

    def score_batch(
        self,
        queries: List[str],
        responses: List[str]
    ) -> List[Dict[str, Any]]:
        """Score a batch of responses.

        Args:
            queries: List of queries.
            responses: List of responses.

        Returns:
            List of score dictionaries.
        """
        return [
            self.score(q, r, return_breakdown=True)
            for q, r in zip(queries, responses)
        ]

    def aggregate_scores(
        self,
        scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate multiple scores.

        Args:
            scores: List of score dictionaries.

        Returns:
            Aggregated statistics.
        """
        overall_scores = [s["overall_score"] for s in scores]
        pass_rate = sum(1 for s in scores if s.get("passed", True)) / len(scores)

        # Per-dimension aggregation
        dimension_stats = {}
        if scores and "dimension_scores" in scores[0]:
            for dim_name in scores[0]["dimension_scores"]:
                dim_scores = [s["dimension_scores"][dim_name] for s in scores]
                dimension_stats[dim_name] = {
                    "mean": np.mean(dim_scores),
                    "std": np.std(dim_scores),
                    "min": min(dim_scores),
                    "max": max(dim_scores)
                }

        return {
            "mean_score": np.mean(overall_scores),
            "std_score": np.std(overall_scores),
            "min_score": min(overall_scores),
            "max_score": max(overall_scores),
            "pass_rate": pass_rate,
            "dimension_stats": dimension_stats
        }


def length_scorer(query: str, response: str) -> float:
    """Score based on response length.

    Args:
        query: Input query.
        response: Model response.

    Returns:
        Score (0-1).
    """
    length = len(response)

    # Optimal length depends on query complexity
    query_length = len(query)

    if query_length < 50:
        optimal_min, optimal_max = 50, 300
    elif query_length < 200:
        optimal_min, optimal_max = 100, 500
    else:
        optimal_min, optimal_max = 200, 1000

    if length < optimal_min:
        return length / optimal_min
    elif length <= optimal_max:
        return 1.0
    else:
        return max(0.0, 1.0 - (length - optimal_max) / optimal_max)


def specificity_scorer(query: str, response: str) -> float:
    """Score based on specificity.

    Args:
        query: Input query.
        response: Model response.

    Returns:
        Score (0-1).
    """
    import re

    score = 0.5

    # Specific details indicators
    if re.search(r"\d+", response):  # Numbers
        score += 0.1

    if re.search(r"\b\d{4}\b", response):  # Years
        score += 0.1

    # Named entities (simple heuristic)
    capitalized = re.findall(r"\b[A-Z][a-z]+\b", response)
    if len(capitalized) > 3:
        score += 0.15

    # Technical terms
    technical_indicators = ["algorithm", "method", "function", "process", "system"]
    if any(term in response.lower() for term in technical_indicators):
        score += 0.15

    return min(1.0, score)


def structure_scorer(query: str, response: str) -> float:
    """Score based on response structure.

    Args:
        query: Input query.
        response: Model response.

    Returns:
        Score (0-1).
    """
    score = 0.5

    # Multiple sentences
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if len(sentences) >= 2:
        score += 0.2

    # Paragraphs
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        score += 0.15

    # Lists
    if any(c in response for c in ["1.", "2.", "-", "•"]):
        score += 0.15

    return min(1.0, score)


def create_default_scorer() -> QualityScorer:
    """Create scorer with default dimensions.

    Returns:
        Configured QualityScorer.
    """
    scorer = QualityScorer()

    scorer.add_dimension("length", 0.25, length_scorer, threshold=0.3)
    scorer.add_dimension("specificity", 0.35, specificity_scorer, threshold=0.4)
    scorer.add_dimension("structure", 0.40, structure_scorer, threshold=0.3)

    return scorer
