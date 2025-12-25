"""
Comprehensive self-evaluation system for model outputs.

Features:
- Multi-metric quality assessment
- Uncertainty quantification
- Failure pattern detection
- LLM-as-judge integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Result from self-evaluation.

    Attributes:
        query: Input query.
        response: Model response.
        adapter_used: Adapter that generated response.
        overall_score: Combined quality score.
        coherence_score: Coherence metric.
        relevance_score: Relevance to query.
        factuality_score: Factual accuracy.
        safety_score: Safety assessment.
        uncertainty_score: Uncertainty estimate.
        confidence: Overall confidence.
        is_failure: Whether this is a failure case.
        failure_type: Type of failure if applicable.
        metadata: Additional metadata.
    """

    query: str
    response: str
    adapter_used: str
    overall_score: float
    coherence_score: float
    relevance_score: float
    factuality_score: float
    safety_score: float
    uncertainty_score: float
    confidence: float
    is_failure: bool
    failure_type: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "adapter_used": self.adapter_used,
            "scores": {
                "overall": self.overall_score,
                "coherence": self.coherence_score,
                "relevance": self.relevance_score,
                "factuality": self.factuality_score,
                "safety": self.safety_score,
                "uncertainty": self.uncertainty_score
            },
            "confidence": self.confidence,
            "is_failure": self.is_failure,
            "failure_type": self.failure_type,
            "metadata": self.metadata
        }


class SelfEvaluator:
    """Comprehensive self-evaluation system.

    Evaluates model outputs across multiple dimensions.

    Example:
        >>> evaluator = SelfEvaluator(use_llm_judge=True)
        >>> result = evaluator.evaluate(query, response, adapter_used)
        >>> print(result.overall_score)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        use_llm_judge: bool = True,
        failure_threshold: float = 0.6
    ):
        """Initialize evaluator.

        Args:
            openai_api_key: OpenAI API key.
            anthropic_api_key: Anthropic API key.
            use_llm_judge: Use LLM for evaluation.
            failure_threshold: Score below which is failure.
        """
        self.use_llm_judge = use_llm_judge and (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE)
        self.failure_threshold = failure_threshold

        # Setup API clients
        if self.use_llm_judge:
            if openai_api_key and OPENAI_AVAILABLE:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            else:
                self.openai_client = None

            if anthropic_api_key and ANTHROPIC_AVAILABLE:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            else:
                self.anthropic_client = None
        else:
            self.openai_client = None
            self.anthropic_client = None

        # Embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = None

        # Metric weights
        self.metric_weights = {
            "coherence": 0.2,
            "relevance": 0.3,
            "factuality": 0.25,
            "safety": 0.15,
            "uncertainty": 0.1
        }

        # Failure pattern storage
        self.failure_patterns: List[Dict] = []

    def evaluate(
        self,
        query: str,
        response: str,
        adapter_used: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Comprehensive evaluation of model output.

        Args:
            query: Input query.
            response: Model's response.
            adapter_used: Adapter that generated this.
            ground_truth: Optional reference answer.
            context: Optional additional context.

        Returns:
            EvaluationResult with all metrics.
        """
        # Compute individual metrics
        coherence_score = self.check_coherence(query, response)
        relevance_score = self.check_relevance(query, response)
        safety_score = self.check_safety(response)
        uncertainty_score = self.estimate_uncertainty(query, response)

        # Factuality check
        if ground_truth:
            factuality_score = self.check_factuality_with_reference(
                response, ground_truth
            )
        else:
            factuality_score = self.check_factuality_heuristic(response)

        # LLM-as-judge
        llm_judge_score = 0.5
        if self.use_llm_judge:
            try:
                llm_judge_score = self.llm_judge_evaluation(query, response)
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}")

        # Weighted overall score
        overall_score = (
            self.metric_weights["coherence"] * coherence_score +
            self.metric_weights["relevance"] * relevance_score +
            self.metric_weights["factuality"] * factuality_score +
            self.metric_weights["safety"] * safety_score +
            self.metric_weights["uncertainty"] * (1 - uncertainty_score)
        )

        # Include LLM judge
        if self.use_llm_judge:
            overall_score = 0.6 * overall_score + 0.4 * llm_judge_score

        # Determine failure
        is_failure = overall_score < self.failure_threshold
        failure_type = None

        if is_failure:
            failure_type = self._classify_failure_type(
                coherence_score,
                relevance_score,
                factuality_score,
                safety_score,
                uncertainty_score
            )
            self._log_failure(
                query, response, adapter_used, failure_type, overall_score
            )

        return EvaluationResult(
            query=query,
            response=response,
            adapter_used=adapter_used,
            overall_score=overall_score,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            factuality_score=factuality_score,
            safety_score=safety_score,
            uncertainty_score=uncertainty_score,
            confidence=1 - uncertainty_score,
            is_failure=is_failure,
            failure_type=failure_type,
            metadata={"llm_judge_score": llm_judge_score if self.use_llm_judge else None}
        )

    def check_coherence(self, query: str, response: str) -> float:
        """Check coherence of response.

        Args:
            query: Input query.
            response: Model response.

        Returns:
            Coherence score (0-1).
        """
        score = 0.5

        # Length check
        if 20 < len(response) < 2000:
            score += 0.1

        # Complete sentences check
        sentences = response.split(".")
        if len(sentences) >= 2:
            score += 0.1

        # Keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap_ratio = len(query_words & response_words) / max(len(query_words), 1)
        score += 0.2 * min(overlap_ratio, 1.0)

        # No failure phrases
        failure_phrases = [
            "i cannot", "i'm unable", "i don't have",
            "sorry", "unfortunately", "as an ai"
        ]
        if not any(phrase in response.lower() for phrase in failure_phrases):
            score += 0.2

        return min(1.0, score)

    def check_relevance(self, query: str, response: str) -> float:
        """Check relevance using semantic similarity.

        Args:
            query: Input query.
            response: Model response.

        Returns:
            Relevance score (0-1).
        """
        if self.embedding_model is None:
            # Fallback to keyword matching
            return self.check_coherence(query, response)

        # Compute embeddings
        query_emb = self.embedding_model.encode(query)
        response_emb = self.embedding_model.encode(response)

        # Cosine similarity
        similarity = np.dot(query_emb, response_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(response_emb)
        )

        # Convert to 0-1 score
        return float((similarity + 1) / 2)

    def check_safety(self, response: str) -> float:
        """Check for safety issues.

        Args:
            response: Model response.

        Returns:
            Safety score (0-1).
        """
        score = 1.0

        # Toxic keywords (simple heuristic)
        toxic_keywords = [
            "kill", "die", "hurt", "harm", "attack", "violent",
            "hate", "stupid", "idiot"
        ]

        response_lower = response.lower()
        for keyword in toxic_keywords:
            if keyword in response_lower:
                score -= 0.15

        # PII patterns
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", response):
            score -= 0.3

        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", response):
            score -= 0.3

        return max(0.0, score)

    def check_factuality_heuristic(self, response: str) -> float:
        """Heuristic factuality check.

        Args:
            response: Model response.

        Returns:
            Factuality score (0-1).
        """
        score = 0.5

        # Hedging (some is good)
        hedging_words = ["might", "could", "possibly", "perhaps", "likely"]
        hedging_count = sum(1 for word in hedging_words if word in response.lower())

        if 1 <= hedging_count <= 3:
            score += 0.2
        elif hedging_count > 5:
            score -= 0.2

        # Specific details
        if re.search(r"\d+", response):
            score += 0.15

        if re.search(r"\b\d{4}\b", response):
            score += 0.15

        return min(1.0, score)

    def check_factuality_with_reference(
        self,
        response: str,
        ground_truth: str
    ) -> float:
        """Check factuality against reference.

        Args:
            response: Model response.
            ground_truth: Reference answer.

        Returns:
            Factuality score (0-1).
        """
        if self.embedding_model is None:
            return 0.5

        # Semantic similarity
        response_emb = self.embedding_model.encode(response)
        truth_emb = self.embedding_model.encode(ground_truth)

        similarity = np.dot(response_emb, truth_emb) / (
            np.linalg.norm(response_emb) * np.linalg.norm(truth_emb)
        )

        semantic_score = (similarity + 1) / 2

        # Fact extraction
        response_facts = set(re.findall(r"\b\d+\.?\d*\b", response))
        truth_facts = set(re.findall(r"\b\d+\.?\d*\b", ground_truth))

        if truth_facts:
            fact_accuracy = len(response_facts & truth_facts) / len(truth_facts)
        else:
            fact_accuracy = 1.0

        return float(0.7 * semantic_score + 0.3 * fact_accuracy)

    def estimate_uncertainty(
        self,
        query: str,
        response: str
    ) -> float:
        """Estimate uncertainty in response.

        Args:
            query: Input query.
            response: Model response.

        Returns:
            Uncertainty score (0-1).
        """
        uncertainty = 0.0

        # Short responses indicate uncertainty
        if len(response) < 50:
            uncertainty += 0.3

        # Hedging language
        hedging_count = sum(
            1 for word in ["maybe", "might", "could", "not sure", "unclear"]
            if word in response.lower()
        )
        uncertainty += min(0.4, hedging_count * 0.1)

        # Questions in response
        question_count = response.count("?")
        uncertainty += min(0.3, question_count * 0.1)

        return min(1.0, uncertainty)

    def llm_judge_evaluation(
        self,
        query: str,
        response: str,
        model: str = "gpt-4"
    ) -> float:
        """Use LLM as judge for quality assessment.

        Args:
            query: Input query.
            response: Model response.
            model: LLM model to use.

        Returns:
            Quality score (0-1).
        """
        judge_prompt = f"""Evaluate the quality of the following AI response.

Query: {query}

Response: {response}

Rate the response on these criteria (0-10 each):
1. Relevance: Does it address the query?
2. Accuracy: Is the information correct?
3. Completeness: Is it comprehensive?
4. Clarity: Is it well-written and clear?
5. Helpfulness: Would this help the user?

Provide scores in this exact format:
Relevance: X
Accuracy: X
Completeness: X
Clarity: X
Helpfulness: X
Overall: X (average of above)

Only provide the scores, no explanation."""

        try:
            if self.openai_client and model.startswith("gpt"):
                result = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.3
                )
                judgment = result.choices[0].message.content
            elif self.anthropic_client:
                result = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                judgment = result.content[0].text
            else:
                return 0.5

            # Parse scores
            overall_match = re.search(r"Overall:\s*(\d+(?:\.\d+)?)", judgment)
            if overall_match:
                return min(1.0, max(0.0, float(overall_match.group(1)) / 10.0))

            return 0.5

        except Exception as e:
            logger.warning(f"LLM judge error: {e}")
            return 0.5

    def _classify_failure_type(
        self,
        coherence: float,
        relevance: float,
        factuality: float,
        safety: float,
        uncertainty: float
    ) -> str:
        """Classify type of failure."""
        scores = {
            "coherence": coherence,
            "relevance": relevance,
            "factuality": factuality,
            "safety": safety
        }

        min_metric = min(scores.items(), key=lambda x: x[1])

        if min_metric[1] < 0.3:
            return f"severe_{min_metric[0]}_failure"
        elif uncertainty > 0.7:
            return "high_uncertainty"
        else:
            return "general_quality_issue"

    def _log_failure(
        self,
        query: str,
        response: str,
        adapter_used: str,
        failure_type: str,
        score: float
    ) -> None:
        """Log failure for pattern analysis."""
        self.failure_patterns.append({
            "query": query,
            "response": response,
            "adapter_used": adapter_used,
            "failure_type": failure_type,
            "score": score
        })

    def analyze_failure_patterns(
        self,
        min_cluster_size: int = 5
    ) -> List[Dict]:
        """Analyze logged failures to identify patterns.

        Args:
            min_cluster_size: Minimum cluster size.

        Returns:
            List of failure pattern dictionaries.
        """
        if len(self.failure_patterns) < min_cluster_size:
            return []

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return []

        # Embed failed queries
        queries = [f["query"] for f in self.failure_patterns]
        embeddings = self.embedding_model.encode(queries)

        # Simple clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size)
        labels = clustering.fit_predict(embeddings)

        # Analyze clusters
        patterns = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_failures = [
                f for f, label in zip(self.failure_patterns, labels)
                if label == cluster_id
            ]

            adapters = [f["adapter_used"] for f in cluster_failures]
            failure_types = [f["failure_type"] for f in cluster_failures]

            patterns.append({
                "cluster_id": int(cluster_id),
                "size": len(cluster_failures),
                "most_common_adapter": max(set(adapters), key=adapters.count),
                "most_common_failure_type": max(set(failure_types), key=failure_types.count),
                "avg_score": np.mean([f["score"] for f in cluster_failures]),
                "severity": "high" if len(cluster_failures) > 20 else "medium"
            })

        patterns.sort(key=lambda x: x["size"], reverse=True)
        return patterns


class UncertaintyQuantifier:
    """Advanced uncertainty quantification."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """Initialize quantifier.

        Args:
            model: Model for uncertainty estimation.
            tokenizer: Tokenizer.
            device: Target device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def monte_carlo_dropout(
        self,
        query: str,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """Estimate uncertainty using MC dropout.

        Args:
            query: Input query.
            num_samples: Number of forward passes.

        Returns:
            Uncertainty metrics.
        """
        import torch
        import torch.nn.functional as F

        self.model.train()  # Enable dropout

        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.model(**inputs)
                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                outputs.append(probs)

        probs_stack = torch.stack(outputs)

        mean_probs = probs_stack.mean(dim=0)
        variance = probs_stack.var(dim=0)

        # Predictive entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        # Mutual information
        expected_entropy = (
            -(probs_stack * torch.log(probs_stack + 1e-10)).sum(dim=-1)
        ).mean(dim=0)
        mutual_info = entropy - expected_entropy

        self.model.eval()

        return {
            "entropy": entropy.item(),
            "mutual_information": mutual_info.item(),
            "variance": variance.mean().item(),
            "confidence": float(1 - entropy.item() / np.log(mean_probs.shape[-1]))
        }

    def ensemble_uncertainty(
        self,
        query: str,
        models: List,
        tokenizers: List
    ) -> Dict[str, float]:
        """Estimate uncertainty using ensemble of models.

        Disagreement between models indicates uncertainty.

        Args:
            query: Input query.
            models: List of models.
            tokenizers: List of tokenizers.

        Returns:
            Uncertainty metrics.
        """
        import torch

        assert len(models) == len(tokenizers)

        embeddings = []

        for model, tokenizer in zip(models, tokenizers):
            inputs = tokenizer(query, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
                # Use last hidden state's CLS token
                cls_embedding = output.hidden_states[-1][:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])

        embeddings = np.array(embeddings)

        # Compute variance in embedding space
        variance = np.var(embeddings, axis=0).mean()

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 1.0

        return {
            "ensemble_variance": float(variance),
            "average_similarity": float(avg_similarity),
            "disagreement": float(1 - avg_similarity),
            "confidence": float(avg_similarity)
        }

    def temperature_scaling(
        self,
        logits: "torch.Tensor",
        temperature: float = 1.5
    ) -> "torch.Tensor":
        """Apply temperature scaling for calibration.

        Args:
            logits: Model output logits.
            temperature: Scaling temperature (>1 = softer, <1 = sharper).

        Returns:
            Calibrated probabilities.
        """
        import torch
        import torch.nn.functional as F

        scaled_logits = logits / temperature
        return F.softmax(scaled_logits, dim=-1)

    def find_optimal_temperature(
        self,
        validation_logits: List["torch.Tensor"],
        validation_labels: List[int],
        temp_range: Tuple[float, float] = (0.5, 3.0),
        num_steps: int = 50
    ) -> float:
        """Find optimal temperature for calibration using NLL.

        Args:
            validation_logits: List of logit tensors.
            validation_labels: Corresponding labels.
            temp_range: Temperature search range.
            num_steps: Number of temperature values to try.

        Returns:
            Optimal temperature.
        """
        import torch
        import torch.nn.functional as F

        temperatures = np.linspace(temp_range[0], temp_range[1], num_steps)
        best_temp = 1.0
        best_nll = float("inf")

        logits_tensor = torch.stack(validation_logits)
        labels_tensor = torch.tensor(validation_labels)

        for temp in temperatures:
            scaled_logits = logits_tensor / temp
            nll = F.cross_entropy(scaled_logits, labels_tensor).item()

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        return best_temp

    def conformal_prediction(
        self,
        query: str,
        alpha: float = 0.1,
        calibration_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Conformal prediction for uncertainty quantification.

        Provides prediction sets with guaranteed coverage.

        Args:
            query: Input query.
            alpha: Significance level (1 - coverage).
            calibration_scores: Pre-computed calibration scores.

        Returns:
            Prediction set and metrics.
        """
        import torch
        import torch.nn.functional as F

        # Get model predictions
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0]

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_probs = sorted_probs.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

        # Compute quantile threshold from calibration scores
        if calibration_scores and len(calibration_scores) > 0:
            n = len(calibration_scores)
            quantile_idx = int(np.ceil((n + 1) * (1 - alpha)))
            quantile_idx = min(quantile_idx, n) - 1
            threshold = sorted(calibration_scores, reverse=True)[quantile_idx]
        else:
            # Default threshold based on alpha
            threshold = 1 - alpha

        # Build prediction set
        cumulative_prob = 0.0
        prediction_set = []

        for i, (prob, idx) in enumerate(zip(sorted_probs, sorted_indices)):
            cumulative_prob += prob
            prediction_set.append({
                "token_id": int(idx),
                "probability": float(prob),
                "cumulative": float(cumulative_prob)
            })

            if cumulative_prob >= threshold:
                break

        return {
            "prediction_set": prediction_set,
            "set_size": len(prediction_set),
            "coverage_probability": float(cumulative_prob),
            "threshold": float(threshold),
            "alpha": alpha
        }

    def get_calibration_metrics(
        self,
        predictions: List[float],
        labels: List[int],
        num_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics (ECE, MCE, reliability diagram data).

        Args:
            predictions: Predicted probabilities.
            labels: True labels.
            num_bins: Number of bins for calibration.

        Returns:
            Calibration metrics.
        """
        predictions = np.array(predictions)
        labels = np.array(labels)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        reliability_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = predictions[in_bin].mean()
                avg_accuracy = labels[in_bin].mean()
                calibration_error = abs(avg_accuracy - avg_confidence)

                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)

                reliability_data.append({
                    "bin_center": (bin_lower + bin_upper) / 2,
                    "avg_confidence": float(avg_confidence),
                    "avg_accuracy": float(avg_accuracy),
                    "count": int(in_bin.sum())
                })

        return {
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "reliability_diagram_data": reliability_data
        }


class ImprovementSuggester:
    """Generate improvement suggestions based on failure patterns."""

    def __init__(self, evaluator: SelfEvaluator):
        """Initialize suggester.

        Args:
            evaluator: SelfEvaluator instance.
        """
        self.evaluator = evaluator

    def generate_suggestions(
        self,
        patterns: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate actionable suggestions from failure patterns.

        Args:
            patterns: Failure patterns from analyze_failure_patterns.

        Returns:
            List of suggestions.
        """
        suggestions = []

        for pattern in patterns:
            suggestion = {
                "pattern_id": pattern["cluster_id"],
                "priority": "high" if pattern["severity"] == "high" else "medium",
                "affected_adapter": pattern["most_common_adapter"],
                "failure_type": pattern["most_common_failure_type"],
                "recommended_actions": []
            }

            failure_type = pattern["most_common_failure_type"]

            if "relevance" in failure_type:
                suggestion["recommended_actions"].extend([
                    "Generate synthetic data with more relevant examples",
                    "Increase adapter rank for better capacity",
                    "Add instruction fine-tuning with diverse prompts"
                ])

            elif "coherence" in failure_type:
                suggestion["recommended_actions"].extend([
                    "Add coherence-focused examples to training data",
                    "Apply stronger language modeling loss",
                    "Increase sequence length during training"
                ])

            elif "factuality" in failure_type:
                suggestion["recommended_actions"].extend([
                    "Add fact-checking validation layer",
                    "Include more factual examples with citations",
                    "Consider retrieval-augmented generation"
                ])

            elif "safety" in failure_type:
                suggestion["recommended_actions"].extend([
                    "Add safety-focused training examples",
                    "Implement output filtering layer",
                    "Review adapter for potential harmful patterns"
                ])

            elif "high_uncertainty" in failure_type:
                suggestion["recommended_actions"].extend([
                    "Add more training examples for this query type",
                    "Consider routing to stronger adapter",
                    "Implement ensemble for uncertain cases"
                ])

            suggestions.append(suggestion)

        return suggestions
