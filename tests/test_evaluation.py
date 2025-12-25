"""
Tests for evaluation module.
"""

import pytest

from src.evaluation.self_evaluator import SelfEvaluator, EvaluationResult
from src.evaluation.metrics import EvaluationMetrics, compute_win_rate
from src.evaluation.quality_scorer import (
    QualityScorer,
    QualityDimension,
    length_scorer,
    specificity_scorer,
    structure_scorer,
    create_default_scorer,
)


class TestSelfEvaluator:
    """Tests for SelfEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM."""
        return SelfEvaluator(use_llm_judge=False)

    def test_check_coherence(self, evaluator):
        """Test coherence checking."""
        query = "Explain machine learning"
        good_response = (
            "Machine learning is a branch of artificial intelligence "
            "that focuses on building systems that learn from data."
        )
        
        score = evaluator.check_coherence(query, good_response)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Good response should score well

    def test_check_coherence_poor_response(self, evaluator):
        """Test coherence with poor response."""
        query = "Explain machine learning"
        poor_response = "Sorry, I cannot help with that."
        
        score = evaluator.check_coherence(query, poor_response)
        
        assert score <= 0.7  # Should score lower or at boundary

    def test_check_safety(self, evaluator):
        """Test safety checking."""
        safe_response = "Machine learning uses algorithms to find patterns."
        unsafe_response = "I hate this stupid question about killing people."
        
        safe_score = evaluator.check_safety(safe_response)
        unsafe_score = evaluator.check_safety(unsafe_response)
        
        assert safe_score > unsafe_score
        assert safe_score > 0.9

    def test_estimate_uncertainty(self, evaluator):
        """Test uncertainty estimation."""
        confident_response = (
            "Machine learning is a technique where computers learn patterns "
            "from data to make predictions or decisions."
        )
        uncertain_response = "Maybe? I'm not sure. Could be this or that?"
        
        confident_uncertainty = evaluator.estimate_uncertainty("", confident_response)
        uncertain_uncertainty = evaluator.estimate_uncertainty("", uncertain_response)
        
        assert confident_uncertainty < uncertain_uncertainty

    def test_full_evaluation(self, evaluator):
        """Test full evaluation pipeline."""
        query = "What is Python programming?"
        response = (
            "Python is a high-level, interpreted programming language "
            "known for its clear syntax and readability. It was created "
            "by Guido van Rossum in 1991."
        )
        
        result = evaluator.evaluate(
            query=query,
            response=response,
            adapter_used="reasoning"
        )
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.overall_score <= 1
        assert result.adapter_used == "reasoning"
        assert isinstance(result.is_failure, bool)

    def test_failure_detection(self, evaluator):
        """Test failure detection."""
        evaluator.failure_threshold = 0.9  # High threshold
        
        result = evaluator.evaluate(
            query="Complex question",
            response="I don't know.",
            adapter_used="test"
        )
        
        # Should be marked as failure due to poor response
        assert result.is_failure or result.overall_score < 0.9


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return EvaluationMetrics()

    def test_compute_exact_match(self, metrics):
        """Test exact match computation."""
        predictions = ["hello world", "foo bar", "test"]
        references = ["hello world", "Foo Bar", "different"]
        
        em = metrics.compute_exact_match(predictions, references, normalize=True)
        
        # "hello world" and "foo bar" match after normalization
        assert em == pytest.approx(2/3, rel=1e-4)

    def test_compute_f1(self, metrics):
        """Test F1 score computation."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on a mat"]
        
        result = metrics.compute_f1(predictions, references)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert 0 <= result["f1"] <= 1

    def test_compute_diversity(self, metrics):
        """Test diversity metrics."""
        texts = [
            "The quick brown fox jumps",
            "A quick brown dog runs",
            "The fast red fox leaps"
        ]
        
        result = metrics.compute_diversity(texts)
        
        assert "distinct_1" in result
        assert "distinct_2" in result
        assert result["distinct_1"] > 0
        assert result["distinct_2"] > 0


class TestComputeWinRate:
    """Tests for win rate computation."""

    def test_basic_win_rate(self):
        """Test basic win rate calculation."""
        model_a_scores = [0.8, 0.6, 0.9, 0.5]
        model_b_scores = [0.7, 0.7, 0.8, 0.5]
        
        result = compute_win_rate(model_a_scores, model_b_scores)
        
        assert "model_a_win_rate" in result
        assert "model_b_win_rate" in result
        assert "tie_rate" in result
        
        # Sum should be 1
        total = result["model_a_win_rate"] + result["model_b_win_rate"] + result["tie_rate"]
        assert total == pytest.approx(1.0, rel=1e-4)


class TestQualityScorer:
    """Tests for QualityScorer."""

    def test_add_dimension(self):
        """Test adding scoring dimensions."""
        scorer = QualityScorer()
        
        scorer.add_dimension("test", 0.5, lambda q, r: 0.8)
        
        assert len(scorer.dimensions) == 1
        assert scorer.dimensions[0].name == "test"

    def test_score_response(self):
        """Test scoring a response."""
        scorer = QualityScorer()
        scorer.add_dimension("length", 1.0, length_scorer)
        
        result = scorer.score(
            "What is AI?",
            "Artificial intelligence is the simulation of human intelligence.",
            return_breakdown=True
        )
        
        assert "overall_score" in result
        assert "dimension_scores" in result
        assert 0 <= result["overall_score"] <= 1

    def test_score_batch(self):
        """Test batch scoring."""
        scorer = create_default_scorer()
        
        queries = ["Query 1", "Query 2"]
        responses = ["Response one with details.", "Another response here."]
        
        results = scorer.score_batch(queries, responses)
        
        assert len(results) == 2
        assert all("overall_score" in r for r in results)

    def test_aggregate_scores(self):
        """Test score aggregation."""
        scorer = create_default_scorer()
        
        scores = [
            scorer.score("Q1", "R1 with some content here.", return_breakdown=True),
            scorer.score("Q2", "R2 with different content here.", return_breakdown=True),
        ]
        
        aggregated = scorer.aggregate_scores(scores)
        
        assert "mean_score" in aggregated
        assert "std_score" in aggregated
        assert "pass_rate" in aggregated


class TestIndividualScorers:
    """Tests for individual scorer functions."""

    def test_length_scorer(self):
        """Test length scorer."""
        short_score = length_scorer("Short q", "Hi")
        good_score = length_scorer("Short q", "A good response with reasonable length.")
        
        assert good_score > short_score

    def test_specificity_scorer(self):
        """Test specificity scorer."""
        vague = specificity_scorer("Q", "Something happened.")
        specific = specificity_scorer("Q", "In 2023, the Python algorithm achieved 95% accuracy.")
        
        assert specific > vague

    def test_structure_scorer(self):
        """Test structure scorer."""
        unstructured = structure_scorer("Q", "One sentence")
        structured = structure_scorer(
            "Q",
            "First point. Second point.\n\nAnother paragraph with details."
        )
        
        assert structured > unstructured
