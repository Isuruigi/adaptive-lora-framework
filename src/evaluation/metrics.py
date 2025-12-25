"""
Evaluation metrics for model assessment.

Includes standard NLP metrics and custom quality measures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class EvaluationMetrics:
    """Collection of evaluation metrics.

    Provides standard NLP metrics including BLEU, ROUGE,
    and custom quality measures.

    Example:
        >>> metrics = EvaluationMetrics()
        >>> scores = metrics.compute_all(predictions, references)
    """

    def __init__(self):
        """Initialize metrics."""
        self.smoothing = SmoothingFunction() if NLTK_AVAILABLE else None

        if ROUGE_AVAILABLE:
            self.rouge = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True
            )
        else:
            self.rouge = None

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        n_gram: int = 4
    ) -> Dict[str, float]:
        """Compute BLEU score.

        Args:
            predictions: List of predicted texts.
            references: List of reference text lists.
            n_gram: Maximum n-gram order.

        Returns:
            Dictionary with BLEU scores.
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, skipping BLEU")
            return {"bleu": 0.0}

        # Tokenize
        pred_tokens = [p.split() for p in predictions]
        ref_tokens = [[r.split() for r in refs] for refs in references]

        # Corpus BLEU
        weights = tuple([1.0 / n_gram] * n_gram)
        corpus_score = corpus_bleu(
            ref_tokens,
            pred_tokens,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )

        # Sentence-level scores
        sentence_scores = [
            sentence_bleu(
                refs,
                pred,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            for pred, refs in zip(pred_tokens, ref_tokens)
        ]

        return {
            "bleu": corpus_score,
            "bleu_mean": np.mean(sentence_scores),
            "bleu_std": np.std(sentence_scores)
        }

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            predictions: List of predicted texts.
            references: List of reference texts.

        Returns:
            Dictionary with ROUGE scores.
        """
        if not ROUGE_AVAILABLE:
            logger.warning("rouge_score not available, skipping ROUGE")
            return {}

        scores = {
            "rouge1_f": [], "rouge1_p": [], "rouge1_r": [],
            "rouge2_f": [], "rouge2_p": [], "rouge2_r": [],
            "rougeL_f": [], "rougeL_p": [], "rougeL_r": []
        }

        for pred, ref in zip(predictions, references):
            result = self.rouge.score(ref, pred)

            for metric in ["rouge1", "rouge2", "rougeL"]:
                scores[f"{metric}_f"].append(result[metric].fmeasure)
                scores[f"{metric}_p"].append(result[metric].precision)
                scores[f"{metric}_r"].append(result[metric].recall)

        return {k: np.mean(v) for k, v in scores.items()}

    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True
    ) -> float:
        """Compute exact match accuracy.

        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            normalize: Normalize texts before comparison.

        Returns:
            Exact match ratio.
        """
        if normalize:
            predictions = [p.strip().lower() for p in predictions]
            references = [r.strip().lower() for r in references]

        matches = sum(1 for p, r in zip(predictions, references) if p == r)
        return matches / len(predictions) if predictions else 0.0

    def compute_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute token-level F1 score.

        Args:
            predictions: List of predicted texts.
            references: List of reference texts.

        Returns:
            Dictionary with precision, recall, F1.
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

            if not pred_tokens or not ref_tokens:
                continue

            overlap = pred_tokens & ref_tokens
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(ref_tokens)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n = len(predictions)
        return {
            "precision": total_precision / n if n else 0.0,
            "recall": total_recall / n if n else 0.0,
            "f1": total_f1 / n if n else 0.0
        }

    def compute_perplexity(
        self,
        log_likelihoods: List[float],
        lengths: List[int]
    ) -> float:
        """Compute perplexity from log likelihoods.

        Args:
            log_likelihoods: Log likelihood for each sequence.
            lengths: Length of each sequence.

        Returns:
            Perplexity value.
        """
        total_ll = sum(log_likelihoods)
        total_length = sum(lengths)

        if total_length == 0:
            return float("inf")

        return np.exp(-total_ll / total_length)

    def compute_diversity(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """Compute diversity metrics.

        Args:
            texts: List of generated texts.

        Returns:
            Dictionary with diversity metrics.
        """
        # Unique n-grams
        all_unigrams = []
        all_bigrams = []

        for text in texts:
            tokens = text.split()
            all_unigrams.extend(tokens)
            all_bigrams.extend(zip(tokens[:-1], tokens[1:]))

        unique_unigrams = len(set(all_unigrams))
        unique_bigrams = len(set(all_bigrams))

        total_unigrams = len(all_unigrams) or 1
        total_bigrams = len(all_bigrams) or 1

        return {
            "distinct_1": unique_unigrams / total_unigrams,
            "distinct_2": unique_bigrams / total_bigrams,
            "unique_unigrams": unique_unigrams,
            "unique_bigrams": unique_bigrams
        }

    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        reference_lists: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """Compute all available metrics.

        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            reference_lists: Optional list of multiple references.

        Returns:
            Dictionary with all metrics.
        """
        results = {}

        # BLEU
        if reference_lists:
            bleu_results = self.compute_bleu(predictions, reference_lists)
        else:
            bleu_results = self.compute_bleu(
                predictions,
                [[r] for r in references]
            )
        results.update(bleu_results)

        # ROUGE
        rouge_results = self.compute_rouge(predictions, references)
        results.update(rouge_results)

        # Exact match
        results["exact_match"] = self.compute_exact_match(predictions, references)

        # F1
        f1_results = self.compute_f1(predictions, references)
        results.update(f1_results)

        # Diversity
        diversity_results = self.compute_diversity(predictions)
        results.update(diversity_results)

        return results


def compute_win_rate(
    model_a_scores: List[float],
    model_b_scores: List[float]
) -> Dict[str, float]:
    """Compute win rate between two models.

    Args:
        model_a_scores: Scores for model A.
        model_b_scores: Scores for model B.

    Returns:
        Dictionary with win rates.
    """
    assert len(model_a_scores) == len(model_b_scores)

    wins_a = sum(1 for a, b in zip(model_a_scores, model_b_scores) if a > b)
    wins_b = sum(1 for a, b in zip(model_a_scores, model_b_scores) if b > a)
    ties = sum(1 for a, b in zip(model_a_scores, model_b_scores) if a == b)

    total = len(model_a_scores)

    return {
        "model_a_win_rate": wins_a / total,
        "model_b_win_rate": wins_b / total,
        "tie_rate": ties / total
    }
