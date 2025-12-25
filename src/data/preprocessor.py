"""
Data preprocessing utilities for training data preparation.

Features:
- Text cleaning and normalization
- Tokenization utilities
- Data augmentation
- Quality filtering
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessing for training data.

    Provides text cleaning, normalization, and quality filtering
    utilities for preparing training data.

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> cleaned = preprocessor.clean_text("Hello  world!")
        >>> filtered = preprocessor.filter_examples(data, min_length=10)
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        max_length: Optional[int] = None,
        min_length: int = 10
    ):
        """Initialize preprocessor.

        Args:
            remove_urls: Remove URLs from text.
            remove_emails: Remove email addresses from text.
            normalize_whitespace: Normalize whitespace characters.
            max_length: Maximum text length (characters).
            min_length: Minimum text length (characters).
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.max_length = max_length
        self.min_length = min_length

        # Compile regex patterns
        self.url_pattern = re.compile(
            r"https?://\S+|www\.\S+"
        )
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub("", text)

        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub("", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(" ", text)
            text = text.strip()

        # Truncate if needed
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def clean_example(
        self,
        example: Dict[str, Any],
        text_fields: List[str] = ["instruction", "input", "output"]
    ) -> Dict[str, Any]:
        """Clean all text fields in an example.

        Args:
            example: Data example dictionary.
            text_fields: Fields to clean.

        Returns:
            Example with cleaned fields.
        """
        cleaned = example.copy()

        for field in text_fields:
            if field in cleaned and isinstance(cleaned[field], str):
                cleaned[field] = self.clean_text(cleaned[field])

        return cleaned

    def filter_examples(
        self,
        examples: List[Dict[str, Any]],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
        custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Filter examples based on criteria.

        Args:
            examples: List of examples.
            min_length: Minimum output length.
            max_length: Maximum output length.
            required_fields: Required non-empty fields.
            custom_filter: Custom filter function.

        Returns:
            Filtered list of examples.
        """
        min_length = min_length or self.min_length
        max_length = max_length or self.max_length
        required_fields = required_fields or ["instruction", "output"]

        filtered = []

        for example in examples:
            # Check required fields
            if not all(example.get(f) for f in required_fields):
                continue

            # Check output length
            output = example.get("output", "")
            if len(output) < min_length:
                continue
            if max_length and len(output) > max_length:
                continue

            # Custom filter
            if custom_filter and not custom_filter(example):
                continue

            filtered.append(example)

        logger.info(f"Filtered {len(examples)} -> {len(filtered)} examples")
        return filtered

    def deduplicate(
        self,
        examples: List[Dict[str, Any]],
        key_fields: List[str] = ["instruction", "output"]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate examples.

        Args:
            examples: List of examples.
            key_fields: Fields to use for deduplication.

        Returns:
            Deduplicated list.
        """
        seen: Set[str] = set()
        deduplicated = []

        for example in examples:
            # Create key from specified fields
            key_parts = [str(example.get(f, "")) for f in key_fields]
            key = "|||".join(key_parts)

            if key not in seen:
                seen.add(key)
                deduplicated.append(example)

        logger.info(
            f"Deduplicated {len(examples)} -> {len(deduplicated)} examples"
        )
        return deduplicated

    def augment_text(
        self,
        text: str,
        methods: List[str] = ["synonym", "paraphrase"]
    ) -> List[str]:
        """Augment text using various methods.

        Args:
            text: Input text.
            methods: Augmentation methods to apply.

        Returns:
            List of augmented texts.
        """
        augmented = [text]

        for method in methods:
            if method == "synonym":
                # Simple word variation (placeholder for more sophisticated NLP)
                variations = self._synonym_augment(text)
                augmented.extend(variations)
            elif method == "case":
                augmented.append(text.lower())
                augmented.append(text.upper())
                augmented.append(text.title())

        return list(set(augmented))

    def _synonym_augment(self, text: str) -> List[str]:
        """Simple synonym-based augmentation.

        Args:
            text: Input text.

        Returns:
            List of augmented texts.
        """
        # Simple replacements (placeholder for NLTK/WordNet)
        synonyms = {
            "important": ["significant", "crucial", "essential"],
            "good": ["excellent", "great", "positive"],
            "bad": ["poor", "negative", "inferior"],
            "use": ["utilize", "employ", "apply"],
        }

        augmented = []
        words = text.split()

        for word in words:
            word_lower = word.lower()
            if word_lower in synonyms:
                for syn in synonyms[word_lower]:
                    new_text = text.replace(word, syn, 1)
                    augmented.append(new_text)

        return augmented[:3]  # Limit augmentations

    def balance_dataset(
        self,
        examples: List[Dict[str, Any]],
        category_field: str,
        target_per_category: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Balance dataset across categories.

        Args:
            examples: List of examples.
            category_field: Field containing category.
            target_per_category: Target examples per category.

        Returns:
            Balanced list of examples.
        """
        # Group by category
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for example in examples:
            cat = example.get(category_field, "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(example)

        # Determine target count
        if target_per_category is None:
            target_per_category = min(len(v) for v in categories.values())

        # Balance
        balanced = []
        for cat, cat_examples in categories.items():
            if len(cat_examples) >= target_per_category:
                balanced.extend(cat_examples[:target_per_category])
            else:
                # Oversample if needed
                balanced.extend(cat_examples)
                needed = target_per_category - len(cat_examples)
                import random
                balanced.extend(random.choices(cat_examples, k=needed))

        logger.info(
            f"Balanced dataset to {target_per_category} per category "
            f"({len(categories)} categories)"
        )

        return balanced

    def process_batch(
        self,
        examples: List[Dict[str, Any]],
        clean: bool = True,
        filter_data: bool = True,
        dedupe: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a batch of examples through full pipeline.

        Args:
            examples: List of examples.
            clean: Apply cleaning.
            filter_data: Apply filtering.
            dedupe: Apply deduplication.

        Returns:
            Processed examples.
        """
        result = examples

        if clean:
            result = [self.clean_example(ex) for ex in result]

        if filter_data:
            result = self.filter_examples(result)

        if dedupe:
            result = self.deduplicate(result)

        return result
