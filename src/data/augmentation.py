"""
Data augmentation utilities for training data.

Features:
- Text augmentation strategies
- Semantic-preserving transformations
- Back-translation (placeholder)
- Synonym replacement
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AugmentedSample:
    """Augmented training sample."""
    
    original: Dict[str, Any]
    augmented: Dict[str, Any]
    augmentation_type: str
    confidence: float = 1.0


class AugmentationStrategy(ABC):
    """Abstract augmentation strategy."""
    
    @abstractmethod
    def augment(self, text: str, **kwargs) -> str:
        """Augment text.
        
        Args:
            text: Input text.
            **kwargs: Additional arguments.
            
        Returns:
            Augmented text.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class SynonymReplacement(AugmentationStrategy):
    """Replace words with synonyms."""
    
    def __init__(self, replacement_ratio: float = 0.1):
        """Initialize.
        
        Args:
            replacement_ratio: Ratio of words to replace.
        """
        self.replacement_ratio = replacement_ratio
        
        # Simple synonym dictionary (extend as needed)
        self.synonyms = {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "negative"],
            "big": ["large", "huge", "massive", "enormous"],
            "small": ["tiny", "little", "minor", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["gradual", "unhurried", "leisurely"],
            "important": ["crucial", "significant", "essential", "vital"],
            "help": ["assist", "aid", "support", "facilitate"],
            "create": ["make", "build", "develop", "produce"],
            "show": ["display", "demonstrate", "present", "exhibit"],
            "use": ["utilize", "employ", "apply"],
            "get": ["obtain", "acquire", "receive", "gain"],
            "make": ["create", "produce", "build", "construct"],
            "like": ["similar to", "such as", "resembling"],
        }
        
    @property
    def name(self) -> str:
        return "synonym_replacement"
    
    def augment(self, text: str, **kwargs) -> str:
        """Replace random words with synonyms."""
        words = text.split()
        num_to_replace = max(1, int(len(words) * self.replacement_ratio))
        
        # Find replaceable words
        replaceable_indices = []
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in self.synonyms:
                replaceable_indices.append(i)
                
        # Replace random subset
        if replaceable_indices:
            indices_to_replace = random.sample(
                replaceable_indices,
                min(num_to_replace, len(replaceable_indices))
            )
            
            for idx in indices_to_replace:
                word = words[idx]
                clean_word = word.lower().strip('.,!?;:')
                
                if clean_word in self.synonyms:
                    synonym = random.choice(self.synonyms[clean_word])
                    
                    # Preserve capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                        
                    # Preserve punctuation
                    if word[-1] in '.,!?;:':
                        synonym += word[-1]
                        
                    words[idx] = synonym
                    
        return ' '.join(words)


class RandomInsertion(AugmentationStrategy):
    """Insert random words."""
    
    def __init__(self, insertion_ratio: float = 0.1):
        """Initialize.
        
        Args:
            insertion_ratio: Ratio of insertions relative to text length.
        """
        self.insertion_ratio = insertion_ratio
        
        # Words that can be safely inserted
        self.insertable_words = [
            "also", "indeed", "actually", "certainly",
            "basically", "essentially", "generally",
            "particularly", "specifically", "notably",
        ]
        
    @property
    def name(self) -> str:
        return "random_insertion"
    
    def augment(self, text: str, **kwargs) -> str:
        """Insert random words at random positions."""
        words = text.split()
        num_insertions = max(1, int(len(words) * self.insertion_ratio))
        
        for _ in range(num_insertions):
            insert_word = random.choice(self.insertable_words)
            insert_pos = random.randint(1, len(words))
            words.insert(insert_pos, insert_word)
            
        return ' '.join(words)


class RandomDeletion(AugmentationStrategy):
    """Randomly delete words."""
    
    def __init__(self, deletion_prob: float = 0.1):
        """Initialize.
        
        Args:
            deletion_prob: Probability of deleting each word.
        """
        self.deletion_prob = deletion_prob
        
        # Words to never delete
        self.protected_words = {'is', 'are', 'was', 'were', 'not', 'no', 'yes'}
        
    @property
    def name(self) -> str:
        return "random_deletion"
    
    def augment(self, text: str, **kwargs) -> str:
        """Randomly delete words."""
        words = text.split()
        
        if len(words) <= 3:
            return text
            
        new_words = []
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            
            if clean_word in self.protected_words:
                new_words.append(word)
            elif random.random() > self.deletion_prob:
                new_words.append(word)
                
        # Ensure we keep at least 50% of words
        if len(new_words) < len(words) * 0.5:
            return text
            
        return ' '.join(new_words)


class RandomSwap(AugmentationStrategy):
    """Randomly swap adjacent words."""
    
    def __init__(self, swap_ratio: float = 0.1):
        """Initialize.
        
        Args:
            swap_ratio: Ratio of swaps relative to text length.
        """
        self.swap_ratio = swap_ratio
        
    @property
    def name(self) -> str:
        return "random_swap"
    
    def augment(self, text: str, **kwargs) -> str:
        """Swap random adjacent word pairs."""
        words = text.split()
        
        if len(words) < 2:
            return text
            
        num_swaps = max(1, int(len(words) * self.swap_ratio))
        
        for _ in range(num_swaps):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            
        return ' '.join(words)


class CaseChange(AugmentationStrategy):
    """Change case of text."""
    
    def __init__(self, mode: str = "random"):
        """Initialize.
        
        Args:
            mode: One of 'lower', 'upper', 'title', 'random'.
        """
        self.mode = mode
        
    @property
    def name(self) -> str:
        return f"case_change_{self.mode}"
    
    def augment(self, text: str, **kwargs) -> str:
        """Change text case."""
        if self.mode == "lower":
            return text.lower()
        elif self.mode == "upper":
            return text.upper()
        elif self.mode == "title":
            return text.title()
        else:  # random
            mode = random.choice(["lower", "title"])
            return text.lower() if mode == "lower" else text.title()


class BackTranslation(AugmentationStrategy):
    """Back-translation augmentation (placeholder for API integration)."""
    
    def __init__(
        self,
        intermediate_lang: str = "de",
        translator=None
    ):
        """Initialize.
        
        Args:
            intermediate_lang: Intermediate language for translation.
            translator: Translation API client.
        """
        self.intermediate_lang = intermediate_lang
        self.translator = translator
        
    @property
    def name(self) -> str:
        return f"back_translation_{self.intermediate_lang}"
    
    def augment(self, text: str, **kwargs) -> str:
        """Perform back-translation.
        
        Note: Requires translation API integration.
        """
        if self.translator is None:
            # Placeholder: return original text
            logger.warning("No translator configured, returning original text")
            return text
            
        try:
            # Translate to intermediate language
            intermediate = self.translator.translate(text, dest=self.intermediate_lang)
            # Translate back to English
            back = self.translator.translate(intermediate, dest="en")
            return back
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text


class DataAugmenter:
    """Apply multiple augmentation strategies to data."""
    
    def __init__(
        self,
        strategies: Optional[List[AugmentationStrategy]] = None,
        augmentation_factor: int = 2
    ):
        """Initialize augmenter.
        
        Args:
            strategies: List of strategies to use.
            augmentation_factor: How many augmented versions per original.
        """
        self.strategies = strategies or [
            SynonymReplacement(),
            RandomInsertion(),
            RandomDeletion(),
            RandomSwap(),
        ]
        self.augmentation_factor = augmentation_factor
        
    def augment_text(
        self,
        text: str,
        num_augments: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """Augment a single text.
        
        Args:
            text: Text to augment.
            num_augments: Number of augmented versions.
            
        Returns:
            List of (augmented_text, strategy_name) tuples.
        """
        num_augments = num_augments or self.augmentation_factor
        results = []
        
        for _ in range(num_augments):
            strategy = random.choice(self.strategies)
            augmented = strategy.augment(text)
            results.append((augmented, strategy.name))
            
        return results
    
    def augment_sample(
        self,
        sample: Dict[str, Any],
        text_fields: List[str] = ["instruction"],
        num_augments: Optional[int] = None
    ) -> List[AugmentedSample]:
        """Augment a training sample.
        
        Args:
            sample: Original sample.
            text_fields: Fields to augment.
            num_augments: Number of augmented versions.
            
        Returns:
            List of AugmentedSample objects.
        """
        num_augments = num_augments or self.augmentation_factor
        results = []
        
        for _ in range(num_augments):
            strategy = random.choice(self.strategies)
            augmented_sample = sample.copy()
            
            for field in text_fields:
                if field in augmented_sample:
                    augmented_sample[field] = strategy.augment(augmented_sample[field])
                    
            results.append(AugmentedSample(
                original=sample,
                augmented=augmented_sample,
                augmentation_type=strategy.name,
                confidence=0.9  # Slightly lower confidence for augmented data
            ))
            
        return results
    
    def augment_dataset(
        self,
        dataset: List[Dict[str, Any]],
        text_fields: List[str] = ["instruction"],
        include_original: bool = True
    ) -> List[Dict[str, Any]]:
        """Augment entire dataset.
        
        Args:
            dataset: Original dataset.
            text_fields: Fields to augment.
            include_original: Whether to include original samples.
            
        Returns:
            Augmented dataset.
        """
        augmented_dataset = []
        
        if include_original:
            augmented_dataset.extend(dataset)
            
        for sample in dataset:
            augmented_samples = self.augment_sample(sample, text_fields)
            for aug in augmented_samples:
                augmented_dataset.append(aug.augmented)
                
        logger.info(
            f"Augmented dataset: {len(dataset)} -> {len(augmented_dataset)} samples"
        )
        
        return augmented_dataset


class InstructionAugmenter:
    """Specialized augmenter for instruction-tuning data."""
    
    def __init__(self):
        """Initialize with instruction-specific strategies."""
        self.rephrasing_templates = [
            "Please {action}",
            "Could you {action}",
            "I need you to {action}",
            "Can you {action}",
            "{action}",
            "I'd like you to {action}",
            "Would you {action}",
        ]
        
    def rephrase_instruction(self, instruction: str) -> str:
        """Rephrase an instruction.
        
        Args:
            instruction: Original instruction.
            
        Returns:
            Rephrased instruction.
        """
        # Simple template-based rephrasing
        template = random.choice(self.rephrasing_templates)
        
        # Extract action from original instruction
        action = instruction.lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "please ", "could you ", "can you ", "i need you to ",
            "i'd like you to ", "would you "
        ]
        
        for prefix in prefixes_to_remove:
            if action.startswith(prefix):
                action = action[len(prefix):]
                break
                
        return template.format(action=action)
    
    def augment_instruction_pair(
        self,
        sample: Dict[str, Any],
        num_versions: int = 2
    ) -> List[Dict[str, Any]]:
        """Augment instruction-output pair.
        
        Args:
            sample: Sample with 'instruction' and 'output' fields.
            num_versions: Number of augmented versions.
            
        Returns:
            List of augmented samples.
        """
        results = []
        
        for _ in range(num_versions):
            new_sample = sample.copy()
            new_sample["instruction"] = self.rephrase_instruction(sample["instruction"])
            results.append(new_sample)
            
        return results
