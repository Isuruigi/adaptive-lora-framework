"""
Data validation utilities.

Features:
- Schema validation with Pydantic
- Quality checks
- Format validation
- Consistency checks
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataQuality(str, Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    """Result of validation."""
    
    is_valid: bool
    quality: DataQuality
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality": self.quality.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# Pydantic schemas for data validation
class TrainingSampleSchema(BaseModel):
    """Schema for a training sample."""
    
    instruction: str = Field(..., min_length=1, max_length=10000)
    input: str = Field(default="", max_length=50000)
    output: str = Field(..., min_length=1, max_length=50000)
    
    @field_validator('instruction')
    @classmethod
    def instruction_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Instruction cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('output')
    @classmethod
    def output_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Output cannot be empty or whitespace only")
        return v.strip()


class RouterSampleSchema(BaseModel):
    """Schema for router training sample."""
    
    query: str = Field(..., min_length=1, max_length=10000)
    optimal_adapter: int = Field(..., ge=0)
    adapter_weights: Optional[List[float]] = None
    complexity: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")
    
    @field_validator('adapter_weights')
    @classmethod
    def validate_weights(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if abs(sum(v) - 1.0) > 0.01:
                raise ValueError("Adapter weights must sum to 1.0")
            if any(w < 0 for w in v):
                raise ValueError("Adapter weights must be non-negative")
        return v


class EvaluationSampleSchema(BaseModel):
    """Schema for evaluation sample."""
    
    query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    reference: Optional[str] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metrics: Optional[Dict[str, float]] = None


class Validator(ABC):
    """Abstract validator."""
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data."""
        pass


class SchemaValidator(Validator):
    """Validate data against Pydantic schema."""
    
    def __init__(self, schema: type):
        """Initialize with schema class.
        
        Args:
            schema: Pydantic model class.
        """
        self.schema = schema
        
    def validate(self, data: Union[Dict, List[Dict]]) -> ValidationResult:
        """Validate data against schema.
        
        Args:
            data: Single item or list of items.
            
        Returns:
            ValidationResult.
        """
        errors = []
        warnings = []
        valid_count = 0
        invalid_count = 0
        
        items = data if isinstance(data, list) else [data]
        
        for i, item in enumerate(items):
            try:
                self.schema(**item)
                valid_count += 1
            except Exception as e:
                invalid_count += 1
                errors.append(f"Item {i}: {str(e)}")
                
        is_valid = invalid_count == 0
        
        if invalid_count > 0:
            quality = DataQuality.INVALID if invalid_count == len(items) else DataQuality.LOW
        else:
            quality = DataQuality.HIGH
            
        return ValidationResult(
            is_valid=is_valid,
            quality=quality,
            errors=errors,
            warnings=warnings,
            stats={"valid": valid_count, "invalid": invalid_count}
        )


class TextQualityValidator(Validator):
    """Validate text quality."""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 50000,
        check_encoding: bool = True,
        check_repetition: bool = True,
        repetition_threshold: float = 0.3
    ):
        """Initialize validator.
        
        Args:
            min_length: Minimum text length.
            max_length: Maximum text length.
            check_encoding: Check for encoding issues.
            check_repetition: Check for repetitive content.
            repetition_threshold: Threshold for repetition detection.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.check_encoding = check_encoding
        self.check_repetition = check_repetition
        self.repetition_threshold = repetition_threshold
        
    def validate(self, text: str) -> ValidationResult:
        """Validate text quality.
        
        Args:
            text: Text to validate.
            
        Returns:
            ValidationResult.
        """
        errors = []
        warnings = []
        
        # Length checks
        if len(text) < self.min_length:
            errors.append(f"Text too short: {len(text)} < {self.min_length}")
        if len(text) > self.max_length:
            errors.append(f"Text too long: {len(text)} > {self.max_length}")
            
        # Encoding check
        if self.check_encoding:
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                errors.append("Text has encoding issues")
                
            # Check for replacement characters
            if '\ufffd' in text:
                warnings.append("Text contains replacement characters")
                
        # Repetition check
        if self.check_repetition:
            repetition_ratio = self._compute_repetition(text)
            if repetition_ratio > self.repetition_threshold:
                warnings.append(f"High repetition ratio: {repetition_ratio:.2f}")
                
        # Quality determination
        if errors:
            quality = DataQuality.INVALID
        elif warnings:
            quality = DataQuality.MEDIUM
        else:
            quality = DataQuality.HIGH
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=quality,
            errors=errors,
            warnings=warnings,
            stats={"length": len(text)}
        )
    
    def _compute_repetition(self, text: str) -> float:
        """Compute repetition ratio using n-grams."""
        words = text.lower().split()
        if len(words) < 5:
            return 0.0
            
        # Use 3-grams
        ngrams = []
        for i in range(len(words) - 2):
            ngrams.append(tuple(words[i:i+3]))
            
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        return 1.0 - unique_ratio


class DatasetValidator(Validator):
    """Validate entire datasets."""
    
    def __init__(
        self,
        sample_schema: type,
        min_samples: int = 10,
        max_duplicates_ratio: float = 0.05,
        required_fields: Optional[List[str]] = None
    ):
        """Initialize validator.
        
        Args:
            sample_schema: Pydantic schema for samples.
            min_samples: Minimum number of samples.
            max_duplicates_ratio: Maximum allowed duplicate ratio.
            required_fields: Required fields in each sample.
        """
        self.sample_schema = sample_schema
        self.min_samples = min_samples
        self.max_duplicates_ratio = max_duplicates_ratio
        self.required_fields = required_fields or []
        
        self.schema_validator = SchemaValidator(sample_schema)
        self.text_validator = TextQualityValidator()
        
    def validate(self, dataset: List[Dict]) -> ValidationResult:
        """Validate dataset.
        
        Args:
            dataset: List of samples.
            
        Returns:
            ValidationResult with comprehensive checks.
        """
        errors = []
        warnings = []
        stats = {
            "total_samples": len(dataset),
            "valid_samples": 0,
            "invalid_samples": 0,
            "duplicates": 0,
        }
        
        # Size check
        if len(dataset) < self.min_samples:
            errors.append(f"Dataset too small: {len(dataset)} < {self.min_samples}")
            
        # Schema validation
        schema_result = self.schema_validator.validate(dataset)
        stats["valid_samples"] = schema_result.stats.get("valid", 0)
        stats["invalid_samples"] = schema_result.stats.get("invalid", 0)
        errors.extend(schema_result.errors[:10])  # Limit error messages
        
        # Duplicate check
        seen_hashes: Set[int] = set()
        duplicates = 0
        
        for sample in dataset:
            sample_hash = hash(str(sorted(sample.items())))
            if sample_hash in seen_hashes:
                duplicates += 1
            seen_hashes.add(sample_hash)
            
        stats["duplicates"] = duplicates
        duplicate_ratio = duplicates / len(dataset) if dataset else 0
        
        if duplicate_ratio > self.max_duplicates_ratio:
            warnings.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
            
        # Field completeness
        for field in self.required_fields:
            missing = sum(1 for s in dataset if field not in s or not s[field])
            if missing > 0:
                warnings.append(f"Field '{field}' missing in {missing} samples")
                
        # Quality determination
        if errors:
            quality = DataQuality.INVALID if stats["invalid_samples"] > len(dataset) * 0.5 else DataQuality.LOW
        elif warnings:
            quality = DataQuality.MEDIUM
        else:
            quality = DataQuality.HIGH
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=quality,
            errors=errors,
            warnings=warnings,
            stats=stats
        )


class ContentFilter:
    """Filter content based on various criteria."""
    
    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        max_special_char_ratio: float = 0.3,
        min_word_count: int = 3
    ):
        """Initialize filter.
        
        Args:
            blocked_patterns: Regex patterns to block.
            max_special_char_ratio: Maximum ratio of special characters.
            min_word_count: Minimum word count.
        """
        self.blocked_patterns = [re.compile(p) for p in (blocked_patterns or [])]
        self.max_special_char_ratio = max_special_char_ratio
        self.min_word_count = min_word_count
        
    def is_valid(self, text: str) -> bool:
        """Check if text passes filters."""
        # Word count
        if len(text.split()) < self.min_word_count:
            return False
            
        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0 and special_chars / len(text) > self.max_special_char_ratio:
            return False
            
        # Blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False
                
        return True
    
    def filter_dataset(self, dataset: List[Dict], text_field: str = "instruction") -> List[Dict]:
        """Filter dataset.
        
        Args:
            dataset: Dataset to filter.
            text_field: Field containing text to check.
            
        Returns:
            Filtered dataset.
        """
        filtered = []
        
        for sample in dataset:
            text = sample.get(text_field, "")
            if self.is_valid(text):
                filtered.append(sample)
                
        logger.info(f"Filtered {len(dataset)} -> {len(filtered)} samples")
        return filtered


def validate_training_data(
    data: List[Dict],
    schema: type = TrainingSampleSchema
) -> ValidationResult:
    """Convenience function to validate training data.
    
    Args:
        data: Training data to validate.
        schema: Schema to validate against.
        
    Returns:
        ValidationResult.
    """
    validator = DatasetValidator(
        sample_schema=schema,
        min_samples=1,
        required_fields=["instruction", "output"]
    )
    return validator.validate(data)
