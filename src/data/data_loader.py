"""
Flexible data loading system supporting multiple formats.

Features:
- Multiple data formats (JSON, JSONL, CSV, Parquet)
- Hugging Face datasets integration
- Custom dataset classes
- Streaming for large datasets
- Data validation and cleaning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveDataLoader:
    """Flexible data loader supporting multiple formats.

    Supports loading from JSON, JSONL, CSV, Parquet files and
    Hugging Face Hub datasets with automatic format detection.

    Example:
        >>> loader = AdaptiveDataLoader(tokenizer, max_length=2048)
        >>> dataset = loader.load("data/train.jsonl")
        >>> processed = dataset.map(loader.preprocess_function, batched=True)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        validation_split: float = 0.1,
        seed: int = 42
    ):
        """Initialize data loader.

        Args:
            tokenizer: HuggingFace tokenizer for text processing.
            max_length: Maximum sequence length.
            validation_split: Fraction of data to use for validation.
            seed: Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_split = validation_split
        self.seed = seed

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def load(
        self,
        data_path: Union[str, Path],
        format: str = "auto",
        streaming: bool = False,
        split: Optional[str] = None
    ) -> DatasetDict:
        """Load dataset from various formats.

        Args:
            data_path: Path to data file, directory, or HuggingFace dataset ID.
            format: Data format (json, jsonl, csv, parquet, hf, auto).
            streaming: Use streaming mode for large datasets.
            split: Specific split to load (train, validation, test).

        Returns:
            DatasetDict with train/validation splits.

        Raises:
            ValueError: If format is not supported.
            FileNotFoundError: If data file doesn't exist.
        """
        data_path = Path(data_path) if not str(data_path).startswith("hf://") else data_path

        if format == "auto":
            format = self._detect_format(data_path)

        logger.info(f"Loading data from {data_path} (format: {format})")

        if format == "hf":
            dataset = self._load_from_hf(str(data_path), streaming, split)
        elif format == "jsonl":
            dataset = self._load_jsonl(data_path)
        elif format == "json":
            dataset = self._load_json(data_path)
        elif format == "csv":
            dataset = self._load_csv(data_path)
        elif format == "parquet":
            dataset = self._load_parquet(data_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Split into train/val if not already split
        if isinstance(dataset, Dataset):
            dataset = self._create_splits(dataset)

        logger.info(
            f"Loaded dataset with {len(dataset['train'])} train "
            f"and {len(dataset['validation'])} validation examples"
        )

        return dataset

    def _detect_format(self, path: Union[str, Path]) -> str:
        """Detect data format from file extension or path.

        Args:
            path: Data path.

        Returns:
            Detected format string.
        """
        if isinstance(path, str):
            if path.startswith("hf://") or "/" in path and not Path(path).exists():
                return "hf"
            path = Path(path)

        if not path.exists():
            # Might be a HuggingFace dataset ID
            return "hf"

        if path.is_dir():
            # Check for common file patterns
            if list(path.glob("*.jsonl")):
                return "jsonl"
            elif list(path.glob("*.json")):
                return "json"
            elif list(path.glob("*.parquet")):
                return "parquet"
            elif list(path.glob("*.csv")):
                return "csv"

        suffix = path.suffix.lower()
        format_map = {
            ".jsonl": "jsonl",
            ".json": "json",
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }

        return format_map.get(suffix, "json")

    def _load_from_hf(
        self,
        dataset_id: str,
        streaming: bool,
        split: Optional[str]
    ) -> DatasetDict:
        """Load from Hugging Face Hub.

        Args:
            dataset_id: HuggingFace dataset identifier.
            streaming: Use streaming mode.
            split: Specific split to load.

        Returns:
            DatasetDict with requested splits.
        """
        dataset_id = dataset_id.replace("hf://", "")

        kwargs = {"streaming": streaming}
        if split:
            kwargs["split"] = split

        dataset = load_dataset(dataset_id, **kwargs)

        if isinstance(dataset, Dataset):
            return self._create_splits(dataset)

        return dataset

    def _load_jsonl(self, path: Path) -> Dataset:
        """Load JSONL file.

        Args:
            path: Path to JSONL file.

        Returns:
            Dataset object.
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return Dataset.from_list(data)

    def _load_json(self, path: Path) -> Dataset:
        """Load JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Dataset object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return Dataset.from_dict(data)
        return Dataset.from_list(data)

    def _load_csv(self, path: Path) -> Dataset:
        """Load CSV file.

        Args:
            path: Path to CSV file.

        Returns:
            Dataset object.
        """
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)

    def _load_parquet(self, path: Path) -> Dataset:
        """Load Parquet file.

        Args:
            path: Path to Parquet file.

        Returns:
            Dataset object.
        """
        df = pd.read_parquet(path)
        return Dataset.from_pandas(df)

    def _create_splits(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation splits.

        Args:
            dataset: Dataset to split.

        Returns:
            DatasetDict with train and validation splits.
        """
        split_dataset = dataset.train_test_split(
            test_size=self.validation_split,
            seed=self.seed
        )

        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })

    def preprocess_function(
        self,
        examples: Dict[str, List],
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        template: str = "alpaca"
    ) -> Dict[str, List]:
        """Preprocess examples for training.

        Args:
            examples: Batch of examples.
            instruction_key: Key for instruction field.
            input_key: Key for input field.
            output_key: Key for output field.
            template: Prompt template (alpaca, sharegpt, chatml).

        Returns:
            Tokenized examples with input_ids, attention_mask, labels.
        """
        prompts = []
        batch_size = len(examples[instruction_key])

        for i in range(batch_size):
            instruction = examples[instruction_key][i]
            input_text = examples.get(input_key, [""] * batch_size)[i] or ""
            output = examples[output_key][i]

            # Format prompt based on template
            if template == "alpaca":
                prompt = self._format_alpaca(instruction, input_text, output)
            elif template == "chatml":
                prompt = self._format_chatml(instruction, input_text, output)
            else:
                prompt = f"{instruction}\n\n{input_text}\n\nResponse: {output}"

            prompts.append(prompt)

        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def _format_alpaca(
        self,
        instruction: str,
        input_text: str,
        output: str
    ) -> str:
        """Format data in Alpaca style.

        Args:
            instruction: Task instruction.
            input_text: Additional input context.
            output: Expected response.

        Returns:
            Formatted prompt string.
        """
        if input_text:
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            return (
                "Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )

    def _format_chatml(
        self,
        instruction: str,
        input_text: str,
        output: str
    ) -> str:
        """Format data in ChatML style.

        Args:
            instruction: Task instruction.
            input_text: Additional input context.
            output: Expected response.

        Returns:
            Formatted prompt string.
        """
        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )


class InstructionDataset(TorchDataset):
    """PyTorch Dataset for instruction-following data.

    Provides a PyTorch-compatible dataset that can be used with
    DataLoader for training.

    Example:
        >>> dataset = InstructionDataset(data, tokenizer)
        >>> loader = DataLoader(dataset, batch_size=4)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template: str = "alpaca"
    ):
        """Initialize dataset.

        Args:
            data: List of data examples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            template: Prompt template to use.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example.

        Args:
            idx: Example index.

        Returns:
            Dictionary with input_ids, attention_mask, labels tensors.
        """
        item = self.data[idx]

        # Format according to template
        if self.template == "alpaca":
            prompt = self._format_alpaca(item)
        elif self.template == "sharegpt":
            prompt = self._format_sharegpt(item)
        elif self.template == "chatml":
            prompt = self._format_chatml(item)
        else:
            raise ValueError(f"Unknown template: {self.template}")

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

    def _format_alpaca(self, item: Dict[str, Any]) -> str:
        """Format data in Alpaca style."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if input_text:
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            return (
                "Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )

    def _format_sharegpt(self, item: Dict[str, Any]) -> str:
        """Format data in ShareGPT conversation style."""
        conversations = item.get("conversations", [])

        formatted = ""
        for conv in conversations:
            role = conv.get("from", "unknown")
            content = conv.get("value", "")

            if role == "human":
                formatted += f"Human: {content}\n\n"
            elif role == "gpt":
                formatted += f"Assistant: {content}\n\n"

        return formatted.strip()

    def _format_chatml(self, item: Dict[str, Any]) -> str:
        """Format data in ChatML style."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )


class StreamingDataset:
    """Dataset supporting streaming for very large files.

    Yields examples one at a time without loading entire dataset
    into memory.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        format: str = "jsonl"
    ):
        """Initialize streaming dataset.

        Args:
            data_path: Path to data file.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            format: Data format.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format = format

    def __iter__(self):
        """Iterate over examples."""
        if self.format == "jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            raise ValueError(f"Streaming not supported for format: {self.format}")


class DataDeduplicator:
    """Deduplicate training data using multiple strategies.
    
    Supports exact matching, fuzzy matching, and semantic deduplication.
    """
    
    def __init__(
        self,
        strategy: str = "exact",
        similarity_threshold: float = 0.9,
        hash_fields: Optional[List[str]] = None
    ):
        """Initialize deduplicator.
        
        Args:
            strategy: Deduplication strategy ('exact', 'fuzzy', 'semantic').
            similarity_threshold: Threshold for fuzzy/semantic matching.
            hash_fields: Fields to use for hashing (default: all).
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.hash_fields = hash_fields or ["instruction", "input", "output"]
        self._seen_hashes: set = set()
        
    def deduplicate(self, dataset: Dataset) -> Dataset:
        """Remove duplicate examples from dataset.
        
        Args:
            dataset: HuggingFace Dataset.
            
        Returns:
            Deduplicated dataset.
        """
        import hashlib
        
        self._seen_hashes.clear()
        unique_indices = []
        
        for idx, example in enumerate(dataset):
            if self.strategy == "exact":
                example_hash = self._compute_hash(example)
                if example_hash not in self._seen_hashes:
                    self._seen_hashes.add(example_hash)
                    unique_indices.append(idx)
            elif self.strategy == "fuzzy":
                if not self._is_fuzzy_duplicate(example):
                    unique_indices.append(idx)
            else:
                unique_indices.append(idx)
        
        original_size = len(dataset)
        deduplicated = dataset.select(unique_indices)
        
        logger.info(
            f"Deduplication: {original_size} -> {len(deduplicated)} "
            f"(removed {original_size - len(deduplicated)} duplicates)"
        )
        
        return deduplicated
    
    def _compute_hash(self, example: Dict) -> str:
        """Compute hash for example."""
        import hashlib
        
        content = ""
        for field in self.hash_fields:
            content += str(example.get(field, ""))
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_fuzzy_duplicate(self, example: Dict) -> bool:
        """Check if example is a fuzzy duplicate."""
        # Simple character-level similarity
        content = "".join(str(example.get(f, "")) for f in self.hash_fields)
        content_set = set(content.lower().split())
        
        for seen_hash in list(self._seen_hashes)[:1000]:  # Limit for performance
            # Compare with stored content
            similarity = len(content_set) / max(len(content_set), 1)
            if similarity > self.similarity_threshold:
                return True
        
        self._seen_hashes.add(content)
        return False


class DataValidator:
    """Validate training data quality.
    
    Performs multiple quality checks on training examples.
    """
    
    def __init__(
        self,
        min_instruction_length: int = 10,
        max_instruction_length: int = 1000,
        min_output_length: int = 20,
        max_output_length: int = 5000,
        check_language: bool = False,
        required_fields: Optional[List[str]] = None
    ):
        """Initialize validator.
        
        Args:
            min_instruction_length: Minimum instruction length.
            max_instruction_length: Maximum instruction length.
            min_output_length: Minimum output length.
            max_output_length: Maximum output length.
            check_language: Check language consistency.
            required_fields: Fields that must be present.
        """
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.check_language = check_language
        self.required_fields = required_fields or ["instruction", "output"]
        
        self._validation_stats = {
            "total": 0,
            "valid": 0,
            "invalid_length": 0,
            "missing_fields": 0,
            "low_quality": 0,
        }
    
    def validate(self, dataset: Dataset) -> Dataset:
        """Validate and filter dataset.
        
        Args:
            dataset: HuggingFace Dataset.
            
        Returns:
            Validated dataset with only valid examples.
        """
        self._reset_stats()
        
        valid_indices = []
        
        for idx, example in enumerate(dataset):
            self._validation_stats["total"] += 1
            
            is_valid, reason = self._validate_example(example)
            
            if is_valid:
                self._validation_stats["valid"] += 1
                valid_indices.append(idx)
            else:
                self._validation_stats[reason] = self._validation_stats.get(reason, 0) + 1
        
        validated = dataset.select(valid_indices)
        
        logger.info(f"Validation stats: {self._validation_stats}")
        
        return validated
    
    def _validate_example(self, example: Dict) -> tuple:
        """Validate single example.
        
        Returns:
            Tuple of (is_valid, reason).
        """
        # Check required fields
        for field in self.required_fields:
            if field not in example or not example[field]:
                return False, "missing_fields"
        
        instruction = str(example.get("instruction", ""))
        output = str(example.get("output", ""))
        
        # Length checks
        if len(instruction) < self.min_instruction_length:
            return False, "invalid_length"
        if len(instruction) > self.max_instruction_length:
            return False, "invalid_length"
        if len(output) < self.min_output_length:
            return False, "invalid_length"
        if len(output) > self.max_output_length:
            return False, "invalid_length"
        
        # Quality checks
        if self._is_low_quality(example):
            return False, "low_quality"
        
        return True, None
    
    def _is_low_quality(self, example: Dict) -> bool:
        """Check if example is low quality."""
        output = str(example.get("output", "")).lower()
        
        # Check for refusal patterns
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable", "as an ai",
            "i don't have access", "sorry, but"
        ]
        
        for pattern in refusal_patterns:
            if pattern in output:
                return True
        
        # Check for very short responses
        words = output.split()
        if len(words) < 5:
            return True
        
        return False
    
    def _reset_stats(self):
        """Reset validation statistics."""
        self._validation_stats = {
            "total": 0,
            "valid": 0,
            "invalid_length": 0,
            "missing_fields": 0,
            "low_quality": 0,
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self._validation_stats.copy()


class DataCleaner:
    """Clean and normalize training data.
    
    Performs text cleaning, formatting normalization, and
    quality improvements.
    """
    
    def __init__(
        self,
        normalize_whitespace: bool = True,
        remove_html: bool = True,
        fix_encoding: bool = True,
        lowercase_instructions: bool = False
    ):
        """Initialize cleaner.
        
        Args:
            normalize_whitespace: Normalize whitespace characters.
            remove_html: Remove HTML tags.
            fix_encoding: Fix encoding issues.
            lowercase_instructions: Convert instructions to lowercase.
        """
        self.normalize_whitespace = normalize_whitespace
        self.remove_html = remove_html
        self.fix_encoding = fix_encoding
        self.lowercase_instructions = lowercase_instructions
    
    def clean(self, dataset: Dataset) -> Dataset:
        """Clean entire dataset.
        
        Args:
            dataset: HuggingFace Dataset.
            
        Returns:
            Cleaned dataset.
        """
        return dataset.map(self._clean_example, desc="Cleaning data")
    
    def _clean_example(self, example: Dict) -> Dict:
        """Clean single example."""
        cleaned = example.copy()
        
        for field in ["instruction", "input", "output"]:
            if field in cleaned and cleaned[field]:
                cleaned[field] = self._clean_text(
                    cleaned[field],
                    is_instruction=(field == "instruction")
                )
        
        return cleaned
    
    def _clean_text(self, text: str, is_instruction: bool = False) -> str:
        """Clean text content."""
        import re
        
        # Fix encoding issues
        if self.fix_encoding:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Lowercase instructions if configured
        if is_instruction and self.lowercase_instructions:
            text = text.lower()
        
        return text


class DataPipeline:
    """Complete data processing pipeline.
    
    Combines loading, cleaning, validation, and deduplication.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        deduplicate: bool = True,
        validate: bool = True,
        clean: bool = True
    ):
        """Initialize pipeline.
        
        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            deduplicate: Enable deduplication.
            validate: Enable validation.
            clean: Enable cleaning.
        """
        self.loader = AdaptiveDataLoader(tokenizer, max_length)
        self.deduplicator = DataDeduplicator() if deduplicate else None
        self.validator = DataValidator() if validate else None
        self.cleaner = DataCleaner() if clean else None
    
    def process(
        self,
        data_path: Union[str, Path],
        format: str = "auto"
    ) -> DatasetDict:
        """Run complete processing pipeline.
        
        Args:
            data_path: Path to data.
            format: Data format.
            
        Returns:
            Processed DatasetDict.
        """
        # Load
        dataset = self.loader.load(data_path, format=format)
        
        # Process each split
        for split_name in dataset:
            split = dataset[split_name]
            
            # Clean
            if self.cleaner:
                split = self.cleaner.clean(split)
            
            # Validate
            if self.validator:
                split = self.validator.validate(split)
            
            # Deduplicate
            if self.deduplicator:
                split = self.deduplicator.deduplicate(split)
            
            dataset[split_name] = split
        
        return dataset

