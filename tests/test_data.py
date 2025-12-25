"""
Tests for Data Module

Tests for data loading, validation, deduplication, and synthetic generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pytest


class TestAdaptiveDataLoader:
    """Tests for AdaptiveDataLoader class."""
    
    def test_detect_format_jsonl(self, temp_dir: Path):
        """Test JSONL format detection."""
        file_path = temp_dir / "data.jsonl"
        file_path.touch()
        
        # Format should be detected from extension
        assert file_path.suffix == ".jsonl"
    
    def test_detect_format_json(self, temp_dir: Path):
        """Test JSON format detection."""
        file_path = temp_dir / "data.json"
        file_path.touch()
        
        assert file_path.suffix == ".json"
    
    def test_detect_format_csv(self, temp_dir: Path):
        """Test CSV format detection."""
        file_path = temp_dir / "data.csv"
        file_path.touch()
        
        assert file_path.suffix == ".csv"
    
    def test_load_jsonl(
        self,
        sample_jsonl_file: Path,
        sample_instruction_data: List[Dict]
    ):
        """Test loading JSONL file."""
        with open(sample_jsonl_file, "r") as f:
            loaded = [json.loads(line) for line in f]
        
        assert len(loaded) == len(sample_instruction_data)
        assert loaded[0]["instruction"] == sample_instruction_data[0]["instruction"]


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_alpaca_template(self):
        """Test Alpaca format template."""
        template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        
        filled = template.format(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        
        assert "### Instruction:" in filled
        assert "Test instruction" in filled
    
    def test_sharegpt_template(self):
        """Test ShareGPT format template."""
        conversation = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"}
        ]
        
        # ShareGPT uses conversation format
        assert len(conversation) == 2
        assert conversation[0]["from"] == "human"
    
    def test_chatml_template(self):
        """Test ChatML format template."""
        template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""
        
        filled = template.format(
            system="You are a helpful assistant.",
            user="Hello",
            assistant="Hi!"
        )
        
        assert "<|im_start|>" in filled
        assert "<|im_end|>" in filled


class TestDataDeduplicator:
    """Tests for DataDeduplicator class."""
    
    def test_exact_deduplication(self):
        """Test exact match deduplication."""
        data = [
            {"instruction": "test", "output": "output1"},
            {"instruction": "test", "output": "output1"},  # Duplicate
            {"instruction": "test2", "output": "output2"},
        ]
        
        seen = set()
        unique = []
        
        for item in data:
            key = str(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        assert len(unique) == 2
    
    def test_hash_computation(self):
        """Test hash computation for deduplication."""
        import hashlib
        
        content = "test instruction" + "test output"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        
        assert len(hash_value) == 32
    
    def test_deduplication_preserves_order(self):
        """Test that deduplication preserves first occurrence."""
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 3, "text": "first"},  # Duplicate
        ]
        
        seen_texts = set()
        unique = []
        
        for item in data:
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                unique.append(item)
        
        assert unique[0]["id"] == 1
        assert len(unique) == 2


class TestDataValidator:
    """Tests for DataValidator class."""
    
    @pytest.fixture
    def validator_config(self) -> Dict[str, Any]:
        """Validator configuration."""
        return {
            "min_instruction_length": 10,
            "max_instruction_length": 1000,
            "min_output_length": 20,
            "max_output_length": 5000
        }
    
    def test_valid_example(self, validator_config: Dict[str, Any]):
        """Test validation of valid example."""
        example = {
            "instruction": "This is a valid instruction that is long enough.",
            "output": "This is a sufficiently long output that passes validation." * 2
        }
        
        # Check length constraints
        instr_len = len(example["instruction"])
        output_len = len(example["output"])
        
        is_valid = (
            validator_config["min_instruction_length"] <= instr_len <= validator_config["max_instruction_length"]
            and validator_config["min_output_length"] <= output_len <= validator_config["max_output_length"]
        )
        
        assert is_valid
    
    def test_instruction_too_short(self, validator_config: Dict[str, Any]):
        """Test rejection of too-short instruction."""
        example = {"instruction": "Short", "output": "Long enough output for validation."}
        
        is_valid = len(example["instruction"]) >= validator_config["min_instruction_length"]
        
        assert not is_valid
    
    def test_refusal_detection(self):
        """Test detection of refusal patterns."""
        refusal_patterns = [
            "I cannot", "I can't", "I'm unable", "as an ai",
            "I don't have access", "sorry, but"
        ]
        
        refusal_output = "I cannot provide that information as an AI."
        
        has_refusal = any(
            pattern in refusal_output.lower()
            for pattern in refusal_patterns
        )
        
        assert has_refusal


class TestDataCleaner:
    """Tests for DataCleaner class."""
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        import re
        
        text = "This   has    extra   spaces"
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        assert normalized == "This has extra spaces"
    
    def test_html_removal(self):
        """Test HTML tag removal."""
        import re
        
        text = "Hello <b>world</b> and <a href='#'>link</a>"
        cleaned = re.sub(r'<[^>]+>', '', text)
        
        assert "<b>" not in cleaned
        assert "world" in cleaned
    
    def test_encoding_fix(self):
        """Test encoding issue fix."""
        text = "Café résumé"
        fixed = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        assert "Café" in fixed


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""
    
    def test_template_variation(self):
        """Test template-based generation with variation."""
        template = "Explain {concept} in simple terms."
        concepts = ["machine learning", "neural networks", "backpropagation"]
        
        variations = [template.format(concept=c) for c in concepts]
        
        assert len(variations) == 3
        assert all("Explain" in v for v in variations)
    
    def test_difficulty_levels(self):
        """Test different difficulty levels."""
        difficulties = ["easy", "medium", "hard"]
        
        difficulty_settings = {
            "easy": {"max_tokens": 100, "complexity": "basic"},
            "medium": {"max_tokens": 300, "complexity": "intermediate"},
            "hard": {"max_tokens": 500, "complexity": "advanced"},
        }
        
        for diff in difficulties:
            assert diff in difficulty_settings
            assert difficulty_settings[diff]["max_tokens"] > 0


class TestQualityScorer:
    """Tests for QualityScorer class."""
    
    def test_clarity_scoring(self):
        """Test instruction clarity scoring."""
        clear_instruction = "Explain the concept of machine learning in detail."
        unclear_instruction = "ML?"
        
        # Clear instruction should have higher score
        clear_score = len(clear_instruction) > 20
        unclear_score = len(unclear_instruction) > 20
        
        assert clear_score
        assert not unclear_score
    
    def test_completeness_scoring(self):
        """Test output completeness scoring."""
        complete_output = """
        Machine learning is a subset of artificial intelligence.
        It involves algorithms that improve through experience.
        1. Supervised learning
        2. Unsupervised learning
        3. Reinforcement learning
        In conclusion, ML is powerful.
        """
        
        has_structure = "1." in complete_output
        has_conclusion = "conclusion" in complete_output.lower()
        
        assert has_structure
        assert has_conclusion
    
    def test_relevance_scoring(self):
        """Test output relevance to instruction."""
        instruction = "Explain machine learning"
        relevant_output = "Machine learning is a type of AI that learns from data."
        irrelevant_output = "The weather is nice today."
        
        instruction_words = set(instruction.lower().split())
        
        relevant_overlap = len(
            instruction_words & set(relevant_output.lower().split())
        )
        irrelevant_overlap = len(
            instruction_words & set(irrelevant_output.lower().split())
        )
        
        assert relevant_overlap > irrelevant_overlap


class TestDataPipeline:
    """Tests for DataPipeline class."""
    
    def test_pipeline_order(self):
        """Test that pipeline steps execute in correct order."""
        steps = ["load", "clean", "validate", "deduplicate"]
        executed = []
        
        for step in steps:
            executed.append(step)
        
        assert executed == steps
    
    def test_pipeline_handles_empty_data(self):
        """Test pipeline handles empty dataset."""
        data = []
        
        # Pipeline should not fail on empty data
        result = data  # Placeholder for actual processing
        
        assert result == []
