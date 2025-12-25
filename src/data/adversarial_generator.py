"""
Adversarial example generation for targeted training.

Features:
- Failure pattern targeting
- Edge case generation
- Perturbation strategies
- Difficulty escalation
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AdversarialExample:
    """Generated adversarial example."""
    
    query: str
    expected_output: str
    target_weakness: str
    difficulty: str  # easy, medium, hard
    perturbation_type: str
    original_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerturbationStrategy(ABC):
    """Abstract perturbation strategy."""
    
    @abstractmethod
    def apply(self, text: str, **kwargs) -> str:
        """Apply perturbation to text."""
        pass


class NegationPerturbation(PerturbationStrategy):
    """Add negation to confuse logic."""
    
    def apply(self, text: str, **kwargs) -> str:
        """Insert or remove negations."""
        negation_words = ["not", "never", "no", "none", "neither"]
        words = text.split()
        
        if random.random() > 0.5:
            # Insert negation
            if "not" not in text.lower():
                insert_pos = random.randint(1, max(1, len(words) - 1))
                words.insert(insert_pos, "not")
        else:
            # Remove negation
            words = [w for w in words if w.lower() not in negation_words]
            
        return " ".join(words)


class ParaphrasePerturbation(PerturbationStrategy):
    """Paraphrase while preserving meaning."""
    
    def __init__(self):
        self.paraphrase_templates = [
            "In other words, {text}",
            "To put it differently, {text}",
            "Essentially, {text}",
            "What I mean is, {text}",
            "{text} - that is to say...",
        ]
        
    def apply(self, text: str, **kwargs) -> str:
        """Apply paraphrasing template."""
        template = random.choice(self.paraphrase_templates)
        return template.format(text=text)


class DistractorPerturbation(PerturbationStrategy):
    """Add distractor information."""
    
    def __init__(self):
        self.distractors = [
            "(though this may not be relevant)",
            "- and by the way, here's some additional context:",
            "Consider also that",
            "It's worth noting, although unrelated,",
            "As a tangent,",
        ]
        
    def apply(self, text: str, **kwargs) -> str:
        """Insert distractors."""
        distractor = random.choice(self.distractors)
        return f"{text} {distractor} {self._generate_noise()}"
    
    def _generate_noise(self) -> str:
        """Generate noise text."""
        noise_topics = [
            "the weather has been unusual lately",
            "many people have different opinions on this",
            "there are multiple perspectives to consider",
            "the historical context is quite interesting",
        ]
        return random.choice(noise_topics)


class AmbiguityPerturbation(PerturbationStrategy):
    """Introduce ambiguity."""
    
    def apply(self, text: str, **kwargs) -> str:
        """Make text more ambiguous."""
        ambiguity_phrases = [
            "maybe", "perhaps", "possibly", "it could be that",
            "in some cases", "depending on the interpretation",
        ]
        words = text.split()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(ambiguity_phrases))
        return " ".join(words)


class ComplexityPerturbation(PerturbationStrategy):
    """Increase query complexity."""
    
    def apply(self, text: str, **kwargs) -> str:
        """Make query more complex."""
        complexity_additions = [
            f"Given that {text}, and considering all edge cases,",
            f"If we assume {text}, and also account for exceptions,",
            f"Taking into account multiple factors, {text}, specifically",
            f"In the context of {text}, with all constraints applied,",
        ]
        return random.choice(complexity_additions)


class AdversarialGenerator:
    """Generate adversarial examples targeting specific weaknesses."""
    
    def __init__(
        self,
        llm_generator=None,
        seed: Optional[int] = None
    ):
        """Initialize generator.
        
        Args:
            llm_generator: Optional LLM for generation.
            seed: Random seed.
        """
        self.llm_generator = llm_generator
        
        if seed is not None:
            random.seed(seed)
            
        self.perturbation_strategies = {
            "negation": NegationPerturbation(),
            "paraphrase": ParaphrasePerturbation(),
            "distractor": DistractorPerturbation(),
            "ambiguity": AmbiguityPerturbation(),
            "complexity": ComplexityPerturbation(),
        }
        
        self.weakness_templates = self._load_weakness_templates()
        
    def _load_weakness_templates(self) -> Dict[str, List[Dict[str, str]]]:
        """Load templates targeting specific weaknesses."""
        return {
            "logic_errors": [
                {
                    "query": "If A implies B, and B implies C, does A imply C?",
                    "expected": "Yes, by transitivity of implication."
                },
                {
                    "query": "All X are Y. Z is not Y. Is Z an X?",
                    "expected": "No, Z cannot be X since all X are Y but Z is not Y."
                },
            ],
            "math_errors": [
                {
                    "query": "What is 15% of 80?",
                    "expected": "12"
                },
                {
                    "query": "If x + 5 = 12, what is x?",
                    "expected": "7"
                },
            ],
            "factual_errors": [
                {
                    "query": "What is the capital of France?",
                    "expected": "Paris"
                },
            ],
            "context_misuse": [
                {
                    "query": "Given the context about cats, what color is the sky?",
                    "expected": "The provided context about cats does not contain information about sky color."
                },
            ],
            "hallucination": [
                {
                    "query": "What did Albert Einstein say about quantum computing in his 1905 papers?",
                    "expected": "Einstein did not discuss quantum computing in 1905; quantum computing was theorized much later."
                },
            ],
        }
    
    def generate_from_weakness(
        self,
        weakness_type: str,
        num_examples: int = 10,
        difficulty: str = "medium"
    ) -> List[AdversarialExample]:
        """Generate examples targeting a specific weakness.
        
        Args:
            weakness_type: Type of weakness to target.
            num_examples: Number of examples.
            difficulty: Difficulty level.
            
        Returns:
            List of adversarial examples.
        """
        examples = []
        templates = self.weakness_templates.get(weakness_type, [])
        
        for _ in range(num_examples):
            if templates:
                template = random.choice(templates)
                query = template["query"]
                expected = template["expected"]
                
                # Apply perturbations based on difficulty
                if difficulty == "hard":
                    query = self._apply_multiple_perturbations(query, 2)
                elif difficulty == "medium":
                    query = self._apply_single_perturbation(query)
                    
                examples.append(AdversarialExample(
                    query=query,
                    expected_output=expected,
                    target_weakness=weakness_type,
                    difficulty=difficulty,
                    perturbation_type="template_based",
                    original_query=template["query"]
                ))
            else:
                # Generate synthetic example
                example = self._generate_synthetic(weakness_type, difficulty)
                if example:
                    examples.append(example)
                    
        return examples
    
    def _apply_single_perturbation(self, text: str) -> str:
        """Apply a single random perturbation."""
        strategy_name = random.choice(list(self.perturbation_strategies.keys()))
        strategy = self.perturbation_strategies[strategy_name]
        return strategy.apply(text)
    
    def _apply_multiple_perturbations(self, text: str, count: int) -> str:
        """Apply multiple perturbations."""
        result = text
        strategies = random.sample(
            list(self.perturbation_strategies.keys()),
            min(count, len(self.perturbation_strategies))
        )
        
        for strategy_name in strategies:
            result = self.perturbation_strategies[strategy_name].apply(result)
            
        return result
    
    def _generate_synthetic(
        self,
        weakness_type: str,
        difficulty: str
    ) -> Optional[AdversarialExample]:
        """Generate synthetic example using LLM if available."""
        if self.llm_generator is None:
            return None
            
        prompt = f"""Generate an adversarial test case targeting the weakness: {weakness_type}
Difficulty: {difficulty}

Provide:
1. A query that would expose this weakness
2. The expected correct response

Format:
Query: [query]
Expected: [expected response]
"""
        try:
            response = self.llm_generator.generate(prompt)
            # Parse response
            lines = response.strip().split("\n")
            query = ""
            expected = ""
            
            for line in lines:
                if line.startswith("Query:"):
                    query = line[6:].strip()
                elif line.startswith("Expected:"):
                    expected = line[9:].strip()
                    
            if query and expected:
                return AdversarialExample(
                    query=query,
                    expected_output=expected,
                    target_weakness=weakness_type,
                    difficulty=difficulty,
                    perturbation_type="llm_generated"
                )
        except Exception as e:
            logger.warning(f"Failed to generate synthetic example: {e}")
            
        return None
    
    def generate_from_failures(
        self,
        failure_examples: List[Dict[str, str]],
        num_per_failure: int = 5
    ) -> List[AdversarialExample]:
        """Generate examples based on actual failure patterns.
        
        Args:
            failure_examples: List of failure examples with 'query' and 'response'.
            num_per_failure: Examples to generate per failure.
            
        Returns:
            List of adversarial examples.
        """
        examples = []
        
        for failure in failure_examples:
            original_query = failure.get("query", "")
            
            for _ in range(num_per_failure):
                # Create variations of the failing query
                perturbed = self._apply_multiple_perturbations(original_query, 1)
                
                examples.append(AdversarialExample(
                    query=perturbed,
                    expected_output="",  # Need human annotation
                    target_weakness="failure_pattern",
                    difficulty="hard",
                    perturbation_type="failure_based",
                    original_query=original_query,
                    metadata={"original_failure": failure}
                ))
                
        return examples
    
    def generate_edge_cases(
        self,
        domain: str = "general",
        num_examples: int = 20
    ) -> List[AdversarialExample]:
        """Generate edge case examples.
        
        Args:
            domain: Domain for edge cases.
            num_examples: Number to generate.
            
        Returns:
            List of edge case examples.
        """
        edge_case_patterns = [
            # Empty/minimal input
            {"query": "", "type": "empty_input"},
            {"query": "?", "type": "minimal_input"},
            {"query": "   ", "type": "whitespace_only"},
            
            # Very long input
            {"query": "word " * 500, "type": "long_input"},
            
            # Special characters
            {"query": "What is <script>alert('test')</script>?", "type": "injection"},
            {"query": "Tell me about $$$^^^***", "type": "special_chars"},
            
            # Contradictory requests
            {"query": "Give me a short but comprehensive explanation", "type": "contradiction"},
            
            # Multi-language
            {"query": "Explain in English pero también en español", "type": "multilingual"},
        ]
        
        examples = []
        
        for pattern in edge_case_patterns[:num_examples]:
            examples.append(AdversarialExample(
                query=pattern["query"],
                expected_output="Handle gracefully",
                target_weakness="edge_case",
                difficulty="hard",
                perturbation_type=pattern["type"]
            ))
            
        return examples
    
    def escalate_difficulty(
        self,
        base_examples: List[AdversarialExample],
        target_difficulty: str = "hard"
    ) -> List[AdversarialExample]:
        """Escalate difficulty of existing examples.
        
        Args:
            base_examples: Base examples to escalate.
            target_difficulty: Target difficulty level.
            
        Returns:
            Escalated examples.
        """
        escalated = []
        
        for example in base_examples:
            if example.difficulty != target_difficulty:
                # Apply additional perturbations
                new_query = self._apply_multiple_perturbations(example.query, 2)
                
                escalated.append(AdversarialExample(
                    query=new_query,
                    expected_output=example.expected_output,
                    target_weakness=example.target_weakness,
                    difficulty=target_difficulty,
                    perturbation_type=f"escalated_{example.perturbation_type}",
                    original_query=example.query
                ))
            else:
                escalated.append(example)
                
        return escalated
