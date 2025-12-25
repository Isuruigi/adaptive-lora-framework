"""
Synthetic data generation system for training data augmentation.

Features:
- Template-based generation with variation
- Difficulty levels (easy, medium, hard)
- LLM-based generation (GPT-4, Claude)
- Quality filtering and validation
- Adversarial example generation
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports
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
class SyntheticExample:
    """Container for synthetic training example.

    Attributes:
        instruction: Task instruction.
        input: Additional input context.
        output: Expected response.
        difficulty: Difficulty level (easy, medium, hard).
        source: Generation source (template, gpt4, claude).
        validation_score: Quality score from validation.
        metadata: Additional metadata.
    """

    instruction: str
    input: str
    output: str
    difficulty: str
    source: str
    validation_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation.
        """
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "difficulty": self.difficulty,
            "source": self.source,
            "validation_score": self.validation_score,
            "metadata": self.metadata
        }


class SyntheticDataGenerator:
    """Generate high-quality synthetic training data.

    Supports both template-based and LLM-based generation with
    quality filtering.

    Example:
        >>> generator = SyntheticDataGenerator(openai_api_key="sk-...")
        >>> examples = generator.generate_batch("reasoning", num_examples=100)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        use_templates: bool = True,
        use_llm: bool = True
    ):
        """Initialize generator.

        Args:
            openai_api_key: OpenAI API key for GPT-based generation.
            anthropic_api_key: Anthropic API key for Claude-based generation.
            use_templates: Enable template-based generation.
            use_llm: Enable LLM-based generation.
        """
        self.use_templates = use_templates
        self.use_llm = use_llm and (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE)

        # Setup API clients
        if use_llm:
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

        self.templates = self._load_templates()
        self.validators: List[Callable] = []

    def generate_batch(
        self,
        task_type: str,
        num_examples: int,
        difficulty: str = "medium",
        target_patterns: Optional[List[str]] = None
    ) -> List[SyntheticExample]:
        """Generate batch of synthetic examples.

        Args:
            task_type: Type of task (reasoning, code, analysis, etc.).
            num_examples: Number of examples to generate.
            difficulty: Difficulty level (easy, medium, hard).
            target_patterns: Specific patterns to target.

        Returns:
            List of validated synthetic examples.
        """
        logger.info(
            f"Generating {num_examples} {difficulty} examples for {task_type}"
        )

        examples: List[SyntheticExample] = []

        # Generate from templates
        if self.use_templates:
            template_count = num_examples // 2 if self.use_llm else num_examples
            template_examples = self._generate_from_templates(
                task_type, template_count, difficulty
            )
            examples.extend(template_examples)
            logger.info(f"Generated {len(template_examples)} template examples")

        # Generate from LLM
        if self.use_llm:
            llm_count = num_examples - len(examples)
            if llm_count > 0:
                llm_examples = self._generate_from_llm(
                    task_type, llm_count, difficulty, target_patterns
                )
                examples.extend(llm_examples)
                logger.info(f"Generated {len(llm_examples)} LLM examples")

        # Validate and filter
        validated_examples = self._validate_examples(examples)
        logger.info(f"Validated {len(validated_examples)} examples")

        return validated_examples[:num_examples]

    def _generate_from_templates(
        self,
        task_type: str,
        num_examples: int,
        difficulty: str
    ) -> List[SyntheticExample]:
        """Generate examples from templates with variation.

        Args:
            task_type: Type of task.
            num_examples: Number of examples.
            difficulty: Difficulty level.

        Returns:
            List of generated examples.
        """
        templates = self.templates.get(task_type, {}).get(difficulty, [])

        if not templates:
            logger.warning(f"No templates for {task_type}/{difficulty}")
            return []

        examples = []
        for _ in range(num_examples):
            template = random.choice(templates)

            # Apply variations to template
            instruction = self._apply_variations(template.get("instruction", ""))
            input_text = self._apply_variations(template.get("input", ""))
            output = template.get("output", "")

            examples.append(SyntheticExample(
                instruction=instruction,
                input=input_text,
                output=output,
                difficulty=difficulty,
                source="template",
                validation_score=0.0,
                metadata={"template_id": template.get("id")}
            ))

        return examples

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_from_llm(
        self,
        task_type: str,
        num_examples: int,
        difficulty: str,
        target_patterns: Optional[List[str]] = None
    ) -> List[SyntheticExample]:
        """Generate examples using LLM.

        Args:
            task_type: Type of task.
            num_examples: Number of examples.
            difficulty: Difficulty level.
            target_patterns: Specific patterns to target.

        Returns:
            List of generated examples.
        """
        prompt = self._create_generation_prompt(
            task_type, difficulty, num_examples, target_patterns
        )

        try:
            # Try OpenAI first
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating high-quality "
                                       "training data for language models."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=4000
                )
                generated_text = response.choices[0].message.content
            elif self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000
                )
                generated_text = response.content[0].text
            else:
                logger.warning("No LLM client available")
                return []

            examples = self._parse_generated_examples(generated_text, difficulty)
            return examples

        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            return []

    def _create_generation_prompt(
        self,
        task_type: str,
        difficulty: str,
        num_examples: int,
        target_patterns: Optional[List[str]] = None
    ) -> str:
        """Create prompt for LLM-based generation.

        Args:
            task_type: Type of task.
            difficulty: Difficulty level.
            num_examples: Number of examples.
            target_patterns: Specific patterns to target.

        Returns:
            Generation prompt.
        """
        base_prompt = f"""Generate {num_examples} diverse training examples for {task_type} tasks at {difficulty} difficulty level.

Each example should have:
1. Instruction: Clear task description
2. Input: Context or data (can be empty string if not needed)
3. Output: High-quality, detailed response

Requirements:
- Examples should be diverse and cover different aspects of {task_type}
- Difficulty: {difficulty}
- Outputs should be detailed, accurate, and well-formatted
- Avoid generic or shallow responses
"""

        if target_patterns:
            base_prompt += (
                f"\nSpecifically target these patterns:\n"
                + "\n".join(f"- {p}" for p in target_patterns)
            )

        base_prompt += """

Format as a JSON array:
```json
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]
```"""

        return base_prompt

    def _validate_examples(
        self,
        examples: List[SyntheticExample]
    ) -> List[SyntheticExample]:
        """Validate generated examples with multiple checks.

        Args:
            examples: List of examples to validate.

        Returns:
            List of validated examples sorted by score.
        """
        validated = []

        for example in examples:
            score = 0.0

            # Length check
            if 10 < len(example.instruction) < 500:
                score += 0.2
            if 20 < len(example.output) < 3000:
                score += 0.2

            # Quality checks
            if self._check_coherence(example):
                score += 0.3
            if self._check_informativeness(example):
                score += 0.3

            example.validation_score = score

            # Keep if score > threshold
            if score >= 0.5:
                validated.append(example)

        # Sort by score
        validated.sort(key=lambda x: x.validation_score, reverse=True)

        return validated

    def _check_coherence(self, example: SyntheticExample) -> bool:
        """Check if output is coherent with instruction.

        Args:
            example: Example to check.

        Returns:
            True if coherent.
        """
        instruction_lower = example.instruction.lower()
        output_lower = example.output.lower()

        # Check for keyword overlap
        instruction_words = set(instruction_lower.split())
        output_words = set(output_lower.split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in"}
        instruction_words -= stop_words
        output_words -= stop_words

        overlap = len(instruction_words & output_words)
        return overlap >= 2

    def _check_informativeness(self, example: SyntheticExample) -> bool:
        """Check if output is informative.

        Args:
            example: Example to check.

        Returns:
            True if informative.
        """
        # Avoid generic responses
        generic_phrases = [
            "i cannot", "i don't know", "sorry", "as an ai",
            "i'm unable", "unfortunately", "i can't"
        ]

        output_lower = example.output.lower()
        for phrase in generic_phrases:
            if phrase in output_lower:
                return False

        # Check for sufficient content
        sentences = [s for s in example.output.split(".") if s.strip()]
        return len(sentences) >= 1

    def _apply_variations(self, text: str) -> str:
        """Apply random variations to template text.

        Args:
            text: Template text with placeholders.

        Returns:
            Text with placeholders replaced.
        """
        variations = {
            "{concept}": random.choice([
                "machine learning", "neural networks", "deep learning",
                "natural language processing", "computer vision"
            ]),
            "{language}": random.choice([
                "Python", "JavaScript", "TypeScript", "Rust", "Go"
            ]),
            "{topic}": random.choice([
                "algorithms", "data structures", "system design",
                "testing", "debugging"
            ]),
            "{number}": str(random.randint(1, 100)),
        }

        for placeholder, value in variations.items():
            text = text.replace(placeholder, value)

        return text

    def _parse_generated_examples(
        self,
        generated_text: str,
        difficulty: str
    ) -> List[SyntheticExample]:
        """Parse LLM-generated examples.

        Args:
            generated_text: Raw LLM output.
            difficulty: Difficulty level.

        Returns:
            List of parsed examples.
        """
        # Extract JSON from markdown code blocks
        json_match = re.search(r"```json\n(.*?)\n```", generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = generated_text

        try:
            examples_data = json.loads(json_str)

            examples = []
            for data in examples_data:
                examples.append(SyntheticExample(
                    instruction=data.get("instruction", ""),
                    input=data.get("input", ""),
                    output=data.get("output", ""),
                    difficulty=difficulty,
                    source="llm",
                    validation_score=0.0,
                    metadata={}
                ))

            return examples

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM output: {e}")
            return []

    def _load_templates(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load task templates.

        Returns:
            Dictionary of templates by task type and difficulty.
        """
        return {
            "reasoning": {
                "easy": [
                    {
                        "id": "reasoning_easy_1",
                        "instruction": "Explain why {concept} is important.",
                        "input": "",
                        "output": "This concept is important because it provides foundational understanding for building more complex systems and solving real-world problems."
                    },
                    {
                        "id": "reasoning_easy_2",
                        "instruction": "What are three benefits of {concept}?",
                        "input": "",
                        "output": "Three key benefits are: 1) Improved efficiency in processing data, 2) Better accuracy in predictions, and 3) Scalability for large-scale applications."
                    }
                ],
                "medium": [
                    {
                        "id": "reasoning_medium_1",
                        "instruction": "Compare and contrast two approaches to {concept}.",
                        "input": "",
                        "output": "When comparing different approaches, we need to consider factors like performance, complexity, and use cases. Each approach has trade-offs depending on the specific requirements."
                    }
                ],
                "hard": [
                    {
                        "id": "reasoning_hard_1",
                        "instruction": "Design a system that leverages {concept} for solving complex optimization problems.",
                        "input": "Consider scalability and real-time requirements.",
                        "output": "A robust system design would incorporate multiple layers: data ingestion, processing pipeline, optimization engine, and monitoring. Key considerations include fault tolerance, horizontal scaling, and latency requirements."
                    }
                ]
            },
            "code": {
                "easy": [
                    {
                        "id": "code_easy_1",
                        "instruction": "Write a function to calculate the factorial of a number in {language}.",
                        "input": "",
                        "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
                    }
                ],
                "medium": [
                    {
                        "id": "code_medium_1",
                        "instruction": "Implement a binary search algorithm in {language}.",
                        "input": "",
                        "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
                    }
                ],
                "hard": []
            },
            "analysis": {
                "easy": [],
                "medium": [],
                "hard": []
            }
        }


class AdversarialGenerator(SyntheticDataGenerator):
    """Generate adversarial examples targeting model weaknesses.

    Extends SyntheticDataGenerator to focus on generating examples
    that address specific failure patterns.
    """

    def generate_targeted_examples(
        self,
        failure_patterns: List[Dict[str, Any]],
        num_examples_per_pattern: int = 10
    ) -> List[SyntheticExample]:
        """Generate examples specifically targeting failure patterns.

        Args:
            failure_patterns: List of identified failure patterns.
                Each pattern: {
                    'type': 'error_type',
                    'description': 'what the model gets wrong',
                    'examples': [...],
                    'severity': 'low'|'medium'|'high'
                }
            num_examples_per_pattern: Examples to generate per pattern.

        Returns:
            List of targeted synthetic examples.
        """
        all_examples = []

        for pattern in failure_patterns:
            logger.info(f"Generating examples for pattern: {pattern.get('type')}")

            examples = self._generate_for_pattern(
                pattern,
                num_examples_per_pattern
            )
            all_examples.extend(examples)

        return self._validate_examples(all_examples)

    def _generate_for_pattern(
        self,
        pattern: Dict[str, Any],
        num_examples: int
    ) -> List[SyntheticExample]:
        """Generate examples for specific failure pattern.

        Args:
            pattern: Failure pattern dictionary.
            num_examples: Number of examples to generate.

        Returns:
            List of generated examples.
        """
        prompt = f"""Generate {num_examples} training examples to address this model weakness:

Error Type: {pattern.get('type', 'unknown')}
Description: {pattern.get('description', '')}
Examples of failures: {pattern.get('examples', [])}

Create diverse examples that would help the model learn to avoid this error.
Make them challenging but correct.

Format as JSON array with instruction/input/output fields."""

        examples = self._generate_from_llm(
            task_type="adversarial",
            num_examples=num_examples,
            difficulty="hard",
            target_patterns=[pattern.get("description", "")]
        )

        return examples


class QualityScorer:
    """Advanced quality scoring for synthetic examples.
    
    Uses multiple rubrics to score example quality.
    """
    
    def __init__(
        self,
        use_llm_scoring: bool = False,
        openai_api_key: Optional[str] = None
    ):
        """Initialize scorer.
        
        Args:
            use_llm_scoring: Use LLM for advanced scoring.
            openai_api_key: OpenAI API key for LLM scoring.
        """
        self.use_llm_scoring = use_llm_scoring
        self.openai_api_key = openai_api_key
        
        # Scoring rubrics
        self.rubrics = {
            "clarity": self._score_clarity,
            "completeness": self._score_completeness,
            "accuracy": self._score_accuracy,
            "relevance": self._score_relevance,
            "coherence": self._score_coherence,
        }
    
    def score(self, example: SyntheticExample) -> Dict[str, float]:
        """Score example on multiple dimensions.
        
        Args:
            example: Example to score.
            
        Returns:
            Dictionary of scores by dimension.
        """
        scores = {}
        
        for rubric_name, scorer_fn in self.rubrics.items():
            scores[rubric_name] = scorer_fn(example)
        
        # Compute overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def score_batch(
        self,
        examples: List[SyntheticExample]
    ) -> List[Dict[str, float]]:
        """Score batch of examples.
        
        Args:
            examples: List of examples.
            
        Returns:
            List of score dictionaries.
        """
        return [self.score(ex) for ex in examples]
    
    def _score_clarity(self, example: SyntheticExample) -> float:
        """Score instruction clarity."""
        instruction = example.instruction
        
        score = 1.0
        
        # Penalize very short instructions
        if len(instruction) < 20:
            score -= 0.3
        
        # Penalize very long instructions
        if len(instruction) > 500:
            score -= 0.2
        
        # Check for question marks or action verbs
        if "?" in instruction or any(
            verb in instruction.lower() 
            for verb in ["explain", "describe", "write", "create", "analyze", "compare"]
        ):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_completeness(self, example: SyntheticExample) -> float:
        """Score output completeness."""
        output = example.output
        
        score = 0.0
        
        # Length-based scoring
        word_count = len(output.split())
        
        if word_count >= 50:
            score += 0.4
        elif word_count >= 20:
            score += 0.2
        
        # Check for structure
        if any(marker in output for marker in ["1.", "- ", "•", "First", "Second"]):
            score += 0.3
        
        # Check for conclusion/summary
        if any(phrase in output.lower() for phrase in ["in conclusion", "to summarize", "overall"]):
            score += 0.2
        
        # Check for examples
        if "example" in output.lower() or "for instance" in output.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_accuracy(self, example: SyntheticExample) -> float:
        """Score factual accuracy (heuristic-based)."""
        output = example.output.lower()
        
        score = 1.0
        
        # Penalize hedging language
        hedging = ["might", "maybe", "perhaps", "possibly", "could be"]
        hedging_count = sum(1 for h in hedging if h in output)
        score -= hedging_count * 0.1
        
        # Penalize contradictions
        if "however" in output and "but" in output and "although" in output:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_relevance(self, example: SyntheticExample) -> float:
        """Score output relevance to instruction."""
        instruction = example.instruction.lower()
        output = example.output.lower()
        
        # Extract key terms from instruction
        instruction_words = set(instruction.split())
        output_words = set(output.split())
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "with"}
        instruction_words -= stopwords
        output_words -= stopwords
        
        if not instruction_words:
            return 0.5
        
        overlap = len(instruction_words & output_words)
        relevance = overlap / len(instruction_words)
        
        return min(1.0, relevance * 1.5)  # Scale up slightly
    
    def _score_coherence(self, example: SyntheticExample) -> float:
        """Score output coherence."""
        output = example.output
        
        score = 1.0
        
        sentences = output.split(".")
        
        # Check for reasonable sentence count
        if len(sentences) < 2:
            score -= 0.3
        
        # Check for very long sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 50:
            score -= 0.2
        
        return max(0.0, min(1.0, score))


class CrossValidator:
    """Cross-validate synthetic examples using multiple models.
    
    Validates generated examples by checking consistency across
    different LLMs.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        consistency_threshold: float = 0.7
    ):
        """Initialize validator.
        
        Args:
            openai_api_key: OpenAI API key.
            anthropic_api_key: Anthropic API key.
            consistency_threshold: Minimum agreement threshold.
        """
        self.consistency_threshold = consistency_threshold
        
        self.validators = []
        
        if openai_api_key and OPENAI_AVAILABLE:
            self.validators.append(("openai", openai.OpenAI(api_key=openai_api_key)))
        
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            self.validators.append(("anthropic", anthropic.Anthropic(api_key=anthropic_api_key)))
    
    def validate(self, example: SyntheticExample) -> Dict[str, Any]:
        """Cross-validate example with multiple models.
        
        Args:
            example: Example to validate.
            
        Returns:
            Validation result with scores and agreement.
        """
        if len(self.validators) < 1:
            logger.warning("No validators configured")
            return {"valid": True, "reason": "no_validators"}
        
        validation_prompt = self._create_validation_prompt(example)
        
        results = []
        
        for name, client in self.validators:
            try:
                result = self._validate_with_model(name, client, validation_prompt)
                results.append(result)
            except Exception as e:
                logger.warning(f"Validation failed with {name}: {e}")
        
        if not results:
            return {"valid": True, "reason": "validation_failed"}
        
        # Calculate agreement
        agreement = sum(1 for r in results if r.get("valid", False)) / len(results)
        
        return {
            "valid": agreement >= self.consistency_threshold,
            "agreement": agreement,
            "results": results
        }
    
    def _create_validation_prompt(self, example: SyntheticExample) -> str:
        """Create validation prompt."""
        return f"""Evaluate this training example for quality and accuracy.

Instruction: {example.instruction}
Input: {example.input}
Output: {example.output}

Rate:
1. Is the output correct and accurate? (yes/no)
2. Is the output relevant to the instruction? (yes/no)
3. Is the output well-structured? (yes/no)
4. Any factual errors? (yes/no)

Respond in JSON format:
{{"valid": true/false, "reason": "explanation"}}"""
    
    def _validate_with_model(
        self,
        name: str,
        client: Any,
        prompt: str
    ) -> Dict[str, Any]:
        """Validate with specific model."""
        if name == "openai":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            text = response.choices[0].message.content
        elif name == "anthropic":
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            text = response.content[0].text
        else:
            return {"valid": True}
        
        # Parse response
        try:
            import json as json_module
            result = json_module.loads(text)
            return result
        except Exception:
            return {"valid": "yes" in text.lower() or "valid" in text.lower()}


class DifficultyEscalator:
    """Progressively increase difficulty of generated examples.
    
    Implements curriculum-style generation where examples become
    progressively more challenging.
    """
    
    def __init__(
        self,
        generator: SyntheticDataGenerator,
        levels: List[str] = None
    ):
        """Initialize escalator.
        
        Args:
            generator: Synthetic data generator.
            levels: Difficulty levels (default: easy, medium, hard).
        """
        self.generator = generator
        self.levels = levels or ["easy", "medium", "hard"]
        
        self.escalation_criteria = {
            "easy": {"max_tokens": 100, "complexity": "basic"},
            "medium": {"max_tokens": 300, "complexity": "intermediate"},
            "hard": {"max_tokens": 500, "complexity": "advanced"},
        }
    
    def generate_curriculum(
        self,
        task_type: str,
        total_examples: int,
        distribution: Optional[Dict[str, float]] = None
    ) -> List[SyntheticExample]:
        """Generate curriculum-ordered examples.
        
        Args:
            task_type: Type of task.
            total_examples: Total examples to generate.
            distribution: Distribution across difficulties (default: 30/40/30).
            
        Returns:
            List of examples ordered by difficulty.
        """
        distribution = distribution or {"easy": 0.3, "medium": 0.4, "hard": 0.3}
        
        all_examples = []
        
        for level in self.levels:
            count = int(total_examples * distribution.get(level, 0.33))
            
            if count > 0:
                logger.info(f"Generating {count} {level} examples")
                
                examples = self.generator.generate_batch(
                    task_type=task_type,
                    num_examples=count,
                    difficulty=level
                )
                
                all_examples.extend(examples)
        
        # Sort by difficulty
        difficulty_order = {level: i for i, level in enumerate(self.levels)}
        all_examples.sort(key=lambda x: difficulty_order.get(x.difficulty, 1))
        
        return all_examples
    
    def escalate_example(
        self,
        example: SyntheticExample,
        target_difficulty: str
    ) -> SyntheticExample:
        """Escalate example to higher difficulty.
        
        Args:
            example: Original example.
            target_difficulty: Target difficulty level.
            
        Returns:
            Escalated example.
        """
        if not self.generator.use_llm:
            return example
        
        prompt = f"""Transform this training example to {target_difficulty} difficulty:

Original:
Instruction: {example.instruction}
Input: {example.input}
Output: {example.output}

Make it more {self.escalation_criteria.get(target_difficulty, {}).get('complexity', 'challenging')}.
Add more depth, nuance, or edge cases.

Respond in JSON format with instruction, input, output fields."""
        
        try:
            escalated = self.generator._generate_from_llm(
                task_type="escalation",
                num_examples=1,
                difficulty=target_difficulty
            )
            
            if escalated:
                return escalated[0]
        except Exception as e:
            logger.warning(f"Escalation failed: {e}")
        
        return example

