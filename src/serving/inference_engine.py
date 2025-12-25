"""
High-performance inference engine with vLLM integration.

Features:
- vLLM-based inference for high throughput
- Dynamic adapter switching
- Batch processing
- Streaming support
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional vLLM import
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using fallback inference")


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        presence_penalty: Presence penalty.
        frequency_penalty: Frequency penalty.
        stop_sequences: Sequences to stop generation.
    """

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None

    def to_vllm_params(self) -> "SamplingParams":
        """Convert to vLLM SamplingParams."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop=self.stop_sequences
        )


@dataclass
class GenerationResult:
    """Result from text generation.

    Attributes:
        text: Generated text.
        tokens_generated: Number of tokens generated.
        finish_reason: Reason for stopping generation.
        adapter_used: Adapter used for generation.
        latency_ms: Generation latency in milliseconds.
    """

    text: str
    tokens_generated: int
    finish_reason: str
    adapter_used: Optional[str]
    latency_ms: float


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> GenerationResult:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated text."""
        pass


class VLLMInferenceEngine(InferenceEngine):
    """High-performance inference engine using vLLM.

    Provides optimized inference with:
    - Continuous batching
    - PagedAttention
    - Dynamic LoRA loading
    - Multi-GPU support

    Example:
        >>> engine = VLLMInferenceEngine(
        ...     model_name="meta-llama/Llama-3-8B",
        ...     adapters_dir=Path("./adapters")
        ... )
        >>> result = engine.generate("What is AI?", GenerationConfig())
    """

    def __init__(
        self,
        model_name: str,
        adapters_dir: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enable_lora: bool = True
    ):
        """Initialize vLLM engine.

        Args:
            model_name: Base model name/path.
            adapters_dir: Directory containing LoRA adapters.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Target GPU memory utilization.
            max_model_len: Maximum model context length.
            enable_lora: Enable LoRA adapter support.
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")

        self.model_name = model_name
        self.adapters_dir = adapters_dir
        self.enable_lora = enable_lora

        # Initialize vLLM
        logger.info(f"Initializing vLLM engine with {model_name}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_loras=4 if enable_lora else 0,
            max_lora_rank=64 if enable_lora else 0,
            trust_remote_code=True
        )

        # Adapter registry
        self.loaded_adapters: Dict[str, LoRARequest] = {}
        self._next_lora_id = 1

    def register_adapter(self, name: str, adapter_path: str) -> None:
        """Register a LoRA adapter.

        Args:
            name: Adapter name.
            adapter_path: Path to adapter weights.
        """
        if not self.enable_lora:
            raise RuntimeError("LoRA not enabled")

        lora_request = LoRARequest(
            lora_name=name,
            lora_int_id=self._next_lora_id,
            lora_local_path=adapter_path
        )

        self.loaded_adapters[name] = lora_request
        self._next_lora_id += 1

        logger.info(f"Registered adapter: {name}")

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> GenerationResult:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.
            config: Generation configuration.
            adapter: Specific adapter to use.

        Returns:
            GenerationResult with generated text.
        """
        import time

        start_time = time.time()

        # Get sampling params
        sampling_params = config.to_vllm_params()

        # Get LoRA request if specified
        lora_request = None
        if adapter and adapter in self.loaded_adapters:
            lora_request = self.loaded_adapters[adapter]

        # Generate
        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request
        )

        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=generated_text,
            tokens_generated=len(output.outputs[0].token_ids),
            finish_reason=finish_reason,
            adapter_used=adapter,
            latency_ms=latency_ms
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            config: Generation configuration.
            adapter: Specific adapter to use.

        Returns:
            List of GenerationResults.
        """
        import time

        start_time = time.time()

        sampling_params = config.to_vllm_params()

        lora_request = None
        if adapter and adapter in self.loaded_adapters:
            lora_request = self.loaded_adapters[adapter]

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )

        total_latency = (time.time() - start_time) * 1000
        per_request_latency = total_latency / len(prompts)

        results = []
        for output in outputs:
            results.append(GenerationResult(
                text=output.outputs[0].text,
                tokens_generated=len(output.outputs[0].token_ids),
                finish_reason=output.outputs[0].finish_reason,
                adapter_used=adapter,
                latency_ms=per_request_latency
            ))

        return results

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated text.

        Args:
            prompt: Input prompt.
            config: Generation configuration.
            adapter: Specific adapter to use.

        Yields:
            Generated text chunks.
        """
        # vLLM streaming
        sampling_params = config.to_vllm_params()

        lora_request = None
        if adapter and adapter in self.loaded_adapters:
            lora_request = self.loaded_adapters[adapter]

        # Note: Actual streaming requires vLLM's async engine
        # This is a simplified implementation
        result = self.generate(prompt, config, adapter)
        words = result.text.split()

        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)


class TransformersInferenceEngine(InferenceEngine):
    """Fallback inference engine using HuggingFace Transformers.

    Used when vLLM is not available or for debugging.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        """Initialize Transformers engine.

        Args:
            model_name: Base model name/path.
            device: Target device.
            load_in_4bit: Use 4-bit quantization.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = device
        self.model_name = model_name

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if load_in_4bit else None

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loaded adapters
        self.loaded_adapters: Dict[str, Any] = {}

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> GenerationResult:
        """Generate text from prompt."""
        import time
        import torch

        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=generated_text,
            tokens_generated=len(generated_ids),
            finish_reason="stop",
            adapter_used=adapter,
            latency_ms=latency_ms
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        return [
            self.generate(prompt, config, adapter)
            for prompt in prompts
        ]

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        adapter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated text."""
        result = self.generate(prompt, config, adapter)
        words = result.text.split()

        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)


def create_inference_engine(
    model_name: str,
    use_vllm: bool = True,
    **kwargs
) -> InferenceEngine:
    """Create appropriate inference engine.

    Args:
        model_name: Base model name/path.
        use_vllm: Use vLLM if available.
        **kwargs: Additional arguments.

    Returns:
        Configured inference engine.
    """
    if use_vllm and VLLM_AVAILABLE:
        return VLLMInferenceEngine(model_name, **kwargs)
    else:
        return TransformersInferenceEngine(model_name, **kwargs)
