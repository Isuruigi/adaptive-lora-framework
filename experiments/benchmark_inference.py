#!/usr/bin/env python3
"""
Inference Benchmark Script

Benchmark inference performance under various conditions:
- Different batch sizes
- Different sequence lengths
- With/without caching
- Different quantization settings
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import statistics

import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    config_name: str
    batch_size: int
    sequence_length: int
    num_iterations: int
    
    # Latency metrics (ms)
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float
    
    # Throughput metrics
    throughput_samples_per_sec: float
    throughput_tokens_per_sec: float
    
    # Memory metrics (GB)
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    
    # Configuration
    quantization: str
    caching_enabled: bool
    flash_attention: bool


class InferenceBenchmark:
    """Benchmark inference performance."""
    
    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> None:
        """Load model with configuration."""
        logger.info(f"Loading model from {self.model_path}")
        # Load model with specified quantization
        # self.model = AutoModelForCausalLM.from_pretrained(...)
        # self.tokenizer = AutoTokenizer.from_pretrained(...)
    
    def generate_test_inputs(
        self,
        batch_size: int,
        sequence_length: int
    ) -> Dict[str, torch.Tensor]:
        """Generate test input tensors."""
        # Generate random input IDs
        input_ids = torch.randint(
            100, 30000,
            (batch_size, sequence_length),
            device=self.device
        )
        
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def warmup(self, num_iterations: int = 10) -> None:
        """Warmup the model."""
        logger.info("Warming up...")
        inputs = self.generate_test_inputs(batch_size=1, sequence_length=128)
        
        for _ in range(num_iterations):
            with torch.no_grad():
                # _ = self.model(**inputs)
                pass
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    def run_benchmark(
        self,
        batch_size: int,
        sequence_length: int,
        num_iterations: int = 100,
        max_new_tokens: int = 128
    ) -> BenchmarkResult:
        """Run benchmark with specific configuration."""
        inputs = self.generate_test_inputs(batch_size, sequence_length)
        latencies = []
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        for _ in tqdm(range(num_iterations), desc=f"Batch={batch_size}, Seq={sequence_length}"):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            with torch.no_grad():
                # outputs = self.model.generate(
                #     **inputs,
                #     max_new_tokens=max_new_tokens,
                #     do_sample=False
                # )
                pass
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
        
        # Memory stats
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1e9
            memory_reserved = torch.cuda.max_memory_reserved() / 1e9
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Calculate metrics
        total_samples = batch_size * num_iterations
        total_tokens = total_samples * (sequence_length + max_new_tokens)
        total_time = sum(latencies) / 1000  # seconds
        
        # Use placeholder values if latencies is empty
        if not latencies:
            latencies = [0.0]
        
        return BenchmarkResult(
            config_name=f"batch{batch_size}_seq{sequence_length}",
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_iterations=num_iterations,
            latency_mean=statistics.mean(latencies),
            latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            latency_p50=statistics.median(latencies),
            latency_p95=sorted(latencies)[int(0.95 * len(latencies))],
            latency_p99=sorted(latencies)[int(0.99 * len(latencies))],
            latency_min=min(latencies),
            latency_max=max(latencies),
            throughput_samples_per_sec=total_samples / total_time if total_time > 0 else 0,
            throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
            gpu_memory_allocated=memory_allocated,
            gpu_memory_reserved=memory_reserved,
            quantization=self.config.get('quantization', 'none'),
            caching_enabled=self.config.get('caching_enabled', False),
            flash_attention=self.config.get('flash_attention', False)
        )
    
    def run_full_benchmark(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        sequence_lengths: List[int] = [128, 256, 512, 1024, 2048],
        num_iterations: int = 50
    ) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite."""
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                try:
                    result = self.run_benchmark(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        num_iterations=num_iterations
                    )
                    results.append(result)
                    
                    logger.info(
                        f"Batch={batch_size}, Seq={seq_len}: "
                        f"Latency={result.latency_p50:.2f}ms, "
                        f"Throughput={result.throughput_samples_per_sec:.1f} samples/s"
                    )
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM at batch={batch_size}, seq={seq_len}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Inference Performance')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/adapters',
        help='Path to model/adapters'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system.yaml',
        help='System configuration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/benchmark_results.json',
        help='Output path for results'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--sequence-lengths',
        type=int,
        nargs='+',
        default=[128, 256, 512, 1024],
        help='Sequence lengths to test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Number of iterations per configuration'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create benchmark
    benchmark = InferenceBenchmark(
        model_path=args.model_path,
        config=config
    )
    
    # Run benchmarks
    logger.info("Starting benchmarks...")
    results = benchmark.run_full_benchmark(
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        num_iterations=args.iterations
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<25} {'Latency P50':<15} {'Throughput':<20} {'Memory':<15}")
    logger.info("-" * 70)
    
    for r in results:
        logger.info(
            f"{r.config_name:<25} "
            f"{r.latency_p50:>10.2f} ms   "
            f"{r.throughput_samples_per_sec:>15.1f} s/s   "
            f"{r.gpu_memory_allocated:>10.2f} GB"
        )
    
    logger.info("=" * 70)
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
