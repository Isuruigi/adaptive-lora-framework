#!/usr/bin/env python3
"""
Data Preparation Script

Prepare training data for the Adaptive LoRA system:
- Download and preprocess datasets
- Split into train/validation/test
- Generate synthetic data
- Validate and clean data
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import (
    AdaptiveDataLoader, DataDeduplicator, DataValidator, DataCleaner, DataPipeline
)
from src.data.synthetic_generator import SyntheticDataGenerator, DifficultyEscalator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_dataset(name: str, output_dir: Path) -> Path:
    """Download a dataset from HuggingFace Hub."""
    from datasets import load_dataset
    
    logger.info(f"Downloading dataset: {name}")
    
    # Map common dataset names
    dataset_mapping = {
        'alpaca': 'tatsu-lab/alpaca',
        'dolly': 'databricks/databricks-dolly-15k',
        'openassistant': 'OpenAssistant/oasst1',
        'code_alpaca': 'sahil2801/CodeAlpaca-20k',
        'math': 'gsm8k',
    }
    
    hf_name = dataset_mapping.get(name, name)
    
    try:
        dataset = load_dataset(hf_name)
        
        output_path = output_dir / f"{name}.jsonl"
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for split in dataset:
                for item in dataset[split]:
                    f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {name} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        raise


def split_dataset(
    data_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Path]:
    """Split dataset into train/validation/test."""
    random.seed(seed)
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Shuffle
    random.shuffle(data)
    
    # Calculate split indices
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save splits
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    output_paths = {}
    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        output_paths[split_name] = output_path
        logger.info(f"Saved {split_name}: {len(split_data)} examples")
    
    return output_paths


def prepare_adapter_data(
    raw_data_dir: Path,
    output_dir: Path,
    adapter_configs: Dict[str, Dict]
) -> None:
    """Prepare data for each adapter type."""
    for adapter_name, config in adapter_configs.items():
        adapter_dir = output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing data for adapter: {adapter_name}")
        
        # Get source data
        source = config.get('source')
        if not source:
            logger.warning(f"No source specified for {adapter_name}")
            continue
        
        source_path = raw_data_dir / source
        if not source_path.exists():
            logger.warning(f"Source not found: {source_path}")
            continue
        
        # Filter and process
        filter_fn = config.get('filter')
        
        with open(source_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # Apply filtering if specified
        if filter_fn:
            # In practice, this would apply task-specific filtering
            pass
        
        # Split
        split_dataset(
            data_path=source_path,
            output_dir=adapter_dir,
            train_ratio=config.get('train_ratio', 0.8),
            val_ratio=config.get('val_ratio', 0.1),
            test_ratio=config.get('test_ratio', 0.1)
        )


def generate_router_data(
    adapter_data_dirs: Dict[str, Path],
    output_dir: Path,
    samples_per_adapter: int = 1000
) -> None:
    """Generate training data for the router."""
    router_dir = output_dir / 'router'
    router_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    for adapter_name, data_dir in adapter_data_dirs.items():
        train_path = data_dir / 'train.jsonl'
        
        if not train_path.exists():
            logger.warning(f"No training data for {adapter_name}")
            continue
        
        with open(train_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # Sample and label
        sampled = random.sample(data, min(samples_per_adapter, len(data)))
        
        for item in sampled:
            all_samples.append({
                'instruction': item.get('instruction', ''),
                'input': item.get('input', ''),
                'adapter': adapter_name
            })
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split
    n = len(all_samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        'train': all_samples[:train_end],
        'eval': all_samples[train_end:val_end],
        'test': all_samples[val_end:]
    }
    
    for split_name, split_data in splits.items():
        output_path = router_dir / f"{split_name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Router {split_name}: {len(split_data)} examples")


def validate_data(data_dir: Path) -> Dict[str, Any]:
    """Validate prepared data."""
    stats = {}
    
    for data_file in data_dir.rglob('*.jsonl'):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        relative_path = data_file.relative_to(data_dir)
        stats[str(relative_path)] = {
            'count': len(data),
            'fields': list(data[0].keys()) if data else [],
            'avg_instruction_length': sum(
                len(d.get('instruction', '')) for d in data
            ) / len(data) if data else 0
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare Training Data')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_prep.yaml',
        help='Data preparation configuration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download datasets'
    )
    parser.add_argument(
        '--generate-synthetic',
        action='store_true',
        help='Generate synthetic data'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate prepared data'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    # Load config if exists
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Step 1: Download datasets
    if args.download:
        datasets = config.get('datasets', ['alpaca', 'dolly'])
        for dataset_name in datasets:
            try:
                download_dataset(dataset_name, raw_dir)
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
    
    # Step 2: Prepare adapter data
    adapter_configs = config.get('adapters', {
        'reasoning': {'source': 'alpaca.jsonl'},
        'code': {'source': 'code_alpaca.jsonl'},
        'analysis': {'source': 'alpaca.jsonl'},
        'base': {'source': 'dolly.jsonl'},
    })
    
    prepare_adapter_data(raw_dir, output_dir, adapter_configs)
    
    # Step 3: Generate router data
    adapter_dirs = {
        name: output_dir / name
        for name in adapter_configs.keys()
    }
    generate_router_data(adapter_dirs, output_dir)
    
    # Step 4: Generate synthetic data
    if args.generate_synthetic:
        logger.info("Generating synthetic data...")
        # generator = SyntheticDataGenerator(...)
        # generator.generate_batch(...)
    
    # Step 5: Validate
    if args.validate:
        logger.info("Validating prepared data...")
        stats = validate_data(output_dir)
        
        logger.info("\nData Statistics:")
        for path, info in stats.items():
            logger.info(f"  {path}: {info['count']} examples")
        
        # Save stats
        with open(output_dir / 'data_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    logger.info(f"\nData preparation complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
