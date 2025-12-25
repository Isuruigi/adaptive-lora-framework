"""
Training experiment runner.

Usage:
    python experiments/train_adapters.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.adapters.lora_trainer import LoRATrainer, MultiAdapterTrainer
from src.config.base_config import load_config
from src.data.data_loader import AdaptiveDataLoader
from src.utils.logger import get_logger
from src.utils.helpers import seed_everything

logger = get_logger(__name__)


def run_single_adapter_experiment(
    config_path: Path,
    adapter_name: str,
    data_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Run single adapter training experiment.

    Args:
        config_path: Path to configuration file.
        adapter_name: Name for the adapter.
        data_path: Path to training data.
        output_dir: Output directory.

    Returns:
        Experiment results.
    """
    logger.info(f"Starting experiment: {adapter_name}")

    # Load configuration
    config = load_config(config_path)
    seed_everything(config.training.seed)

    # Create output directory
    experiment_dir = output_dir / f"{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = LoRATrainer(
        model_name=config.model.model_name,
        output_dir=experiment_dir,
        lora_config=config.lora.model_dump(),
        training_config=config.training.model_dump(),
        use_4bit=config.model.load_in_4bit,
        use_wandb=config.monitoring.use_wandb,
        wandb_project=config.monitoring.wandb_project,
        wandb_run_name=f"{adapter_name}_{datetime.now().strftime('%Y%m%d')}"
    )

    # Load and preprocess data
    data_loader = AdaptiveDataLoader(
        trainer.tokenizer,
        max_length=config.training.max_seq_length
    )
    dataset = data_loader.load(data_path)

    train_dataset = dataset["train"].map(
        data_loader.preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = dataset["validation"].map(
            data_loader.preprocess_function,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )

    # Train
    result = trainer.train(train_dataset, eval_dataset)

    # Save results
    results = {
        "adapter_name": adapter_name,
        "config": config.to_dict(),
        "metrics": result.metrics if hasattr(result, "metrics") else {},
        "output_dir": str(experiment_dir),
        "timestamp": datetime.now().isoformat()
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Experiment complete: {experiment_dir}")

    return results


def run_multi_adapter_experiment(
    config_path: Path,
    adapters_config: Dict[str, Dict],
    output_dir: Path
) -> Dict[str, Any]:
    """Run multi-adapter training experiment.

    Args:
        config_path: Path to base configuration.
        adapters_config: Configuration for each adapter.
        output_dir: Output directory.

    Returns:
        Combined experiment results.
    """
    logger.info("Starting multi-adapter experiment")

    config = load_config(config_path)
    seed_everything(config.training.seed)

    # Create output directory
    experiment_dir = output_dir / f"multi_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize multi-adapter trainer
    trainer = MultiAdapterTrainer(
        base_model=config.model.model_name,
        output_dir=experiment_dir,
        adapters_config=adapters_config
    )

    # Train all adapters
    results = trainer.train_all_adapters()

    # Save combined results
    combined_results = {
        "experiment_type": "multi_adapter",
        "base_config": config.to_dict(),
        "adapter_results": results,
        "output_dir": str(experiment_dir),
        "timestamp": datetime.now().isoformat()
    }

    with open(experiment_dir / "combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2, default=str)

    logger.info(f"Multi-adapter experiment complete: {experiment_dir}")

    return combined_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run training experiments")

    parser.add_argument(
        "--config",
        type=str,
        default="src/config/model_configs.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="reasoning",
        help="Adapter name"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments",
        help="Output directory"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run multi-adapter experiment"
    )

    args = parser.parse_args()

    if args.multi:
        # Example multi-adapter config
        adapters_config = {
            "reasoning": {
                "data_path": args.data,
                "lora_config": {"r": 16, "lora_alpha": 32},
                "training_config": {"num_train_epochs": 3}
            },
            "code": {
                "data_path": args.data,
                "lora_config": {"r": 16, "lora_alpha": 32},
                "training_config": {"num_train_epochs": 3}
            }
        }

        results = run_multi_adapter_experiment(
            config_path=Path(args.config),
            adapters_config=adapters_config,
            output_dir=Path(args.output)
        )
    else:
        results = run_single_adapter_experiment(
            config_path=Path(args.config),
            adapter_name=args.adapter,
            data_path=Path(args.data),
            output_dir=Path(args.output)
        )

    print(f"\nResults saved to: {results.get('output_dir')}")


if __name__ == "__main__":
    main()
