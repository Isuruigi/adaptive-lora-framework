#!/usr/bin/env python3
"""
Hyperparameter Search

Automated hyperparameter optimization using Optuna for:
- LoRA parameters (r, alpha, dropout)
- Training parameters (lr, batch size, etc.)
- Router architecture
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Run: pip install optuna")


@dataclass
class SearchResult:
    """Container for search results."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    search_space: Dict[str, Any]


class HyperparameterSearch:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Path,
        study_name: str = "adaptive_lora_hpo"
    ):
        self.base_config = base_config
        self.output_dir = output_dir
        self.study_name = study_name
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter search")
    
    def create_lora_objective(self):
        """Create objective function for LoRA hyperparameters."""
        def objective(trial: optuna.Trial) -> float:
            # Sample LoRA hyperparameters
            lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32, 64])
            lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 64, 128])
            lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.2, step=0.05)
            
            # Sample training hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
            warmup_ratio = trial.suggest_float('warmup_ratio', 0.01, 0.1)
            weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
            
            # Create config
            config = {
                'lora': {
                    'r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                },
                'training': {
                    'learning_rate': learning_rate,
                    'per_device_train_batch_size': batch_size,
                    'warmup_ratio': warmup_ratio,
                    'weight_decay': weight_decay,
                }
            }
            
            # Train and evaluate (placeholder)
            # In practice, this would:
            # 1. Create trainer with these hyperparameters
            # 2. Train for a few epochs
            # 3. Evaluate on validation set
            # 4. Return the metric to optimize
            
            eval_loss = 0.5 + 0.1 * trial.number / 100  # Placeholder
            
            return eval_loss
        
        return objective
    
    def create_router_objective(self):
        """Create objective function for router hyperparameters."""
        def objective(trial: optuna.Trial) -> float:
            # Sample router hyperparameters
            hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 4)
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            
            # Sample training hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            config = {
                'model': {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'dropout': dropout,
                },
                'training': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                }
            }
            
            # Train and evaluate (placeholder)
            accuracy = 0.8 + 0.01 * trial.number / 100  # Placeholder
            
            return 1 - accuracy  # Minimize (1 - accuracy)
        
        return objective
    
    def run_search(
        self,
        search_type: str = 'lora',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> SearchResult:
        """Run hyperparameter search."""
        logger.info(f"Starting {search_type} hyperparameter search with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{search_type}",
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Select objective
        if search_type == 'lora':
            objective = self.create_lora_objective()
            search_space = {
                'lora_r': [4, 8, 16, 32, 64],
                'lora_alpha': [8, 16, 32, 64, 128],
                'lora_dropout': [0.0, 0.2],
                'learning_rate': [1e-5, 5e-4],
                'batch_size': [2, 4, 8, 16],
            }
        elif search_type == 'router':
            objective = self.create_router_objective()
            search_space = {
                'hidden_dim': [128, 256, 512],
                'num_layers': [1, 4],
                'num_heads': [2, 4, 8],
                'dropout': [0.0, 0.3],
                'learning_rate': [1e-5, 1e-3],
            }
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Create result
        result = SearchResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            study_name=study.study_name,
            search_space=search_space
        )
        
        # Save study
        self._save_study(study, search_type)
        
        return result
    
    def _save_study(self, study: optuna.Study, search_type: str) -> None:
        """Save study results and visualizations."""
        study_dir = self.output_dir / search_type
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        with open(study_dir / 'best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save all trials
        trials_data = [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'state': t.state.name
            }
            for t in study.trials
        ]
        
        with open(study_dir / 'all_trials.json', 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Generate visualizations
        try:
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_html(str(study_dir / 'optimization_history.html'))
            
            # Parameter importances
            fig = plot_param_importances(study)
            fig.write_html(str(study_dir / 'param_importances.html'))
            
            # Parallel coordinate
            fig = plot_parallel_coordinate(study)
            fig.write_html(str(study_dir / 'parallel_coordinate.html'))
            
            logger.info(f"Visualizations saved to {study_dir}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search')
    parser.add_argument(
        '--search-type',
        type=str,
        default='lora',
        choices=['lora', 'router'],
        help='Type of hyperparameters to search'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system.yaml',
        help='Base configuration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/hpo',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run search
    searcher = HyperparameterSearch(base_config, output_dir)
    
    result = searcher.run_search(
        search_type=args.search_type,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs
    )
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETER SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best value: {result.best_value:.4f}")
    logger.info(f"Total trials: {result.n_trials}")
    logger.info("\nBest parameters:")
    for param, value in result.best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("=" * 60)
    
    # Save final result
    with open(output_dir / f'{args.search_type}_result.json', 'w') as f:
        json.dump(asdict(result), f, indent=2)


if __name__ == '__main__':
    main()
