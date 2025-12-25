#!/usr/bin/env python3
"""
Ablation Studies

Systematic experiments to understand the contribution of each component:
- Router architecture variations
- LoRA configuration impact
- Active learning strategies
- Training data size effects
"""

import argparse
import json
import itertools
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import copy

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Container for ablation study results."""
    study_name: str
    variant_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    training_time_seconds: float


class AblationStudy:
    """Run ablation studies on system components."""
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Path
    ):
        self.base_config = base_config
        self.output_dir = output_dir
        self.results: List[AblationResult] = []
    
    def run_single_experiment(
        self,
        config: Dict[str, Any],
        variant_name: str
    ) -> Dict[str, float]:
        """Run a single experiment and return metrics."""
        logger.info(f"Running experiment: {variant_name}")
        
        # Placeholder - in practice this would:
        # 1. Train the model with the given config
        # 2. Evaluate on validation set
        # 3. Return metrics
        
        metrics = {
            'accuracy': 0.85,
            'latency_ms': 100.0,
            'quality_score': 0.80,
        }
        
        return metrics
    
    def ablate_router_architecture(self) -> List[AblationResult]:
        """Ablation study on router architecture."""
        study_name = "router_architecture"
        results = []
        
        architectures = [
            {'type': 'standard', 'hidden_dim': 256, 'num_layers': 2},
            {'type': 'hierarchical', 'hidden_dim': 256, 'num_layers': 2},
            {'type': 'dynamic', 'hidden_dim': 256, 'num_layers': 2},
            {'type': 'standard', 'hidden_dim': 128, 'num_layers': 2},
            {'type': 'standard', 'hidden_dim': 512, 'num_layers': 2},
            {'type': 'standard', 'hidden_dim': 256, 'num_layers': 1},
            {'type': 'standard', 'hidden_dim': 256, 'num_layers': 4},
        ]
        
        for arch in architectures:
            config = copy.deepcopy(self.base_config)
            config['model'] = {**config.get('model', {}), **arch}
            
            variant_name = f"{arch['type']}_h{arch['hidden_dim']}_l{arch['num_layers']}"
            
            metrics = self.run_single_experiment(config, variant_name)
            
            results.append(AblationResult(
                study_name=study_name,
                variant_name=variant_name,
                config=arch,
                metrics=metrics,
                training_time_seconds=0.0
            ))
        
        return results
    
    def ablate_lora_config(self) -> List[AblationResult]:
        """Ablation study on LoRA configuration."""
        study_name = "lora_config"
        results = []
        
        configs = [
            {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05},
            {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05},
            {'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05},
            {'r': 64, 'lora_alpha': 128, 'lora_dropout': 0.05},
            {'r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05},
            {'r': 16, 'lora_alpha': 64, 'lora_dropout': 0.05},
            {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.0},
            {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1},
        ]
        
        for lora_cfg in configs:
            config = copy.deepcopy(self.base_config)
            config['lora'] = {**config.get('lora', {}), **lora_cfg}
            
            variant_name = f"r{lora_cfg['r']}_a{lora_cfg['lora_alpha']}_d{lora_cfg['lora_dropout']}"
            
            metrics = self.run_single_experiment(config, variant_name)
            
            results.append(AblationResult(
                study_name=study_name,
                variant_name=variant_name,
                config=lora_cfg,
                metrics=metrics,
                training_time_seconds=0.0
            ))
        
        return results
    
    def ablate_active_learning(self) -> List[AblationResult]:
        """Ablation study on active learning strategies."""
        study_name = "active_learning"
        results = []
        
        strategies = [
            {'enabled': False, 'strategy': None},
            {'enabled': True, 'strategy': 'random'},
            {'enabled': True, 'strategy': 'uncertainty', 'method': 'entropy'},
            {'enabled': True, 'strategy': 'uncertainty', 'method': 'margin'},
            {'enabled': True, 'strategy': 'uncertainty', 'method': 'least_confidence'},
            {'enabled': True, 'strategy': 'diversity', 'method': 'kmeans'},
            {'enabled': True, 'strategy': 'diversity', 'method': 'coreset'},
            {'enabled': True, 'strategy': 'hybrid', 'uncertainty_weight': 0.5},
            {'enabled': True, 'strategy': 'hybrid', 'uncertainty_weight': 0.7},
        ]
        
        for al_cfg in strategies:
            config = copy.deepcopy(self.base_config)
            config['active_learning'] = {**config.get('active_learning', {}), **al_cfg}
            
            if not al_cfg['enabled']:
                variant_name = "no_active_learning"
            else:
                method = al_cfg.get('method', al_cfg.get('uncertainty_weight', ''))
                variant_name = f"{al_cfg['strategy']}_{method}"
            
            metrics = self.run_single_experiment(config, variant_name)
            
            results.append(AblationResult(
                study_name=study_name,
                variant_name=variant_name,
                config=al_cfg,
                metrics=metrics,
                training_time_seconds=0.0
            ))
        
        return results
    
    def ablate_data_size(self) -> List[AblationResult]:
        """Ablation study on training data size."""
        study_name = "data_size"
        results = []
        
        data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        for fraction in data_fractions:
            config = copy.deepcopy(self.base_config)
            config['data'] = {**config.get('data', {}), 'train_fraction': fraction}
            
            variant_name = f"data_{int(fraction * 100)}pct"
            
            metrics = self.run_single_experiment(config, variant_name)
            
            results.append(AblationResult(
                study_name=study_name,
                variant_name=variant_name,
                config={'train_fraction': fraction},
                metrics=metrics,
                training_time_seconds=0.0
            ))
        
        return results
    
    def run_all_studies(self) -> Dict[str, List[AblationResult]]:
        """Run all ablation studies."""
        all_results = {}
        
        logger.info("Running router architecture ablation...")
        all_results['router_architecture'] = self.ablate_router_architecture()
        
        logger.info("Running LoRA config ablation...")
        all_results['lora_config'] = self.ablate_lora_config()
        
        logger.info("Running active learning ablation...")
        all_results['active_learning'] = self.ablate_active_learning()
        
        logger.info("Running data size ablation...")
        all_results['data_size'] = self.ablate_data_size()
        
        return all_results
    
    def save_results(self, results: Dict[str, List[AblationResult]]) -> None:
        """Save ablation results."""
        output_path = self.output_dir / 'ablation_results.json'
        
        serializable = {
            study: [asdict(r) for r in study_results]
            for study, study_results in results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, List[AblationResult]]) -> None:
        """Print summary of ablation results."""
        for study_name, study_results in results.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STUDY: {study_name}")
            logger.info(f"{'=' * 60}")
            
            # Find best variant
            best = max(study_results, key=lambda x: x.metrics.get('accuracy', 0))
            
            for r in study_results:
                marker = " <-- BEST" if r == best else ""
                logger.info(
                    f"{r.variant_name:<30} "
                    f"Acc: {r.metrics.get('accuracy', 0):.3f} "
                    f"Latency: {r.metrics.get('latency_ms', 0):.1f}ms"
                    f"{marker}"
                )


def main():
    parser = argparse.ArgumentParser(description='Run Ablation Studies')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system.yaml',
        help='Base configuration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/ablation',
        help='Output directory'
    )
    parser.add_argument(
        '--studies',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'router', 'lora', 'active_learning', 'data_size'],
        help='Which studies to run'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study runner
    study = AblationStudy(base_config, output_dir)
    
    # Run studies
    if 'all' in args.studies:
        results = study.run_all_studies()
    else:
        results = {}
        if 'router' in args.studies:
            results['router_architecture'] = study.ablate_router_architecture()
        if 'lora' in args.studies:
            results['lora_config'] = study.ablate_lora_config()
        if 'active_learning' in args.studies:
            results['active_learning'] = study.ablate_active_learning()
        if 'data_size' in args.studies:
            results['data_size'] = study.ablate_data_size()
    
    # Save and print results
    study.save_results(results)
    study.print_summary(results)


if __name__ == '__main__':
    main()
