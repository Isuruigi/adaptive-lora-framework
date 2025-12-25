"""
Production-ready LoRA/QLoRA trainer with comprehensive features.

Features:
- QLoRA support (4-bit quantization)
- Gradient checkpointing
- Mixed precision training
- W&B experiment tracking
- Multi-adapter training
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.helpers import count_parameters, format_number
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class LoRATrainer:
    """Production-ready LoRA/QLoRA trainer.

    Provides comprehensive training capabilities for LoRA adapters
    including 4-bit quantization, gradient checkpointing, and
    experiment tracking.

    Example:
        >>> trainer = LoRATrainer(
        ...     model_name="meta-llama/Llama-3-8B",
        ...     output_dir=Path("./outputs/reasoning"),
        ...     lora_config={'r': 16, 'lora_alpha': 32},
        ...     training_config={'num_train_epochs': 3}
        ... )
        >>> trainer.train(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        use_4bit: bool = True,
        use_8bit: bool = False,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ):
        """Initialize LoRA trainer.

        Args:
            model_name: HuggingFace model ID or local path.
            output_dir: Directory for saving outputs.
            lora_config: LoRA configuration dictionary.
            training_config: Training hyperparameters dictionary.
            use_4bit: Use 4-bit quantization (QLoRA).
            use_8bit: Use 8-bit quantization.
            use_wandb: Enable W&B logging.
            wandb_project: W&B project name.
            wandb_run_name: W&B run name.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.lora_config = lora_config
        self.training_config = training_config
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.use_wandb and wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={**lora_config, **training_config}
                )
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Setup model and tokenizer
        self.model, self.tokenizer = self._setup_model()

    def _setup_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and prepare model for LoRA training.

        Returns:
            Tuple of (model, tokenizer).
        """
        logger.info(f"Loading model: {self.model_name}")

        # Quantization config
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not (self.use_4bit or self.use_8bit) else None,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Prepare for k-bit training
        if self.use_4bit or self.use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            target_modules=self.lora_config.get(
                "target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias=self.lora_config.get("bias", "none"),
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        # Log parameter counts
        self._log_trainable_parameters(model)

        return model, tokenizer

    def _log_trainable_parameters(self, model: PreTrainedModel) -> None:
        """Log number of trainable parameters.

        Args:
            model: The PEFT model.
        """
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        percentage = 100 * trainable / total if total > 0 else 0

        logger.info(
            f"Trainable params: {format_number(trainable)} | "
            f"All params: {format_number(total)} | "
            f"Trainable%: {percentage:.2f}%"
        )

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        callbacks: Optional[List] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Trainer:
        """Train LoRA adapter.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Validation dataset.
            callbacks: Optional training callbacks.
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            HuggingFace Trainer object.
        """
        logger.info("Preparing training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.training_config.get(
                "per_device_train_batch_size", 4
            ),
            per_device_eval_batch_size=self.training_config.get(
                "per_device_eval_batch_size", 4
            ),
            gradient_accumulation_steps=self.training_config.get(
                "gradient_accumulation_steps", 8
            ),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.03),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 100),
            eval_steps=self.training_config.get("eval_steps", 100) if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.training_config.get("save_total_limit", 3),
            load_best_model_at_end=True if eval_dataset else False,
            bf16=self.training_config.get("bf16", True),
            fp16=self.training_config.get("fp16", False),
            gradient_checkpointing=True,
            report_to="wandb" if self.use_wandb else "none",
            save_strategy="steps",
            optim="paged_adamw_8bit" if (self.use_4bit or self.use_8bit) else "adamw_torch",
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            seed=self.training_config.get("seed", 42),
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )

        # Train
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        self._save_model(trainer)

        return trainer

    def _save_model(self, trainer: Trainer) -> None:
        """Save trained adapter.

        Args:
            trainer: HuggingFace Trainer instance.
        """
        output_path = self.output_dir / "final_adapter"

        # Save adapter
        trainer.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save config
        config = {
            "model_name": self.model_name,
            "lora_config": self.lora_config,
            "training_config": self.training_config,
            "use_4bit": self.use_4bit,
            "use_8bit": self.use_8bit,
        }

        with open(self.output_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Model saved to {output_path}")

        # Log to W&B
        if self.use_wandb:
            try:
                artifact = wandb.Artifact(
                    name=f"lora_adapter_{wandb.run.id}",
                    type="model"
                )
                artifact.add_dir(str(output_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log artifact to W&B: {e}")

    @classmethod
    def load_adapter(
        cls,
        base_model_name: str,
        adapter_path: Union[str, Path],
        device_map: str = "auto",
        use_4bit: bool = True
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load trained LoRA adapter.

        Args:
            base_model_name: Base model name/path.
            adapter_path: Path to adapter weights.
            device_map: Device mapping strategy.
            use_4bit: Use 4-bit quantization.

        Returns:
            Tuple of (model, tokenizer).
        """
        adapter_path = Path(adapter_path)
        logger.info(f"Loading adapter from {adapter_path}")

        # Quantization config
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )

        # Load adapter
        model = PeftModel.from_pretrained(model, str(adapter_path))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        return model, tokenizer

    def evaluate(self, eval_dataset, metrics_fn=None) -> Dict[str, float]:
        """Evaluate the trained model.

        Args:
            eval_dataset: Evaluation dataset.
            metrics_fn: Optional custom metrics function.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()

        # Use Trainer for evaluation
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "eval"),
            per_device_eval_batch_size=self.training_config.get(
                "per_device_eval_batch_size", 4
            ),
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")

        return metrics


class MultiAdapterTrainer:
    """Train multiple specialized adapters.

    Provides utilities for training and managing multiple LoRA
    adapters for different tasks.

    Example:
        >>> trainer = MultiAdapterTrainer(
        ...     base_model="meta-llama/Llama-3-8B",
        ...     output_dir=Path("./outputs"),
        ...     adapters_config={
        ...         'reasoning': {'data_path': '...', 'lora_config': {...}},
        ...         'code': {'data_path': '...', 'lora_config': {...}},
        ...     }
        ... )
        >>> results = trainer.train_all_adapters()
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Path,
        adapters_config: Dict[str, Dict[str, Any]]
    ):
        """Initialize multi-adapter trainer.

        Args:
            base_model: Base model name/path.
            output_dir: Root output directory.
            adapters_config: Configuration for each adapter.
                {
                    'adapter_name': {
                        'data_path': str,
                        'lora_config': dict,
                        'training_config': dict
                    }
                }
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.adapters_config = adapters_config

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_all_adapters(self) -> Dict[str, Dict[str, Any]]:
        """Train all configured adapters.

        Returns:
            Dictionary of results for each adapter.
        """
        results = {}

        for adapter_name, config in self.adapters_config.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training adapter: {adapter_name}")
            logger.info(f"{'='*60}\n")

            try:
                result = self._train_single_adapter(adapter_name, config)
                results[adapter_name] = {
                    "status": "success",
                    "output_dir": str(self.output_dir / adapter_name),
                    "metrics": result
                }
            except Exception as e:
                logger.error(f"Failed to train {adapter_name}: {e}")
                results[adapter_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Save summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def _train_single_adapter(
        self,
        adapter_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train a single adapter.

        Args:
            adapter_name: Name of the adapter.
            config: Adapter configuration.

        Returns:
            Training metrics.
        """
        from src.data.data_loader import AdaptiveDataLoader

        adapter_output_dir = self.output_dir / adapter_name

        # Initialize trainer
        trainer = LoRATrainer(
            model_name=self.base_model,
            output_dir=adapter_output_dir,
            lora_config=config.get("lora_config", {}),
            training_config=config.get("training_config", {}),
            use_wandb=True,
            wandb_project=f"multi-adapter-{adapter_name}"
        )

        # Load data
        data_loader = AdaptiveDataLoader(trainer.tokenizer)
        dataset = data_loader.load(config["data_path"])

        # Preprocess
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

        return result.metrics if hasattr(result, "metrics") else {}

    def train_sequential(
        self,
        adapter_order: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Train adapters in a specific order.

        Useful for curriculum learning where some adapters
        should be trained before others.

        Args:
            adapter_order: Order to train adapters.

        Returns:
            Dictionary of results.
        """
        if adapter_order is None:
            adapter_order = list(self.adapters_config.keys())

        results = {}

        for adapter_name in adapter_order:
            if adapter_name not in self.adapters_config:
                logger.warning(f"Adapter {adapter_name} not in config, skipping")
                continue

            config = self.adapters_config[adapter_name]
            result = self._train_single_adapter(adapter_name, config)
            results[adapter_name] = result

        return results


class EarlyStoppingCallback:
    """Early stopping callback for training.
    
    Monitors a metric and stops training when no improvement.
    """
    
    def __init__(
        self,
        metric: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        """Initialize early stopping.
        
        Args:
            metric: Metric to monitor.
            patience: Number of evaluations to wait.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, eval_result: Dict[str, float]) -> bool:
        """Check if training should stop.
        
        Args:
            eval_result: Evaluation results dictionary.
            
        Returns:
            True if training should stop.
        """
        current = eval_result.get(self.metric)
        
        if current is None:
            return False
        
        improved = False
        
        if self.mode == "min":
            if current < self.best_value - self.min_delta:
                improved = True
        else:
            if current > self.best_value + self.min_delta:
                improved = True
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} evaluations "
                f"without improvement. Best {self.metric}: {self.best_value:.4f}"
            )
            
        return self.should_stop
    
    def reset(self):
        """Reset callback state."""
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False


class LearningRateFinder:
    """Find optimal learning rate using LR range test.
    
    Implements the learning rate range test from Leslie Smith's paper.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        optimizer_cls=None,
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_steps: int = 100,
        smooth_factor: float = 0.05
    ):
        """Initialize LR finder.
        
        Args:
            model: PyTorch model.
            train_dataloader: Training data loader.
            optimizer_cls: Optimizer class.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
            num_steps: Number of steps for the test.
            smooth_factor: Smoothing factor for loss.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer_cls = optimizer_cls or torch.optim.AdamW
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor
        
        self.lrs = []
        self.losses = []
    
    def find(self) -> Dict[str, Any]:
        """Run learning rate finder.
        
        Returns:
            Dictionary with suggested LR and history.
        """
        # Calculate multiplicative factor
        lr_mult = (self.max_lr / self.min_lr) ** (1 / self.num_steps)
        
        # Setup optimizer with minimum LR
        optimizer = self.optimizer_cls(
            self.model.parameters(),
            lr=self.min_lr
        )
        
        # Save initial state
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        current_lr = self.min_lr
        smoothed_loss = 0
        best_loss = float('inf')
        
        data_iter = iter(self.train_dataloader)
        
        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(self.model.device)
            elif isinstance(batch, dict):
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            else:
                outputs = self.model(batch)
                loss = outputs
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record
            self.lrs.append(current_lr)
            
            # Smooth loss
            if step == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = (
                    self.smooth_factor * loss.item() + 
                    (1 - self.smooth_factor) * smoothed_loss
                )
            
            self.losses.append(smoothed_loss)
            
            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss or not torch.isfinite(loss):
                logger.info(f"Stopping LR finder at step {step}, loss exploded")
                break
            
            # Update LR
            current_lr *= lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Restore initial state
        self.model.load_state_dict(initial_state)
        
        # Find suggested LR
        suggested_lr = self._suggest_lr()
        
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        
        return {
            "suggested_lr": suggested_lr,
            "lrs": self.lrs,
            "losses": self.losses
        }
    
    def _suggest_lr(self) -> float:
        """Suggest optimal learning rate.
        
        Uses the point of steepest descent.
        """
        if len(self.losses) < 3:
            return self.min_lr
        
        # Find minimum loss
        min_loss_idx = self.losses.index(min(self.losses))
        
        # Look for steepest descent before minimum
        gradients = []
        for i in range(1, min_loss_idx):
            grad = (self.losses[i] - self.losses[i-1]) / (self.lrs[i] - self.lrs[i-1])
            gradients.append((i, grad))
        
        if not gradients:
            return self.lrs[min_loss_idx] / 10
        
        # Find steepest negative gradient
        steepest = min(gradients, key=lambda x: x[1])
        
        return self.lrs[steepest[0]]
    
    def plot(self, save_path: Optional[str] = None):
        """Plot LR vs Loss curve.
        
        Args:
            save_path: Optional path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.lrs, self.losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting")


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna.
    
    Searches for optimal LoRA and training hyperparameters.
    """
    
    def __init__(
        self,
        base_model: str,
        train_dataset,
        eval_dataset,
        output_dir: Path,
        n_trials: int = 20,
        timeout: Optional[int] = None
    ):
        """Initialize optimizer.
        
        Args:
            base_model: Base model name.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            output_dir: Output directory.
            n_trials: Number of optimization trials.
            timeout: Optional timeout in seconds.
        """
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.best_params = None
        self.best_value = None
        
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Returns:
            Best hyperparameters found.
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            return {}
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            lora_config = {
                "r": trial.suggest_categorical("lora_r", [4, 8, 16, 32, 64]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64, 128]),
                "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            }
            
            training_config = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
                "gradient_accumulation_steps": trial.suggest_categorical("grad_accum", [4, 8, 16]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.1),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "num_train_epochs": 1,  # Limited for optimization
                "logging_steps": 50,
                "save_steps": 500,
            }
            
            trial_dir = self.output_dir / f"trial_{trial.number}"
            
            try:
                trainer = LoRATrainer(
                    model_name=self.base_model,
                    output_dir=trial_dir,
                    lora_config=lora_config,
                    training_config=training_config,
                    use_wandb=False
                )
                
                result = trainer.train(
                    self.train_dataset,
                    self.eval_dataset
                )
                
                eval_loss = result.metrics.get("eval_loss", float('inf'))
                
                return eval_loss
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        study = optuna.create_study(
            direction="minimize",
            study_name="lora_optimization"
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best eval loss: {self.best_value:.4f}")
        
        # Save results
        results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "all_trials": [
                {"params": t.params, "value": t.value}
                for t in study.trials
            ]
        }
        
        with open(self.output_dir / "optimization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return self.best_params
    
    def get_best_config(self) -> tuple:
        """Get best LoRA and training configs.
        
        Returns:
            Tuple of (lora_config, training_config).
        """
        if not self.best_params:
            return {}, {}
        
        lora_config = {
            "r": self.best_params.get("lora_r", 16),
            "lora_alpha": self.best_params.get("lora_alpha", 32),
            "lora_dropout": self.best_params.get("lora_dropout", 0.05),
        }
        
        training_config = {
            "learning_rate": self.best_params.get("learning_rate", 2e-4),
            "per_device_train_batch_size": self.best_params.get("batch_size", 4),
            "gradient_accumulation_steps": self.best_params.get("grad_accum", 8),
            "warmup_ratio": self.best_params.get("warmup_ratio", 0.03),
            "weight_decay": self.best_params.get("weight_decay", 0.01),
        }
        
        return lora_config, training_config


class CheckpointManager:
    """Enhanced checkpoint management for training.
    
    Handles saving, loading, and cleaning up checkpoints.
    """
    
    def __init__(
        self,
        output_dir: Path,
        max_checkpoints: int = 3,
        save_best_only: bool = False,
        metric: str = "eval_loss",
        mode: str = "min"
    ):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Checkpoint directory.
            max_checkpoints: Maximum checkpoints to keep.
            save_best_only: Only save if best so far.
            metric: Metric to track for best model.
            mode: 'min' or 'max' for metric comparison.
        """
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_value = float('inf') if mode == "min" else float('-inf')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model,
        tokenizer,
        step: int,
        metrics: Dict[str, float]
    ) -> Optional[Path]:
        """Save checkpoint.
        
        Args:
            model: Model to save.
            tokenizer: Tokenizer to save.
            step: Training step.
            metrics: Current metrics.
            
        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        current_value = metrics.get(self.metric, 0)
        
        # Check if we should save
        if self.save_best_only:
            is_better = (
                current_value < self.best_value if self.mode == "min"
                else current_value > self.best_value
            )
            
            if not is_better:
                return None
            
            self.best_value = current_value
        
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata
        metadata = {
            "step": step,
            "metrics": metrics,
            "path": str(checkpoint_path),
            "timestamp": str(Path(checkpoint_path).stat().st_mtime)
        }
        
        with open(checkpoint_path / "checkpoint_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.checkpoints.append(metadata)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_best(self) -> Optional[Path]:
        """Load best checkpoint path.
        
        Returns:
            Path to best checkpoint.
        """
        if not self.checkpoints:
            return None
        
        if self.mode == "min":
            best = min(
                self.checkpoints,
                key=lambda x: x["metrics"].get(self.metric, float('inf'))
            )
        else:
            best = max(
                self.checkpoints,
                key=lambda x: x["metrics"].get(self.metric, float('-inf'))
            )
        
        return Path(best["path"])
    
    def load_latest(self) -> Optional[Path]:
        """Load latest checkpoint path.
        
        Returns:
            Path to latest checkpoint.
        """
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x["step"])
        return Path(latest["path"])
    
    def _cleanup(self):
        """Remove old checkpoints beyond max limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by metric value
        if self.mode == "min":
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: x["metrics"].get(self.metric, float('inf'))
            )
        else:
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: x["metrics"].get(self.metric, float('-inf')),
                reverse=True
            )
        
        # Keep best checkpoints
        to_keep = sorted_checkpoints[:self.max_checkpoints]
        to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for ckpt in to_remove:
            path = Path(ckpt["path"])
            if path.exists():
                import shutil
                shutil.rmtree(path)
                logger.debug(f"Removed checkpoint: {path}")
        
        self.checkpoints = to_keep

