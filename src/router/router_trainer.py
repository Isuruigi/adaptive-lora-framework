"""
Router training pipeline with multiple training strategies.

Features:
- Multi-objective loss functions
- Curriculum learning
- Active learning
- Reinforcement learning option
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.router.router_model import AdapterRouter, RouterOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RouterDataset(Dataset):
    """Dataset for router training.

    Example data format:
        {
            'query': "What is machine learning?",
            'optimal_adapter': 2,
            'adapter_weights': [0.1, 0.2, 0.6, 0.1],
            'complexity': 'medium',
            'adapter_scores': [0.7, 0.8, 0.95, 0.6]
        }
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512
    ):
        """Initialize dataset.

        Args:
            data: List of training examples.
            tokenizer: Tokenizer for encoding queries.
            max_length: Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.complexity_map = {"easy": 0, "medium": 1, "hard": 2}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize query
        encoding = self.tokenizer(
            item["query"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Prepare labels
        adapter_weights = torch.tensor(
            item["adapter_weights"], dtype=torch.float32
        )
        complexity_label = self.complexity_map[item["complexity"]]
        adapter_scores = torch.tensor(
            item.get("adapter_scores", item["adapter_weights"]),
            dtype=torch.float32
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "adapter_weights": adapter_weights,
            "complexity_label": complexity_label,
            "adapter_scores": adapter_scores
        }


class RouterTrainer:
    """Train router network with multiple strategies."""

    def __init__(
        self,
        router_model: AdapterRouter,
        tokenizer,
        output_dir: Path,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_wandb: bool = True,
        device: str = "cuda"
    ):
        """Initialize trainer.

        Args:
            router_model: Router model to train.
            tokenizer: Tokenizer for encoding.
            output_dir: Output directory.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            use_wandb: Enable W&B logging.
            device: Target device.
        """
        self.router = router_model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.router.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.router.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss weights
        self.loss_weights = {
            "routing": 1.0,
            "complexity": 0.5,
            "ranking": 0.3
        }

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        target_weights: torch.Tensor,
        target_complexity: torch.Tensor,
        adapter_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-objective loss.

        Args:
            output: Router output with logits.
            target_weights: Ground truth adapter weights.
            target_complexity: Ground truth complexity labels.
            adapter_scores: Actual performance scores.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Routing loss - KL divergence
        predicted_weights = F.softmax(output["weight_logits"], dim=-1)
        routing_loss = F.kl_div(
            predicted_weights.log(),
            target_weights,
            reduction="batchmean"
        )

        # Complexity classification loss
        complexity_loss = F.cross_entropy(
            output["complexity_logits"],
            target_complexity
        )

        # Ranking loss
        ranking_loss = self._compute_ranking_loss(
            predicted_weights,
            adapter_scores
        )

        # Total weighted loss
        total_loss = (
            self.loss_weights["routing"] * routing_loss +
            self.loss_weights["complexity"] * complexity_loss +
            self.loss_weights["ranking"] * ranking_loss
        )

        losses = {
            "total": total_loss.item(),
            "routing": routing_loss.item(),
            "complexity": complexity_loss.item(),
            "ranking": ranking_loss.item()
        }

        return total_loss, losses

    def _compute_ranking_loss(
        self,
        predicted_weights: torch.Tensor,
        adapter_scores: torch.Tensor,
        margin: float = 0.1
    ) -> torch.Tensor:
        """Compute ranking loss for adapter ordering."""
        batch_size, num_adapters = predicted_weights.shape

        losses = []
        for i in range(num_adapters):
            for j in range(i + 1, num_adapters):
                score_diff = adapter_scores[:, i] - adapter_scores[:, j]
                weight_diff = predicted_weights[:, i] - predicted_weights[:, j]

                target = torch.sign(score_diff)
                loss = F.relu(margin - target * weight_diff)
                losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=predicted_weights.device)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.router.train()

        total_loss = 0.0
        all_losses = {"routing": 0.0, "complexity": 0.0, "ranking": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_weights = batch["adapter_weights"].to(self.device)
            target_complexity = batch["complexity_label"].to(self.device)
            adapter_scores = batch["adapter_scores"].to(self.device)

            # Forward pass
            output = self.router(
                input_ids,
                attention_mask,
                return_logits=True
            )

            # Compute loss
            loss, losses = self.compute_loss(
                output,
                target_weights,
                target_complexity,
                adapter_scores
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for k in all_losses:
                all_losses[k] += losses[k]

            pbar.set_postfix({"loss": loss.item()})

            # Log to W&B
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/routing_loss": losses["routing"],
                    "train/complexity_loss": losses["complexity"],
                    "train/ranking_loss": losses["ranking"],
                    "epoch": epoch
                })

        # Average losses
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_losses = {k: v / num_batches for k, v in all_losses.items()}

        return {"total_loss": avg_loss, **avg_losses}

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate router.

        Args:
            eval_loader: Evaluation data loader.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.router.eval()

        total_loss = 0.0
        all_losses = {"routing": 0.0, "complexity": 0.0, "ranking": 0.0}

        correct_complexity = 0
        total_samples = 0
        routing_accuracies = []

        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_weights = batch["adapter_weights"].to(self.device)
            target_complexity = batch["complexity_label"].to(self.device)
            adapter_scores = batch["adapter_scores"].to(self.device)

            # Forward pass
            output = self.router(
                input_ids,
                attention_mask,
                return_logits=True
            )

            # Compute loss
            loss, losses = self.compute_loss(
                output,
                target_weights,
                target_complexity,
                adapter_scores
            )

            total_loss += loss.item()
            for k in all_losses:
                all_losses[k] += losses[k]

            # Complexity accuracy
            pred_complexity = output["complexity_logits"].argmax(dim=-1)
            correct_complexity += (pred_complexity == target_complexity).sum().item()

            # Routing accuracy (top-1)
            pred_weights = F.softmax(output["weight_logits"], dim=-1)
            pred_adapter = pred_weights.argmax(dim=-1)
            target_adapter = target_weights.argmax(dim=-1)
            routing_acc = (pred_adapter == target_adapter).float().mean().item()
            routing_accuracies.append(routing_acc)

            total_samples += input_ids.size(0)

        num_batches = len(eval_loader)
        metrics = {
            "eval/loss": total_loss / num_batches,
            "eval/routing_loss": all_losses["routing"] / num_batches,
            "eval/complexity_loss": all_losses["complexity"] / num_batches,
            "eval/ranking_loss": all_losses["ranking"] / num_batches,
            "eval/complexity_accuracy": correct_complexity / total_samples,
            "eval/routing_accuracy": np.mean(routing_accuracies)
        }

        if self.use_wandb:
            wandb.log(metrics)

        return metrics

    def train(
        self,
        train_data: List[Dict],
        eval_data: List[Dict],
        num_epochs: int = 10,
        batch_size: int = 32,
        curriculum: bool = True
    ) -> None:
        """Full training loop.

        Args:
            train_data: Training examples.
            eval_data: Validation examples.
            num_epochs: Number of epochs.
            batch_size: Batch size.
            curriculum: Use curriculum learning.
        """
        # Sort by difficulty for curriculum learning
        if curriculum:
            train_data = self._sort_by_difficulty(train_data)

        # Create datasets
        train_dataset = RouterDataset(train_data, self.tokenizer)
        eval_dataset = RouterDataset(eval_data, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not curriculum,
            num_workers=4
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        best_eval_loss = float("inf")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")

            # Evaluate
            eval_metrics = self.evaluate(eval_loader)
            logger.info(f"Eval Loss: {eval_metrics['eval/loss']:.4f}")
            logger.info(f"Routing Acc: {eval_metrics['eval/routing_accuracy']:.2%}")
            logger.info(f"Complexity Acc: {eval_metrics['eval/complexity_accuracy']:.2%}")

            # Save best model
            if eval_metrics["eval/loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval/loss"]
                self.save_checkpoint("best_router.pt")
                logger.info(f"Saved best model (loss: {best_eval_loss:.4f})")

            scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"router_epoch_{epoch + 1}.pt")

    def _sort_by_difficulty(self, data: List[Dict]) -> List[Dict]:
        """Sort data by difficulty for curriculum learning."""
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        return sorted(data, key=lambda x: difficulty_order.get(x.get("complexity", "medium"), 1))

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.router.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_weights": self.loss_weights
        }

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(
            self.output_dir / filename,
            map_location=self.device
        )

        self.router.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_weights = checkpoint["loss_weights"]


class ActiveLearningRouter(RouterTrainer):
    """Router trainer with active learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = 0.3

    def select_uncertain_examples(
        self,
        unlabeled_pool: List[str],
        n_samples: int = 100,
        strategy: str = "entropy"
    ) -> List[str]:
        """Select most uncertain examples for labeling.

        Args:
            unlabeled_pool: List of unlabeled queries.
            n_samples: Number of samples to select.
            strategy: Selection strategy (entropy, margin, confidence).

        Returns:
            Selected queries.
        """
        self.router.eval()
        uncertainties = []

        for query in tqdm(unlabeled_pool, desc="Computing uncertainties"):
            encoding = self.tokenizer(
                query,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                output = self.router(input_ids, attention_mask)

            if strategy == "entropy":
                weights = output.adapter_weights[0]
                entropy = -(weights * torch.log(weights + 1e-10)).sum().item()
                uncertainties.append(entropy)
            elif strategy == "margin":
                weights = output.adapter_weights[0]
                top2 = torch.topk(weights, 2).values
                margin = (top2[0] - top2[1]).item()
                uncertainties.append(-margin)
            elif strategy == "confidence":
                confidence = output.confidence[0].item()
                uncertainties.append(1 - confidence)

        # Select top uncertain
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        return [unlabeled_pool[i] for i in uncertain_indices]


class ReinforcementRouter(RouterTrainer):
    """Train router with reinforcement learning.

    Uses REINFORCE with baseline for variance reduction.
    Reward = quality of final output using selected adapters.

    Example:
        >>> trainer = ReinforcementRouter(router, tokenizer, output_dir)
        >>> trainer.train_with_rl(queries, adapter_executors, num_episodes=1000)
    """

    def __init__(
        self,
        *args,
        reward_fn: Optional[Callable[[str, List[int], str], float]] = None,
        gamma: float = 0.99,
        **kwargs
    ):
        """Initialize RL trainer.

        Args:
            *args: RouterTrainer args.
            reward_fn: Custom reward function (query, adapters, response) -> float.
            gamma: Discount factor.
            **kwargs: RouterTrainer kwargs.
        """
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn or self._default_reward
        self.gamma = gamma

        # Baseline for variance reduction (REINFORCE with baseline)
        encoder_dim = self.router.encoder.config.hidden_size
        self.baseline = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)

        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            lr=1e-4
        )

        # Experience replay buffer
        self.replay_buffer: List[Dict] = []
        self.max_buffer_size = 10000

    def _default_reward(
        self,
        query: str,
        adapters: List[int],
        response: str
    ) -> float:
        """Default reward function based on response quality.

        Args:
            query: Input query.
            adapters: Selected adapter indices.
            response: Generated response.

        Returns:
            Reward value (0-1).
        """
        # Simple heuristic - can be replaced with LLM-as-judge
        if len(response) < 50:
            return 0.0

        # Check for failure patterns
        failure_patterns = ["i cannot", "sorry", "i'm unable", "as an ai"]
        if any(p in response.lower() for p in failure_patterns):
            return 0.2

        # Length-based reward (capped)
        length_reward = min(1.0, len(response) / 500)

        # Keyword overlap with query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)

        return 0.6 * length_reward + 0.4 * min(overlap, 1.0)

    def compute_policy_gradient_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute REINFORCE loss with baseline.

        Loss = -Σ log π(a|s) * (R - b(s))

        Args:
            log_probs: Log probabilities of actions.
            rewards: Received rewards.
            baseline_values: Baseline predictions.

        Returns:
            Tuple of (policy_loss, baseline_loss).
        """
        advantages = rewards - baseline_values.detach()
        policy_loss = -(log_probs * advantages).mean()

        # Baseline MSE loss
        baseline_loss = F.mse_loss(baseline_values.squeeze(), rewards)

        return policy_loss, baseline_loss

    def train_with_rl(
        self,
        queries: List[str],
        adapter_executors: Dict[int, Callable[[str], str]],
        num_episodes: int = 1000,
        batch_size: int = 16,
        update_frequency: int = 10,
        entropy_coef: float = 0.01
    ) -> Dict[str, List[float]]:
        """Train router with RL.

        Args:
            queries: List of training queries.
            adapter_executors: Dict mapping adapter_id to execution function.
            num_episodes: Number of training episodes.
            batch_size: Batch size per episode.
            update_frequency: Episodes between logging.
            entropy_coef: Entropy bonus coefficient.

        Returns:
            Training history.
        """
        history = {
            "rewards": [],
            "policy_loss": [],
            "baseline_loss": [],
            "entropy": []
        }

        for episode in range(num_episodes):
            # Sample batch of queries
            batch_queries = np.random.choice(
                queries, size=min(batch_size, len(queries)), replace=False
            ).tolist()

            log_probs_batch = []
            rewards_batch = []
            baseline_values_batch = []
            entropies_batch = []

            for query in batch_queries:
                # Encode query
                encoding = self.tokenizer(
                    query,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Router prediction
                output = self.router(
                    encoding["input_ids"],
                    encoding["attention_mask"]
                )

                # Sample action (adapter selection)
                weights = output.adapter_weights[0]
                dist = torch.distributions.Categorical(weights)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # Execute with selected adapter
                adapter_id = action.item()
                if adapter_id in adapter_executors:
                    try:
                        response = adapter_executors[adapter_id](query)
                    except Exception as e:
                        logger.warning(f"Adapter {adapter_id} failed: {e}")
                        response = ""
                else:
                    response = ""

                # Compute reward
                reward = self.reward_fn(query, [adapter_id], response)

                # Baseline value
                with torch.no_grad():
                    encoder_output = self.router.encoder(
                        encoding["input_ids"],
                        encoding["attention_mask"]
                    )
                    cls_embedding = encoder_output.last_hidden_state[:, 0, :]

                baseline_value = self.baseline(cls_embedding)

                log_probs_batch.append(log_prob)
                rewards_batch.append(reward)
                baseline_values_batch.append(baseline_value)
                entropies_batch.append(entropy)

                # Store in replay buffer
                self._store_experience({
                    "query": query,
                    "adapter_id": adapter_id,
                    "reward": reward,
                    "log_prob": log_prob.item()
                })

            # Convert to tensors
            log_probs = torch.stack(log_probs_batch)
            rewards = torch.tensor(rewards_batch, device=self.device)
            baseline_values = torch.cat(baseline_values_batch)
            entropies = torch.stack(entropies_batch)

            # Compute losses
            policy_loss, baseline_loss = self.compute_policy_gradient_loss(
                log_probs, rewards, baseline_values
            )

            # Add entropy bonus
            entropy_bonus = entropy_coef * entropies.mean()
            total_loss = policy_loss - entropy_bonus

            # Update router
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
            self.optimizer.step()

            # Update baseline
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            # Record history
            history["rewards"].append(rewards.mean().item())
            history["policy_loss"].append(policy_loss.item())
            history["baseline_loss"].append(baseline_loss.item())
            history["entropy"].append(entropies.mean().item())

            # Log progress
            if episode % update_frequency == 0:
                avg_reward = np.mean(history["rewards"][-update_frequency:])
                logger.info(
                    f"Episode {episode}: Reward={avg_reward:.3f}, "
                    f"Policy Loss={policy_loss.item():.3f}, "
                    f"Entropy={entropies.mean().item():.3f}"
                )

                if self.use_wandb:
                    wandb.log({
                        "rl/mean_reward": avg_reward,
                        "rl/policy_loss": policy_loss.item(),
                        "rl/baseline_loss": baseline_loss.item(),
                        "rl/entropy": entropies.mean().item(),
                        "episode": episode
                    })

        return history

    def _store_experience(self, experience: Dict) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def sample_from_buffer(
        self,
        batch_size: int
    ) -> List[Dict]:
        """Sample experiences from replay buffer.

        Args:
            batch_size: Number of samples.

        Returns:
            List of experience dicts.
        """
        if len(self.replay_buffer) < batch_size:
            return self.replay_buffer.copy()
        indices = np.random.choice(
            len(self.replay_buffer), size=batch_size, replace=False
        )
        return [self.replay_buffer[i] for i in indices]

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training stats.
        """
        if not self.replay_buffer:
            return {}

        rewards = [e["reward"] for e in self.replay_buffer]
        adapter_counts = {}
        for e in self.replay_buffer:
            aid = e["adapter_id"]
            adapter_counts[aid] = adapter_counts.get(aid, 0) + 1

        return {
            "buffer_size": len(self.replay_buffer),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "adapter_distribution": adapter_counts
        }
