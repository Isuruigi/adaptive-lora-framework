"""
Reinforcement learning-based router training.

Features:
- Policy gradient (REINFORCE)
- Proximal Policy Optimization (PPO)
- Reward shaping from evaluation signals
- Experience replay
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Experience:
    """Single experience tuple for RL training."""
    
    state: Dict[str, torch.Tensor]  # Input tokens
    action: int  # Selected adapter index
    reward: float  # Reward signal
    next_state: Optional[Dict[str, torch.Tensor]] = None
    done: bool = False
    log_prob: float = 0.0
    value: float = 0.0
    action_probs: Optional[torch.Tensor] = None


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize buffer.
        
        Args:
            capacity: Maximum buffer size.
        """
        self.buffer: deque = deque(maxlen=capacity)
        
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()


class RewardShaper:
    """Shape rewards from evaluation signals."""
    
    def __init__(
        self,
        quality_weight: float = 0.5,
        latency_weight: float = 0.2,
        cost_weight: float = 0.1,
        consistency_weight: float = 0.2
    ):
        """Initialize reward shaper.
        
        Args:
            quality_weight: Weight for quality score.
            latency_weight: Weight for latency penalty.
            cost_weight: Weight for compute cost penalty.
            consistency_weight: Weight for routing consistency.
        """
        self.quality_weight = quality_weight
        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.consistency_weight = consistency_weight
        
        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history: List[float] = []
        
    def compute_reward(
        self,
        quality_score: float,
        latency_ms: float,
        compute_cost: float = 0.0,
        routing_entropy: float = 0.0,
        target_latency_ms: float = 500.0
    ) -> float:
        """Compute shaped reward.
        
        Args:
            quality_score: Quality evaluation score [0, 1].
            latency_ms: Response latency in milliseconds.
            compute_cost: Normalized compute cost [0, 1].
            routing_entropy: Entropy of routing distribution.
            target_latency_ms: Target latency threshold.
            
        Returns:
            Shaped reward value.
        """
        # Quality reward (primary signal)
        quality_reward = quality_score
        
        # Latency penalty (0 if under target, negative otherwise)
        latency_ratio = latency_ms / target_latency_ms
        latency_penalty = -max(0, latency_ratio - 1.0)
        
        # Cost penalty
        cost_penalty = -compute_cost
        
        # Consistency bonus (prefer confident routing)
        consistency_bonus = -routing_entropy * 0.5
        
        # Combine
        reward = (
            self.quality_weight * quality_reward +
            self.latency_weight * latency_penalty +
            self.cost_weight * cost_penalty +
            self.consistency_weight * consistency_bonus
        )
        
        return reward
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 100:
            self.reward_mean = np.mean(self.reward_history[-100:])
            self.reward_std = np.std(self.reward_history[-100:]) + 1e-8
            
        return (reward - self.reward_mean) / self.reward_std


class PolicyGradientTrainer:
    """REINFORCE algorithm for router training."""
    
    def __init__(
        self,
        router_model: nn.Module,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cuda"
    ):
        """Initialize trainer.
        
        Args:
            router_model: Router network.
            learning_rate: Learning rate.
            gamma: Discount factor.
            entropy_coef: Entropy regularization coefficient.
            device: Device for training.
        """
        self.router = router_model.to(device)
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device
        
        self.reward_shaper = RewardShaper()
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        
    def select_action(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[int, torch.Tensor]:
        """Select adapter using policy.
        
        Args:
            state: Input tokens.
            
        Returns:
            Tuple of (selected adapter index, log probability).
        """
        state = {k: v.to(self.device) for k, v in state.items()}
        
        self.router.eval()
        with torch.no_grad():
            output = self.router(**state)
            
        probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
        probs = probs.squeeze()
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_transition(self, log_prob: torch.Tensor, reward: float) -> None:
        """Store transition for episode."""
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(reward)
        
    def finish_episode(self) -> Dict[str, float]:
        """Update policy after episode ends.
        
        Returns:
            Training metrics.
        """
        if not self.episode_rewards:
            return {"loss": 0.0}
            
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy loss
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode
        avg_reward = np.mean(self.episode_rewards)
        self.episode_log_probs = []
        self.episode_rewards = []
        
        return {"loss": loss.item(), "avg_reward": avg_reward}


class PPOTrainer:
    """Proximal Policy Optimization for router training."""
    
    def __init__(
        self,
        router_model: nn.Module,
        value_model: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cuda"
    ):
        """Initialize PPO trainer.
        
        Args:
            router_model: Policy network.
            value_model: Value network (optional, uses shared backbone if None).
            learning_rate: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_epsilon: PPO clipping epsilon.
            entropy_coef: Entropy bonus coefficient.
            value_coef: Value loss coefficient.
            max_grad_norm: Max gradient norm for clipping.
            ppo_epochs: Number of PPO update epochs.
            batch_size: Minibatch size.
            device: Device for training.
        """
        self.router = router_model.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Value head (simple linear if not provided)
        if value_model is None:
            self.value_head = nn.Linear(256, 1).to(device)  # Assumes hidden_dim=256
        else:
            self.value_head = value_model.to(device)
            
        self.optimizer = torch.optim.Adam(
            list(self.router.parameters()) + list(self.value_head.parameters()),
            lr=learning_rate
        )
        
        self.buffer = ReplayBuffer(capacity=10000)
        self.reward_shaper = RewardShaper()
        
    def get_value(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimate for state."""
        state = {k: v.to(self.device) for k, v in state.items()}
        
        if hasattr(self.router, 'encoder'):
            with torch.no_grad():
                outputs = self.router.encoder(**state)
                hidden = outputs.last_hidden_state[:, 0, :]
        else:
            hidden = torch.zeros(1, 256, device=self.device)
            
        return self.value_head(hidden)
    
    def select_action(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action with value estimate.
        
        Returns:
            Tuple of (action, log_prob, value, action_probs).
        """
        state = {k: v.to(self.device) for k, v in state.items()}
        
        self.router.eval()
        with torch.no_grad():
            output = self.router(**state)
            
        probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
        probs = probs.squeeze()
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        value = self.get_value(state)
        
        return action.item(), log_prob, value.squeeze(), probs
    
    def store_experience(self, experience: Experience) -> None:
        """Store experience in buffer."""
        self.buffer.push(experience)
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.
        
        Args:
            rewards: List of rewards.
            values: List of value estimates.
            dones: List of done flags.
            next_value: Value of next state.
            
        Returns:
            Tuple of (advantages, returns).
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + torch.tensor(values[:-1], device=self.device)
        
        return advantages, returns
    
    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """Perform PPO update.
        
        Args:
            experiences: List of experiences.
            
        Returns:
            Training metrics.
        """
        if len(experiences) < self.batch_size:
            return {"loss": 0.0}
            
        # Extract data
        states = [e.state for e in experiences]
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], device=self.device)
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]
        
        # Compute advantages
        next_value = self.get_value(states[-1]).item() if not dones[-1] else 0.0
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.ppo_epochs):
            # Shuffle and create minibatches
            indices = list(range(len(experiences)))
            random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                batch_probs = []
                batch_values = []
                
                for idx in batch_indices:
                    state = {k: v.to(self.device) for k, v in states[idx].items()}
                    output = self.router(**state)
                    probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
                    batch_probs.append(probs.squeeze())
                    batch_values.append(self.get_value(state).squeeze())
                    
                batch_probs = torch.stack(batch_probs)
                batch_values = torch.stack(batch_values)
                
                # Compute new log probs
                batch_actions = actions[batch_indices]
                dist = Categorical(batch_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio
                batch_old_log_probs = old_log_probs[batch_indices]
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                batch_advantages = advantages[batch_indices]
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                batch_returns = returns[batch_indices]
                value_loss = F.mse_loss(batch_values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.router.parameters()) + list(self.value_head.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
        n_updates = self.ppo_epochs * (len(experiences) // self.batch_size + 1)
        
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "avg_reward": np.mean(rewards)
        }


class RLRouterTrainer:
    """High-level RL trainer for router with environment interaction."""
    
    def __init__(
        self,
        router_model: nn.Module,
        evaluator,
        tokenizer,
        algorithm: str = "ppo",
        device: str = "cuda",
        **kwargs
    ):
        """Initialize RL trainer.
        
        Args:
            router_model: Router network.
            evaluator: Self-evaluator for reward computation.
            tokenizer: Tokenizer.
            algorithm: RL algorithm ('reinforce', 'ppo').
            device: Device.
            **kwargs: Algorithm-specific arguments.
        """
        self.router = router_model
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.device = device
        
        if algorithm == "ppo":
            self.trainer = PPOTrainer(router_model, device=device, **kwargs)
        else:
            self.trainer = PolicyGradientTrainer(router_model, device=device, **kwargs)
            
        self.reward_shaper = RewardShaper()
        
    def train_episode(
        self,
        queries: List[str],
        adapter_inference_fn,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """Train on a batch of queries.
        
        Args:
            queries: Training queries.
            adapter_inference_fn: Function to run inference with selected adapter.
            max_steps: Maximum steps per episode.
            
        Returns:
            Training metrics.
        """
        experiences = []
        
        for query in queries[:max_steps]:
            # Tokenize
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Select action
            if isinstance(self.trainer, PPOTrainer):
                action, log_prob, value, probs = self.trainer.select_action(inputs)
            else:
                action, log_prob = self.trainer.select_action(inputs)
                value = 0.0
                probs = None
                
            # Execute action (run inference with selected adapter)
            import time
            start_time = time.time()
            response = adapter_inference_fn(query, adapter_idx=action)
            latency_ms = (time.time() - start_time) * 1000
            
            # Evaluate response
            eval_result = self.evaluator.evaluate(query, response)
            quality_score = eval_result.get("overall_score", 0.5)
            
            # Compute reward
            routing_entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item() if probs is not None else 0.0
            reward = self.reward_shaper.compute_reward(
                quality_score=quality_score,
                latency_ms=latency_ms,
                routing_entropy=routing_entropy
            )
            
            # Store experience
            exp = Experience(
                state=inputs,
                action=action,
                reward=reward,
                log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                value=value.item() if isinstance(value, torch.Tensor) else value,
                action_probs=probs
            )
            experiences.append(exp)
            
            # For REINFORCE
            if isinstance(self.trainer, PolicyGradientTrainer):
                self.trainer.store_transition(log_prob, reward)
                
        # Update
        if isinstance(self.trainer, PPOTrainer):
            metrics = self.trainer.update(experiences)
        else:
            metrics = self.trainer.finish_episode()
            
        return metrics
    
    def train(
        self,
        train_queries: List[str],
        adapter_inference_fn,
        num_episodes: int = 100,
        queries_per_episode: int = 32
    ) -> List[Dict[str, float]]:
        """Full training loop.
        
        Args:
            train_queries: All training queries.
            adapter_inference_fn: Inference function.
            num_episodes: Number of episodes.
            queries_per_episode: Queries per episode.
            
        Returns:
            List of metrics per episode.
        """
        all_metrics = []
        
        for episode in range(num_episodes):
            # Sample queries
            batch_queries = random.sample(
                train_queries,
                min(queries_per_episode, len(train_queries))
            )
            
            metrics = self.train_episode(batch_queries, adapter_inference_fn)
            all_metrics.append(metrics)
            
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"loss={metrics.get('loss', 0):.4f}, "
                    f"avg_reward={metrics.get('avg_reward', 0):.4f}"
                )
                
        return all_metrics
