"""
Uncertainty quantification for model predictions.

Features:
- MC Dropout uncertainty estimation
- Ensemble disagreement
- Temperature scaling calibration
- Confidence intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""
    
    mean_prediction: np.ndarray
    epistemic_uncertainty: float  # Model uncertainty (reducible)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    samples: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_prediction": self.mean_prediction.tolist() if isinstance(self.mean_prediction, np.ndarray) else self.mean_prediction,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "confidence_interval": self.confidence_interval,
        }
    
    @property
    def is_confident(self) -> bool:
        """Check if prediction is confident (low uncertainty)."""
        return self.total_uncertainty < 0.3


class MCDropoutEstimator:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        dropout_rate: float = 0.1
    ):
        """Initialize estimator.
        
        Args:
            model: Neural network with dropout layers.
            n_samples: Number of forward passes.
            dropout_rate: Dropout rate to apply.
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def _enable_dropout(self) -> None:
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def _disable_dropout(self) -> None:
        """Disable dropout."""
        self.model.eval()
        
    @torch.no_grad()
    def estimate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str = "cuda"
    ) -> UncertaintyEstimate:
        """Estimate uncertainty using MC Dropout.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            device: Device for inference.
            
        Returns:
            UncertaintyEstimate with predictions and uncertainties.
        """
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        self._enable_dropout()
        
        # Collect samples
        samples = []
        for _ in range(self.n_samples):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
            samples.append(probs.cpu().numpy())
            
        self._disable_dropout()
        
        samples = np.array(samples)  # (n_samples, batch_size, num_classes)
        
        # Compute statistics
        mean_pred = samples.mean(axis=0).squeeze()
        var_pred = samples.var(axis=0).squeeze()
        
        # Epistemic uncertainty (variance of means)
        epistemic = var_pred.mean()
        
        # Aleatoric uncertainty (mean of variances - approximation)
        # For classification, use entropy of mean prediction
        aleatoric = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        # Confidence interval (95%)
        lower = np.percentile(samples.mean(axis=(1, 2)), 2.5) if samples.ndim > 2 else np.percentile(samples.mean(axis=1), 2.5)
        upper = np.percentile(samples.mean(axis=(1, 2)), 97.5) if samples.ndim > 2 else np.percentile(samples.mean(axis=1), 97.5)
        
        return UncertaintyEstimate(
            mean_prediction=mean_pred,
            epistemic_uncertainty=float(epistemic),
            aleatoric_uncertainty=float(aleatoric),
            total_uncertainty=float(total),
            confidence_interval=(float(lower), float(upper)),
            samples=samples
        )


class EnsembleEstimator:
    """Ensemble-based uncertainty estimation."""
    
    def __init__(self, models: List[nn.Module]):
        """Initialize with ensemble of models.
        
        Args:
            models: List of trained models.
        """
        self.models = models
        
    def add_model(self, model: nn.Module) -> None:
        """Add model to ensemble."""
        self.models.append(model)
        
    @torch.no_grad()
    def estimate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str = "cuda"
    ) -> UncertaintyEstimate:
        """Estimate uncertainty using ensemble disagreement.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            device: Device.
            
        Returns:
            UncertaintyEstimate.
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
            
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            model.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = output.adapter_weights if hasattr(output, 'adapter_weights') else F.softmax(output, dim=-1)
            predictions.append(probs.cpu().numpy())
            
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(axis=0).squeeze()
        
        # Epistemic uncertainty (disagreement between models)
        epistemic = predictions.var(axis=0).mean()
        
        # Aleatoric (entropy of mean)
        aleatoric = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
        
        total = epistemic + aleatoric
        
        # Confidence interval
        lower = np.percentile(predictions.mean(axis=(1, 2)) if predictions.ndim > 2 else predictions.mean(axis=1), 2.5)
        upper = np.percentile(predictions.mean(axis=(1, 2)) if predictions.ndim > 2 else predictions.mean(axis=1), 97.5)
        
        return UncertaintyEstimate(
            mean_prediction=mean_pred,
            epistemic_uncertainty=float(epistemic),
            aleatoric_uncertainty=float(aleatoric),
            total_uncertainty=float(total),
            confidence_interval=(float(lower), float(upper)),
            samples=predictions
        )


class TemperatureScaling:
    """Temperature scaling for calibrated probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        """Initialize scaler.
        
        Args:
            temperature: Initial temperature.
        """
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """Learn optimal temperature from validation data.
        
        Args:
            logits: Model logits (N, C).
            labels: True labels (N,).
            lr: Learning rate.
            max_iter: Maximum iterations.
            
        Returns:
            Optimal temperature.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        logger.info(f"Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()
    
    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.
        
        Args:
            logits: Raw logits.
            
        Returns:
            Scaled probabilities.
        """
        return F.softmax(logits / self.temperature, dim=-1)


class CalibrationMetrics:
    """Compute calibration metrics."""
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities (N, C).
            labels: True labels (N,).
            n_bins: Number of bins.
            
        Returns:
            ECE value.
        """
        confidences = predictions.max(axis=1)
        predicted_labels = predictions.argmax(axis=1)
        accuracies = (predicted_labels == labels).astype(float)
        
        ece = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
                
        return ece
    
    @staticmethod
    def maximum_calibration_error(
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error (MCE).
        
        Args:
            predictions: Predicted probabilities.
            labels: True labels.
            n_bins: Number of bins.
            
        Returns:
            MCE value.
        """
        confidences = predictions.max(axis=1)
        predicted_labels = predictions.argmax(axis=1)
        accuracies = (predicted_labels == labels).astype(float)
        
        mce = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence - avg_accuracy))
                
        return mce
    
    @staticmethod
    def reliability_diagram(
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """Generate data for reliability diagram.
        
        Args:
            predictions: Predicted probabilities.
            labels: True labels.
            n_bins: Number of bins.
            
        Returns:
            Dictionary with bin_centers, accuracies, and confidences.
        """
        confidences = predictions.max(axis=1)
        predicted_labels = predictions.argmax(axis=1)
        accuracies = (predicted_labels == labels).astype(float)
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                
        return {
            "bin_centers": bin_centers,
            "accuracies": bin_accuracies,
            "confidences": bin_confidences
        }


class UncertaintyQuantifier:
    """Unified interface for uncertainty quantification."""
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "mc_dropout",
        n_samples: int = 30,
        ensemble_models: Optional[List[nn.Module]] = None
    ):
        """Initialize quantifier.
        
        Args:
            model: Primary model.
            method: Estimation method ('mc_dropout', 'ensemble').
            n_samples: Number of samples for MC dropout.
            ensemble_models: Optional ensemble for ensemble method.
        """
        self.model = model
        self.method = method
        
        if method == "mc_dropout":
            self.estimator = MCDropoutEstimator(model, n_samples=n_samples)
        elif method == "ensemble":
            models = ensemble_models or [model]
            self.estimator = EnsembleEstimator(models)
        else:
            self.estimator = MCDropoutEstimator(model, n_samples=n_samples)
            
        self.temp_scaling = TemperatureScaling()
        
    def quantify(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str = "cuda"
    ) -> UncertaintyEstimate:
        """Quantify uncertainty for input.
        
        Args:
            input_ids: Input tokens.
            attention_mask: Attention mask.
            device: Device.
            
        Returns:
            Uncertainty estimate.
        """
        return self.estimator.estimate(input_ids, attention_mask, device)
    
    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor
    ) -> float:
        """Calibrate using validation data.
        
        Args:
            val_logits: Validation logits.
            val_labels: Validation labels.
            
        Returns:
            Calibrated temperature.
        """
        return self.temp_scaling.calibrate(val_logits, val_labels)
    
    def get_calibration_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute all calibration metrics.
        
        Args:
            predictions: Predicted probabilities.
            labels: True labels.
            
        Returns:
            Dictionary of calibration metrics.
        """
        return {
            "ece": CalibrationMetrics.expected_calibration_error(predictions, labels),
            "mce": CalibrationMetrics.maximum_calibration_error(predictions, labels),
        }
