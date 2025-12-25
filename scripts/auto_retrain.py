"""
Automated Retraining Scheduler

Monitors model quality metrics and triggers retraining when performance degrades.
Integrates with the drift detection and active learning modules.

Usage:
    python -m scripts.auto_retrain --config configs/monitoring.yaml
    
    # Or run as a background service
    python -m scripts.auto_retrain --daemon --interval 3600
"""

import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoRetrainingScheduler:
    """
    Automated retraining scheduler that monitors metrics and triggers retraining.
    
    Features:
    - Quality threshold monitoring
    - Drift detection integration
    - Configurable retraining triggers
    - Slack/email notifications
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        drift_threshold: float = 0.2,
        min_samples_for_retrain: int = 1000,
        cooldown_hours: int = 24
    ):
        """
        Initialize scheduler.
        
        Args:
            quality_threshold: Minimum quality score before triggering retrain
            drift_threshold: PSI threshold for drift detection
            min_samples_for_retrain: Minimum new samples before retraining
            cooldown_hours: Hours to wait between retraining runs
        """
        self.quality_threshold = quality_threshold
        self.drift_threshold = drift_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.cooldown_hours = cooldown_hours
        self.last_retrain_time: Optional[datetime] = None
        
    def check_quality_metrics(self) -> Dict[str, float]:
        """
        Check current quality metrics from monitoring system.
        
        Returns:
            Dictionary of adapter -> quality score
        """
        # In production, fetch from Prometheus/database
        # This is a placeholder that would integrate with your monitoring
        try:
            from src.monitoring.performance_tracker import PerformanceTracker
            tracker = PerformanceTracker()
            return tracker.get_adapter_quality_scores()
        except Exception as e:
            logger.warning(f"Could not fetch metrics: {e}")
            return {}
    
    def check_drift(self) -> Dict[str, float]:
        """
        Check drift scores for each adapter.
        
        Returns:
            Dictionary of adapter -> drift score (PSI)
        """
        try:
            from src.monitoring.drift_detector import DriftDetector
            detector = DriftDetector()
            return detector.get_current_drift_scores()
        except Exception as e:
            logger.warning(f"Could not fetch drift scores: {e}")
            return {}
    
    def get_new_sample_count(self) -> int:
        """Get count of new samples since last training."""
        # In production, query from database
        # Placeholder implementation
        return 0
    
    def should_retrain(self) -> tuple[bool, str]:
        """
        Determine if retraining should be triggered.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check cooldown
        if self.last_retrain_time:
            hours_since = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return False, f"Cooldown active ({self.cooldown_hours - hours_since:.1f}h remaining)"
        
        # Check quality degradation
        quality_scores = self.check_quality_metrics()
        for adapter, score in quality_scores.items():
            if score < self.quality_threshold:
                return True, f"Quality degraded for {adapter}: {score:.2f} < {self.quality_threshold}"
        
        # Check drift
        drift_scores = self.check_drift()
        for adapter, psi in drift_scores.items():
            if psi > self.drift_threshold:
                return True, f"Drift detected for {adapter}: PSI={psi:.3f} > {self.drift_threshold}"
        
        # Check sample count
        new_samples = self.get_new_sample_count()
        if new_samples >= self.min_samples_for_retrain:
            return True, f"Sufficient new samples: {new_samples} >= {self.min_samples_for_retrain}"
        
        return False, "No retrain conditions met"
    
    def trigger_retraining(self, adapter: str = "all", reason: str = "") -> bool:
        """
        Trigger the retraining pipeline.
        
        Args:
            adapter: Adapter to retrain ("all" for all adapters)
            reason: Reason for retraining (for logging)
            
        Returns:
            True if retraining was triggered successfully
        """
        logger.info(f"🔄 Triggering retraining for {adapter}")
        logger.info(f"   Reason: {reason}")
        
        try:
            # In production, this would:
            # 1. Fetch latest training data from active learning
            # 2. Launch training job (local, Colab, or cloud)
            # 3. Validate new model
            # 4. Deploy if validation passes
            
            from src.router.active_learning import ActiveLearningManager
            
            # Get high-value samples for retraining
            al_manager = ActiveLearningManager()
            training_samples = al_manager.get_samples_for_training(
                adapter=adapter if adapter != "all" else None,
                min_confidence=0.5
            )
            
            logger.info(f"   Collected {len(training_samples)} samples for training")
            
            # Log retraining event
            self._log_retrain_event(adapter, reason, len(training_samples))
            
            self.last_retrain_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False
    
    def _log_retrain_event(self, adapter: str, reason: str, sample_count: int):
        """Log retraining event to file/database."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "adapter": adapter,
            "reason": reason,
            "sample_count": sample_count
        }
        
        log_file = Path("logs/retrain_history.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def run_once(self) -> bool:
        """
        Run a single check cycle.
        
        Returns:
            True if retraining was triggered
        """
        logger.info("Checking retraining conditions...")
        
        should_retrain, reason = self.should_retrain()
        
        if should_retrain:
            return self.trigger_retraining(reason=reason)
        else:
            logger.info(f"No retraining needed: {reason}")
            return False
    
    def run_daemon(self, interval_seconds: int = 3600):
        """
        Run as a continuous daemon service.
        
        Args:
            interval_seconds: Seconds between checks
        """
        logger.info(f"Starting auto-retrain daemon (interval: {interval_seconds}s)")
        
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
            
            time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automated Retraining Scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds")
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--drift-threshold", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--cooldown", type=int, default=24, help="Cooldown hours")
    
    args = parser.parse_args()
    
    scheduler = AutoRetrainingScheduler(
        quality_threshold=args.quality_threshold,
        drift_threshold=args.drift_threshold,
        min_samples_for_retrain=args.min_samples,
        cooldown_hours=args.cooldown
    )
    
    if args.daemon:
        scheduler.run_daemon(args.interval)
    else:
        scheduler.run_once()


if __name__ == "__main__":
    main()
