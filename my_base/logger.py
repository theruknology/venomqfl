"""
Logging configuration for QFLBackdoorAttacks.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import wandb
from datetime import datetime

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    config: Optional[dict] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name (required if use_wandb=True)
        config: Configuration dictionary for W&B initialization
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Initialize W&B logging if requested
    if use_wandb:
        if not wandb_project:
            raise ValueError("wandb_project must be provided when use_wandb=True")
        
        run_name = config.get('run_name') if config else None
        if not run_name:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=config,
            reinit=True
        )
        logger.info(f"Initialized W&B logging: project={wandb_project}, run={run_name}")
    
    return logger

def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    step: Optional[int] = None,
    use_wandb: bool = False
) -> None:
    """
    Log metrics to console, file, and optionally W&B.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        step: Current step/epoch number
        use_wandb: Whether to log to W&B
    """
    # Format metrics string
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    if step is not None:
        metrics_str = f"Step {step} | {metrics_str}"
    
    # Log to console/file
    logger.info(metrics_str)
    
    # Log to W&B
    if use_wandb:
        wandb.log(metrics, step=step)

def log_system_info(logger: logging.Logger) -> None:
    """Log system information and package versions."""
    import torch
    import qiskit
    import pennylane
    import platform
    import psutil
    
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "Qiskit": qiskit.__version__,
        "PennyLane": pennylane.version(),
        "CUDA Available": torch.cuda.is_available(),
        "GPU Devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "CPU Count": psutil.cpu_count(),
        "Platform": platform.platform(),
    }
    
    logger.info("System Information:")
    for k, v in info.items():
        logger.info(f"  {k}: {v}")

class MetricLogger:
    """Helper class to track and log training metrics."""
    
    def __init__(
        self,
        logger: logging.Logger,
        use_wandb: bool = False,
        log_interval: int = 10
    ):
        self.logger = logger
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.reset()
    
    def reset(self):
        """Reset metric accumulator."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: dict):
        """Update running metrics."""
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = 0
                self.counts[k] = 0
            self.metrics[k] += v
            self.counts[k] += 1
    
    def get_avg_metrics(self):
        """Get averaged metrics."""
        return {k: self.metrics[k] / self.counts[k] for k in self.metrics}
    
    def log(self, step: Optional[int] = None, force: bool = False):
        """Log current metrics if interval is reached or forced."""
        if force or step % self.log_interval == 0:
            avg_metrics = self.get_avg_metrics()
            log_metrics(self.logger, avg_metrics, step, self.use_wandb)
            self.reset() 