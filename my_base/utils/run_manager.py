"""
Experiment run management utilities.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)

class RunManager:
    """Manages experiment runs, including directory creation and metrics tracking."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_dir: str = "runs",
        create_dirs: bool = True
    ):
        """
        Initialize RunManager.
        
        Args:
            config: Experiment configuration
            base_dir: Base directory for all runs
            create_dirs: Whether to create run directory structure
        """
        self.config = config
        self.base_dir = Path(base_dir)
        
        # Generate run name and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get('run_name', '')
        if not run_name:
            run_name = self._generate_run_name()
        
        self.run_dir = self.base_dir / f"{timestamp}_{run_name}"
        self.metrics_file = self.run_dir / "metrics.csv"
        self.config_file = self.run_dir / "config.yaml"
        
        # Create directory structure if requested
        if create_dirs:
            self._create_run_directory()
    
    def _generate_run_name(self) -> str:
        """Generate descriptive run name from config."""
        components = [
            self.config.get('fl_type', 'fl'),
            self.config.get('dataset', 'data'),
            self.config.get('model', 'model')
        ]
        
        # Add attack/defense if present
        if attack := self.config.get('attack'):
            components.append(f"atk_{attack}")
        if defense := self.config.get('defense'):
            components.append(f"def_{defense}")
            
        return "_".join(components)
    
    def _create_run_directory(self) -> None:
        """Create run directory structure."""
        # Create main directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        
        # Save config
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        
        # Initialize metrics file with header
        if not self.metrics_file.exists():
            pd.DataFrame().to_csv(self.metrics_file, index=False)
        
        logger.info(f"Created run directory: {self.run_dir}")
    
    def get_checkpoint_dir(self) -> Path:
        """Get path to checkpoints directory."""
        return self.run_dir / "checkpoints"
    
    def get_plot_dir(self) -> Path:
        """Get path to plots directory."""
        return self.run_dir / "plots"
    
    def get_log_dir(self) -> Path:
        """Get path to logs directory."""
        return self.run_dir / "logs"
    
    def save_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Save metrics to CSV file.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            prefix: Prefix for metric names
        """
        # Prepare metrics data
        data = metrics.copy()
        if step is not None:
            data['step'] = step
        if prefix:
            data = {f"{prefix}_{k}": v for k, v in data.items()}
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        
        # Append to CSV
        df = pd.DataFrame([data])
        df.to_csv(self.metrics_file, mode='a', header=False, index=False)
    
    def load_metrics(self) -> pd.DataFrame:
        """Load all metrics from CSV file."""
        if not self.metrics_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.metrics_file)
    
    def save_artifact(
        self,
        artifact: Any,
        name: str,
        artifact_type: str = "misc"
    ) -> None:
        """
        Save arbitrary artifact to run directory.
        
        Args:
            artifact: Object to save
            name: Artifact name
            artifact_type: Type of artifact (for directory organization)
        """
        artifact_dir = self.run_dir / "artifacts" / artifact_type
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        path = artifact_dir / name
        if isinstance(artifact, (dict, list)):
            with open(path, 'w') as f:
                json.dump(artifact, f)
        else:
            # Try to use object's save method
            try:
                artifact.save(path)
            except AttributeError:
                logger.warning(f"Could not save artifact {name}: no save method")
    
    def load_artifact(
        self,
        name: str,
        artifact_type: str = "misc"
    ) -> Any:
        """
        Load artifact from run directory.
        
        Args:
            name: Artifact name
            artifact_type: Type of artifact
            
        Returns:
            Loaded artifact
        """
        path = self.run_dir / "artifacts" / artifact_type / name
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        if path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        else:
            # Let caller handle loading
            return path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get run summary including config and final metrics."""
        metrics_df = self.load_metrics()
        
        summary = {
            'run_dir': str(self.run_dir),
            'config': self.config,
            'start_time': metrics_df['timestamp'].min() if not metrics_df.empty else None,
            'end_time': metrics_df['timestamp'].max() if not metrics_df.empty else None,
            'final_metrics': metrics_df.iloc[-1].to_dict() if not metrics_df.empty else {}
        }
        
        return summary
    
    @classmethod
    def load_run(cls, run_dir: str) -> "RunManager":
        """
        Load existing run from directory.
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            RunManager: Loaded run manager
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Load config
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Create manager without creating directories
        manager = cls(config, base_dir=run_dir.parent, create_dirs=False)
        manager.run_dir = run_dir
        
        return manager 