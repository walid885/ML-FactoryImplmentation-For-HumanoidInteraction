# src/utils/config.py
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class RLConfig:
    """Configuration class for RL algorithms"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    num_workers: int = 4
    eps_clip: float = 0.2
    epochs: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'eps_clip': self.eps_clip,
            'epochs': self.epochs
        }