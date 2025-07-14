# src/algorithms/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class RLAlgorithmBase(ABC):
    """Base class for all RL algorithms"""
    
    algorithm_type = None
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_trained = False
        self.model_path = None
    
    @abstractmethod
    def train(self, environment, num_episodes: int) -> Dict[str, Any]:
        """Train the algorithm"""
        pass
    
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Predict action for given state"""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model"""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model"""
        pass