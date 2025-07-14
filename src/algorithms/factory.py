# src/algorithms/factory.py
from typing import Dict, Type, Any
from algorithms.base import RLAlgorithmBase

class RLAlgorithmFactory:
    """Factory class for creating RL algorithms"""
    
    def __init__(self):
        self._algorithms: Dict[str, Type[RLAlgorithmBase]] = {}
    
    def register_algorithm(self, name: str, algorithm_class: Type[RLAlgorithmBase]):
        """Register a new algorithm class"""
        if not issubclass(algorithm_class, RLAlgorithmBase):
            raise ValueError(f"Algorithm {name} must inherit from RLAlgorithmBase")
        
        self._algorithms[name.lower()] = algorithm_class
        print(f"Registered algorithm: {name}")
    
    def create_algorithm(self, name: str, config: Dict[str, Any]) -> RLAlgorithmBase:
        """Create an algorithm instance"""
        name = name.lower()
        
        if name not in self._algorithms:
            available = list(self._algorithms.keys())
            raise ValueError(f"Algorithm '{name}' not found. Available: {available}")
        
        algorithm_class = self._algorithms[name]
        return algorithm_class(config)
    
    def get_available_algorithms(self) -> list:
        """Get list of available algorithms"""
        return list(self._algorithms.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if algorithm is registered"""
        return name.lower() in self._algorithms

# Global factory instance
_global_factory = None

def get_algorithm_factory() -> RLAlgorithmFactory:
    """Get the global algorithm factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = RLAlgorithmFactory()
        
        # Auto-register all available algorithms
        try:
            from algorithms.concrete.ppo import PPOAlgorithm
            _global_factory.register_algorithm("ppo", PPOAlgorithm)
        except ImportError:
            pass
        
        try:
            from algorithms.concrete.ddpg import DDPGAlgorithm
            _global_factory.register_algorithm("ddpg", DDPGAlgorithm)
        except ImportError:
            pass
        
        try:
            from algorithms.concrete.TD3 import TD3Algorithm
            _global_factory.register_algorithm("td3", TD3Algorithm)
        except ImportError:
            pass
        
        try:
            from algorithms.concrete.SAC import SACAlgorithm
            _global_factory.register_algorithm("sac", SACAlgorithm)
        except ImportError:
            pass
    
    return _global_factory