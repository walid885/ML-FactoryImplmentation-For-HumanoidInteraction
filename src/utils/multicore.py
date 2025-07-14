import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import gymnasium as gym
from typing import List, Callable, Any, Dict
import psutil
import torch

class MultiCoreTrainer:
    """Multicore training utilities for Ryzen 5 5600H optimization"""
    
    def __init__(self, num_workers: int = None):
        self.num_cores = psutil.cpu_count(logical=False)  # Physical cores
        self.num_threads = psutil.cpu_count(logical=True)  # Logical cores
        self.num_workers = num_workers or min(8, self.num_threads - 2)  # Reserve 2 for system
        
        # Optimize for Ryzen 5 5600H
        if self.num_cores == 6:  # Ryzen 5 5600H has 6 cores
            self.num_workers = 6
            torch.set_num_threads(2)  # 2 threads per core
        
        print(f"Using {self.num_workers} workers on {self.num_cores} cores")
    
    def parallel_env_step(self, env_configs: List[Dict], action_fn: Callable) -> List[Any]:
        """Execute environment steps in parallel"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._env_worker, config, action_fn) 
                      for config in env_configs]
            return [future.result() for future in as_completed(futures)]
    
    def _env_worker(self, env_config: Dict, action_fn: Callable) -> Dict:
        """Worker function for parallel environment execution"""
        env = gym.make(env_config['name'])
        obs, _ = env.reset()
        
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'total_reward': 0
        }
        
        for step in range(env_config.get('max_steps', 200)):
            action = action_fn(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            episode_data['states'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            episode_data['total_reward'] += reward
            
            obs = next_obs
            if done or truncated:
                break
        
        env.close()
        return episode_data

class ParallelEnvironment:
    """Parallel environment wrapper for faster data collection"""
    
    def __init__(self, env_name: str, num_envs: int = 8):
        self.env_name = env_name
        self.num_envs = num_envs
        self.multicore = MultiCoreTrainer(num_envs)
    
    def collect_batch(self, policy_fn: Callable, batch_size: int = 256) -> Dict:
        """Collect batch of experiences in parallel"""
        episodes_per_env = max(1, batch_size // (self.num_envs * 200))
        
        env_configs = [{
            'name': self.env_name,
            'max_steps': 200,
            'episodes': episodes_per_env
        } for _ in range(self.num_envs)]
        
        results = self.multicore.parallel_env_step(env_configs, policy_fn)
        
        # Aggregate results
        batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'total_rewards': []
        }
        
        for result in results:
            batch_data['states'].extend(result['states'])
            batch_data['actions'].extend(result['actions'])
            batch_data['rewards'].extend(result['rewards'])
            batch_data['dones'].extend(result['dones'])
            batch_data['total_rewards'].append(result['total_reward'])
        
        return batch_data

def optimize_pytorch_multicore():
    """Optimize PyTorch for multicore Ryzen performance"""
    torch.set_num_threads(6)  # Use all physical cores
    torch.set_num_interop_threads(2)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # AMD Ryzen optimizations
    import os
    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'