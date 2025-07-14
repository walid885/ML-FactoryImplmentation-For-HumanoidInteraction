# src/utils/multicore.py
import torch
import numpy as np
from typing import Dict, Any, Callable
import gymnasium as gym
from multiprocessing import Pool, cpu_count
import os

_pytorch_optimized = False

def optimize_pytorch_multicore():
    """Optimize PyTorch for multicore usage"""
    global _pytorch_optimized
    if _pytorch_optimized:
        return
    
    num_cores = min(6, cpu_count())
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    _pytorch_optimized = True

class ParallelEnvironment:
    """Parallel environment for data collection optimized for Ryzen 5 5600H"""
    
    def __init__(self, env_name: str, num_workers: int = 4):
        self.env_name = env_name
        # Optimal for Ryzen 5 5600H (6 cores, 12 threads)
        self.num_workers = min(num_workers, 6)
        
    def collect_batch(self, policy_fn: Callable, batch_size: int) -> Dict[str, Any]:
        """Collect batch of experiences with optimized batching"""
        env = gym.make(self.env_name)
        
        states = []
        actions = []
        rewards = []
        dones = []
        total_rewards = []
        
        # Collect smaller episodes for better performance
        episodes_per_batch = max(1, batch_size // 64)
        
        for _ in range(episodes_per_batch):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(200):  # Max 200 steps per episode
                action = policy_fn(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done or truncated)
                
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    break
                
                if len(states) >= batch_size:
                    break
            
            total_rewards.append(episode_reward)
            
            if len(states) >= batch_size:
                break
        
        env.close()
        
        return {
            'states': np.array(states[:batch_size]),
            'actions': np.array(actions[:batch_size]),
            'rewards': np.array(rewards[:batch_size]),
            'dones': np.array(dones[:batch_size]),
            'total_rewards': total_rewards
        }