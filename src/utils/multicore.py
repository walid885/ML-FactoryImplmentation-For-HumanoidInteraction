# src/utils/multicore.py
import torch
import numpy as np
from typing import Dict, Any, Callable
import gymnasium as gym
from multiprocessing import Pool, cpu_count
import os

def optimize_pytorch_multicore():
    """Optimize PyTorch for multicore usage"""
    num_cores = min(6, cpu_count())
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    os.environ['OMP_NUM_THREADS'] = str(num_cores)

class ParallelEnvironment:
    """Parallel environment for data collection"""
    
    def __init__(self, env_name: str, num_workers: int = 4):
        self.env_name = env_name
        self.num_workers = num_workers
        
    def collect_batch(self, policy_fn: Callable, batch_size: int) -> Dict[str, Any]:
        """Collect batch of experiences"""
        # Simplified implementation for testing
        env = gym.make(self.env_name)
        
        states = []
        actions = []
        rewards = []
        dones = []
        total_rewards = []
        
        for _ in range(batch_size):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(100):  # Max 100 steps per episode
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
            
            total_rewards.append(episode_reward)
        
        env.close()
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'total_rewards': total_rewards
        }