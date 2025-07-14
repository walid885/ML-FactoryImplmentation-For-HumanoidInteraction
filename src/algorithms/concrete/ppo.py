import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
import gymnasium as gym
from algorithms.base import RLAlgorithmBase
from utils.multicore import ParallelEnvironment, optimize_pytorch_multicore

class PPONetwork(nn.Module):
    """Lightweight PPO network for fast training"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value

class PPOAlgorithm(RLAlgorithmBase):
    """PPO implementation optimized for Ryzen 5 5600H"""
    
    algorithm_type = "on_policy_model_free"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        optimize_pytorch_multicore()
        
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.epochs = config.get('epochs', 4)
        self.batch_size = config.get('batch_size', 256)
        self.num_workers = config.get('num_workers', 4)  # Optimal for Ryzen 5 5600H
        
        self.network = None
        self.optimizer = None
        self.parallel_env = None
        
    def _init_network(self, state_dim: int, action_dim: int):
        """Initialize network and optimizer"""
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
    def train(self, environment, num_episodes: int) -> Dict[str, Any]:
        """Train PPO with parallel data collection"""
        env_name = environment if isinstance(environment, str) else "CartPole-v1"
        
        # Initialize parallel environment
        self.parallel_env = ParallelEnvironment(env_name, self.num_workers)
        
        # Get environment dimensions
        temp_env = gym.make(env_name)
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.n
        temp_env.close()
        
        # Initialize network
        self._init_network(state_dim, action_dim)
        
        metrics = {'episode_rewards': [], 'losses': []}
        
        for episode in range(num_episodes):
            # Collect batch of experiences
            batch_data = self.parallel_env.collect_batch(
                self._get_action, self.batch_size
            )
            
            # Update policy
            loss = self._update_policy(batch_data)
            metrics['losses'].append(loss)
            
            avg_reward = np.mean(batch_data['total_rewards'])
            metrics['episode_rewards'].append(avg_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        self.is_trained = True
        return metrics
    
    def _get_action(self, state: np.ndarray) -> int:
        """Get action from policy"""
        if self.network is None:
            return np.random.randint(0, 2)  # Random action for CartPole
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, _ = self.network(state_tensor)
            probs = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def _update_policy(self, batch_data: Dict) -> float:
        """Update policy using PPO objective"""
        states = torch.FloatTensor(batch_data['states'])
        actions = torch.LongTensor(batch_data['actions'])
        rewards = torch.FloatTensor(batch_data['rewards'])
        dones = torch.BoolTensor(batch_data['dones'])
        
        # Compute advantages
        advantages = self._compute_advantages(rewards, dones)
        
        total_loss = 0
        for _ in range(self.epochs):
            # Forward pass
            policy_logits, values = self.network(states)
            
            # Policy loss
            probs = torch.softmax(policy_logits, dim=-1)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            ratio = action_probs / (action_probs.detach() + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.epochs
    
    def _compute_advantages(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute advantages using GAE"""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            advantage = rewards[t] + self.gamma * advantage * (~dones[t]).float()
            advantages[t] = advantage
        
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def predict(self, state: np.ndarray) -> int:
        """Predict action for given state"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self._get_action(state)
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        self.model_path = filepath
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.config = checkpoint['config']
        
        # Recreate network
        state_dim = checkpoint['network_state']['shared.0.weight'].shape[1]
        action_dim = checkpoint['network_state']['policy_head.weight'].shape[0]
        self._init_network(state_dim, action_dim)
        
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.is_trained = True
        self.model_path = filepath