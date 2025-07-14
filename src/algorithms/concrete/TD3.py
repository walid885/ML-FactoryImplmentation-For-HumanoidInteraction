# src/algorithms/concrete/TD3.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
from algorithms.base import RLAlgorithmBase
from utils.multicore import optimize_pytorch_multicore

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Twin critics for TD3
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class TD3Algorithm(RLAlgorithmBase):
    def __init__(self, config):
        super().__init__(config)
        optimize_pytorch_multicore()
        
        # Hyperparameters
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 1000000)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_freq = config.get('policy_freq', 2)
        self.exploration_noise = config.get('exploration_noise', 0.1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks_initialized = False
        self.total_it = 0
        
    def _initialize_networks(self, state_dim, action_dim, max_action):
        """Initialize networks after knowing environment dimensions"""
        self.max_action = max_action
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.networks_initialized = True
        
    def soft_update(self, target, source):
        """Soft update of target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, env_name, num_episodes):
        """Train the TD3 algorithm"""
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        if not self.networks_initialized:
            self._initialize_networks(state_dim, action_dim, max_action)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action with noise
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.actor(state_tensor).cpu().data.numpy().flatten()
                
                # Add exploration noise
                noise = np.random.normal(0, self.exploration_noise, size=action_dim)
                action = np.clip(action + noise, -max_action, max_action)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # Train if enough samples
                if len(self.replay_buffer) > self.batch_size:
                    self._update_networks()
            
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        env.close()
        self.is_trained = True
        return {'episode_rewards': episode_rewards}
    
    def _update_networks(self):
        """Update actor and critic networks with TD3 improvements"""
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Clipped Double Q-Learning
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (self.gamma * target_q * (~dones).float())
        
        # Update Critics
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed Policy Update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
    
    def predict(self, state):
        """Predict action for given state"""
        if not self.networks_initialized:
            raise ValueError("Model not trained yet")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        return action
    
    def save(self, filepath):
        """Save the model"""
        if not self.networks_initialized:
            raise ValueError("Model not trained yet")
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'total_it': self.total_it
        }, filepath)
    
    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Need to initialize networks first
        if not self.networks_initialized:
            # Try to infer dimensions from saved model
            dummy_state = torch.randn(1, 4)  # Assume CartPole for testing
            dummy_action = torch.randn(1, 1)
            self._initialize_networks(4, 1, 1.0)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint.get('total_it', 0)
        self.is_trained = True