# src/algorithms/concrete/SAC.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from algorithms.base import RLAlgorithmBase
from utils.multicore import optimize_pytorch_multicore

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Twin critics for SAC
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

class SACAlgorithm(RLAlgorithmBase):
    def __init__(self, config):
        super().__init__(config)
        optimize_pytorch_multicore()
        
        # Hyperparameters
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 3e-4)
        self.lr_alpha = config.get('lr_alpha', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 1000000)
        self.alpha = config.get('alpha', 0.2)
        self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks_initialized = False
        
    def _initialize_networks(self, state_dim, action_dim, max_action):
        """Initialize networks after knowing environment dimensions"""
        self.max_action = max_action
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.networks_initialized = True
        
    def soft_update(self, target, source):
        """Soft update of target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, env_name, num_episodes):
        """Train the SAC algorithm"""
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
                # Select action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.actor.sample(state_tensor)
                action = action.cpu().data.numpy().flatten()
                
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
                alpha_val = self.log_alpha.exp().item() if self.automatic_entropy_tuning else self.alpha
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Alpha: {alpha_val:.3f}")
        
        env.close()
        self.is_trained = True
        return {'episode_rewards': episode_rewards}
    
    def _update_networks(self):
        """Update actor and critic networks with SAC"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
        
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_pi
            target_q = rewards + (self.gamma * next_q * (~dones).float())
        
        # Update Critics
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_pi - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update Alpha (temperature parameter)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic_target, self.critic)
    
    def predict(self, state):
        """Predict action for given state"""
        if not self.networks_initialized:
            raise ValueError("Model not trained yet")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor)
        return action.cpu().data.numpy().flatten()
    
    def save(self, filepath):
        """Save the model"""
        if not self.networks_initialized:
            raise ValueError("Model not trained yet")
        
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config
        }
        
        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Need to initialize networks first
        if not self.networks_initialized:
            # Infer dimensions from saved model weights
            actor_weight = checkpoint['actor_state_dict']['net.0.weight']
            state_dim = actor_weight.shape[1]  # Input dimension
            
            # Find action dimension from mean layer
            mean_weight = checkpoint['actor_state_dict']['mean.weight']
            action_dim = mean_weight.shape[0]  # Output dimension
            
            # Initialize with inferred dimensions
            self._initialize_networks(state_dim, action_dim, 1.0)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.is_trained = True