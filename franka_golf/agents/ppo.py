"""PPO agent for Franka Golf."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

from ..utils.device import get_device
from .base import BaseAgent


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        
        # Actor
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        
        # Critic
        value = self.critic(shared_features)
        
        return action_mean, action_std, value


class PPOAgent(BaseAgent):
    """PPO agent for continuous action spaces."""

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.device = get_device()
        
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        hidden_dim = kwargs.get("hidden_dim", 256)
        lr = kwargs.get("lr", 3e-4)
        
        self.network = PPONetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_ratio = kwargs.get("clip_ratio", 0.2)
        self.value_loss_coef = kwargs.get("value_loss_coef", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.ppo_epochs = kwargs.get("ppo_epochs", 4)
        self.batch_size = kwargs.get("batch_size", 64)
        
        # Buffer for collecting trajectories
        self.trajectory_buffer = []
        self.max_trajectory_length = kwargs.get("max_trajectory_length", 2048)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the policy network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_mean, action_std, value = self.network(obs_tensor)
            
            if deterministic:
                action = action_mean
                log_prob = torch.zeros(1)  # Deterministic action has zero log prob
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Store for later use in store_transition
            self._last_value = value.cpu().numpy()[0]
            self._last_log_prob = log_prob.cpu().numpy()[0]
            
            action = action.cpu().numpy()[0]
            return np.clip(action, -1, 1)

    def learn(self, batch: dict) -> dict:
        """Learn from collected trajectories."""
        if not self.trajectory_buffer:
            return {"loss": 0.0}
        
        # Process trajectories
        obs, actions, rewards, values, log_probs, dones = self._process_trajectories()
        
        # Compute advantages and returns
        advantages = self._compute_advantages(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0.0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(obs))
            
            for start_idx in range(0, len(obs), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_obs = torch.FloatTensor(obs[batch_indices]).to(self.device)
                batch_actions = torch.FloatTensor(actions[batch_indices]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[batch_indices]).to(self.device)
                batch_returns = torch.FloatTensor(returns[batch_indices]).to(self.device)
                batch_old_log_probs = torch.FloatTensor(log_probs[batch_indices]).to(self.device)
                
                # Forward pass
                action_mean, action_std, values = self.network(batch_obs)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_loss_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        # Clear buffer
        self.trajectory_buffer = []
        
        return {"loss": total_loss / num_updates if num_updates > 0 else 0.0}

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in trajectory buffer."""
        self.trajectory_buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "value": self._last_value,
            "log_prob": self._last_log_prob,
            "done": done
        })

    def _process_trajectories(self):
        """Process collected trajectories into arrays."""
        obs = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        for transition in self.trajectory_buffer:
            obs.append(transition["obs"])
            actions.append(transition["action"])
            rewards.append(transition["reward"])
            values.append(transition["value"])
            log_probs.append(transition["log_prob"])
            dones.append(transition["done"])
        
        return (np.array(obs), np.array(actions), np.array(rewards),
                np.array(values), np.array(log_probs), np.array(dones))

    def _compute_advantages(self, rewards, values, dones):
        """Compute GAE advantages."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        
        return advantages

    def save(self, path: str):
        """Save the trained network."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load the trained network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 