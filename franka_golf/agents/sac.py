"""SAC agent for Franka Golf."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random

from ..utils.device import get_device
from .base import BaseAgent


class SACNetwork(nn.Module):
    """Actor-Critic networks for SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Critic networks (Q1 and Q2)
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, obs):
        """Get action from actor network."""
        features = self.actor(obs)
        mean = torch.tanh(self.actor_mean(features))
        logstd = self.actor_logstd(features)
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        
        dist = Normal(mean, std)
        action = dist.rsample()  # Use reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing
        action = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def get_q_values(self, obs, action):
        """Get Q values from both critic networks."""
        x = torch.cat([obs, action], dim=-1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2


class SACAgent(BaseAgent):
    """SAC agent for continuous action spaces."""

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.device = get_device()
        
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        hidden_dim = kwargs.get("hidden_dim", 256)
        lr = kwargs.get("lr", 3e-4)
        
        self.network = SACNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = SACNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Separate optimizers for actor and critics
        self.actor_optimizer = optim.Adam(
            list(self.network.actor_mean.parameters()) + 
            list(self.network.actor_logstd.parameters()) + 
            list(self.network.actor.parameters()), 
            lr=lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.network.critic1.parameters()) + 
            list(self.network.critic2.parameters()), 
            lr=lr
        )
        
        # SAC parameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.alpha = kwargs.get("alpha", 0.2)
        self.batch_size = kwargs.get("batch_size", 64)
        self.memory = deque(maxlen=100000)
        self.update_freq = kwargs.get("update_freq", 1)
        self.steps = 0

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the policy network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            if deterministic:
                # For deterministic actions, use mean without sampling
                features = self.network.actor(obs_tensor)
                mean = torch.tanh(self.network.actor_mean(features))
                action = mean
            else:
                action, _ = self.network.get_action(obs_tensor)
            
            action = action.cpu().numpy()[0]
            return np.clip(action, -1, 1)

    def learn(self, batch: dict) -> dict:
        """Learn from experience replay."""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch_data = random.sample(self.memory, self.batch_size)
        obs_batch = torch.FloatTensor([d["obs"] for d in batch_data]).to(self.device)
        action_batch = torch.FloatTensor([d["action"] for d in batch_data]).to(self.device)
        reward_batch = torch.FloatTensor([d["reward"] for d in batch_data]).to(self.device)
        next_obs_batch = torch.FloatTensor([d["next_obs"] for d in batch_data]).to(self.device)
        done_batch = torch.BoolTensor([d["done"] for d in batch_data]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.network.get_action(next_obs_batch)
            next_q1, next_q2 = self.target_network.get_q_values(next_obs_batch, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward_batch.unsqueeze(1) + self.gamma * (1 - done_batch.unsqueeze(1)) * (
                next_q - self.alpha * next_log_prob
            )
        
        current_q1, current_q2 = self.network.get_q_values(obs_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_action, new_log_prob = self.network.get_action(obs_batch)
        new_q1, new_q2 = self.network.get_q_values(obs_batch, new_action)
        new_q = torch.min(new_q1, new_q2)
        actor_loss = (self.alpha * new_log_prob - new_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target network
        if self.steps % self.update_freq == 0:
            self._update_target_network()
        
        self.steps += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def _update_target_network(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.memory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        })

    def save(self, path: str):
        """Save the trained network."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load the trained network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"]) 