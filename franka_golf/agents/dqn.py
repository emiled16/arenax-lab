"""DQN agent for Franka Golf."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from ..utils.device import get_device
from .base import BaseAgent


class DQNNetwork(nn.Module):
    """Neural network for DQN."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent(BaseAgent):
    """DQN agent with continuous action space discretization."""

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.device = get_device()
        
        obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        hidden_dim = kwargs.get("hidden_dim", 256)
        lr = kwargs.get("lr", 1e-3)
        
        # Discretize action space
        self.action_bins = kwargs.get("action_bins", 5)
        self.discrete_actions = self._create_discrete_actions()
        
        self.q_network = DQNNetwork(obs_dim, len(self.discrete_actions), hidden_dim).to(self.device)
        self.target_network = DQNNetwork(obs_dim, len(self.discrete_actions), hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # DQN parameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.batch_size = kwargs.get("batch_size", 32)
        self.memory = deque(maxlen=10000)
        self.update_freq = kwargs.get("update_freq", 100)
        self.steps = 0

    def _create_discrete_actions(self):
        """Create discrete action set."""
        actions = []
        for i in range(self.action_bins):
            for j in range(self.action_bins):
                for k in range(self.action_bins):
                    action = np.array([
                        (i - self.action_bins // 2) / (self.action_bins // 2),
                        (j - self.action_bins // 2) / (self.action_bins // 2),
                        (k - self.action_bins // 2) / (self.action_bins // 2),
                        0, 0, 0, 0  # Orientation and gripper
                    ])
                    actions.append(action)
        return actions

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if not deterministic and random.random() < self.epsilon:
            action_idx = random.randrange(len(self.discrete_actions))
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax().item()
        
        return self.discrete_actions[action_idx]

    def learn(self, batch: dict) -> dict:
        """Learn from experience replay."""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch_data = random.sample(self.memory, self.batch_size)
        obs_batch = torch.FloatTensor([d["obs"] for d in batch_data]).to(self.device)
        action_batch = torch.LongTensor([d["action_idx"] for d in batch_data]).to(self.device)
        reward_batch = torch.FloatTensor([d["reward"] for d in batch_data]).to(self.device)
        next_obs_batch = torch.FloatTensor([d["next_obs"] for d in batch_data]).to(self.device)
        done_batch = torch.BoolTensor([d["done"] for d in batch_data]).to(self.device)
        
        # Compute Q values
        current_q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_obs_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        return {"loss": loss.item(), "epsilon": self.epsilon}

    def store_transition(self, obs, action_idx, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.memory.append({
            "obs": obs,
            "action_idx": action_idx,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        })

    def save(self, path: str):
        """Save the trained network."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        """Load the trained network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"] 