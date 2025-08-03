"""Behavioral cloning agent for Franka Golf."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.device import get_device
from .base import BaseAgent


class BCNetwork(nn.Module):
    """Neural network for behavioral cloning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class BehavioralCloningAgent(BaseAgent):
    """Behavioral cloning agent."""

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.device = get_device()
        
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        hidden_dim = kwargs.get("hidden_dim", 256)
        lr = kwargs.get("lr", 1e-3)
        
        self.network = BCNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # For data collection
        self.expert_data = []
        self.is_collecting = False

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the trained network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.network(obs_tensor).cpu().numpy()[0]
            
        if not deterministic and self.training:
            action += np.random.normal(0, 0.1, action.shape)
            
        return np.clip(action, -1, 1)

    def learn(self, batch: dict) -> dict:
        """Learn from expert demonstrations."""
        if not self.expert_data:
            return {"loss": 0.0}
            
        # Convert to tensors
        observations = torch.FloatTensor([d["obs"] for d in self.expert_data]).to(self.device)
        actions = torch.FloatTensor([d["action"] for d in self.expert_data]).to(self.device)
        
        # Create dataloader
        dataset = TensorDataset(observations, actions)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        total_loss = 0.0
        num_batches = 0
        
        for obs_batch, action_batch in dataloader:
            self.optimizer.zero_grad()
            predicted_actions = self.network(obs_batch)
            loss = self.criterion(predicted_actions, action_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def collect_expert_data(self, observation: np.ndarray, action: np.ndarray):
        """Collect expert demonstration data."""
        if self.is_collecting:
            self.expert_data.append({
                "obs": observation.copy(),
                "action": action.copy()
            })

    def start_collecting(self):
        """Start collecting expert data."""
        self.is_collecting = True
        self.expert_data = []

    def stop_collecting(self):
        """Stop collecting expert data."""
        self.is_collecting = False

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