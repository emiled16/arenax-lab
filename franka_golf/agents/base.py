"""Base agent class for Franka Golf."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, observation_space, action_space, **kwargs):
        """Initialize the agent."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.training = True

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given an observation."""
        pass

    @abstractmethod
    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Learn from a batch of experience."""
        pass

    def save(self, path: str):
        """Save the agent to a file."""
        pass

    def load(self, path: str):
        """Load the agent from a file."""
        pass

    def train(self):
        """Set the agent to training mode."""
        self.training = True

    def eval(self):
        """Set the agent to evaluation mode."""
        self.training = False 