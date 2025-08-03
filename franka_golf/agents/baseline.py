"""Rule-based baseline agent for Franka Golf."""

import numpy as np
from .base import BaseAgent


class BaselineAgent(BaseAgent):
    """Rule-based baseline agent using simple heuristics."""

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.phase = "approach"  # approach, grasp, swing, follow
        self.target_pos = None
        self.grasp_threshold = 0.05
        self.swing_threshold = 0.1

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action based on current phase and observation."""
        # Parse observation
        joint_pos = observation[:9]
        joint_vel = observation[9:18]
        ball_pos = observation[18:21]
        club_pos = observation[21:24]
        club_quat = observation[24:28]
        hole_pos = observation[28:31]
        
        # Get end-effector position (approximate from joint positions)
        ee_pos = self._get_ee_position(joint_pos)
        
        action = np.zeros(7)
        
        if self.phase == "approach":
            # Move towards club
            direction = club_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            action[:3] = direction * 0.1  # Small step towards club
            
            if np.linalg.norm(ee_pos - club_pos) < self.grasp_threshold:
                self.phase = "grasp"
                
        elif self.phase == "grasp":
            # Close gripper and lift
            action[6] = -1.0  # Close gripper
            action[2] = 0.05  # Move up
            self.phase = "swing"
            
        elif self.phase == "swing":
            # Swing towards ball
            direction = ball_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            action[:3] = direction * 0.2  # Stronger movement
            
            if np.linalg.norm(ee_pos - ball_pos) < self.swing_threshold:
                self.phase = "follow"
                
        elif self.phase == "follow":
            # Follow ball towards hole
            direction = hole_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            action[:3] = direction * 0.05
            
        # Add small noise for exploration
        if not deterministic:
            action += np.random.normal(0, 0.01, 7)
            
        return np.clip(action, -1, 1)

    def learn(self, batch: dict) -> dict:
        """Baseline agent doesn't learn."""
        return {"loss": 0.0}

    def _get_ee_position(self, joint_pos):
        """Approximate end-effector position from joint positions."""
        # Simple forward kinematics approximation
        # This is a rough estimate - in practice you'd use proper FK
        base_pos = np.array([0.0, 0.0, 0.5])
        ee_offset = np.array([0.0, 0.0, 0.5])  # Approximate arm length
        return base_pos + ee_offset

    def save(self, path: str):
        """Save agent state."""
        np.save(path, {"phase": self.phase})

    def load(self, path: str):
        """Load agent state."""
        data = np.load(path, allow_pickle=True).item()
        self.phase = data["phase"] 