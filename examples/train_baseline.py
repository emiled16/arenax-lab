#!/usr/bin/env python3
"""Example script for training the baseline agent."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from franka_golf.agents import BaselineAgent
from franka_golf.training.trainer import Trainer
from franka_golf.utils.device import print_device_info

def main():
    """Train the baseline agent."""
    print("Training Baseline Agent")
    print_device_info()
    
    # Import environment
    try:
        from sai_rl import SAIClient
        sai = SAIClient(comp_id="franka-ml-hiring")
        env = sai.make_env()
    except ImportError:
        import sai_mujoco
        import gymnasium as gym
        env = gym.make("FrankaIkGolfCourseEnv-v0")
    
    # Create agent
    agent = BaselineAgent(env.observation_space, env.action_space)
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        max_episodes=100,
        max_steps=650,
        eval_freq=10,
        save_freq=20,
        log_dir="logs/baseline"
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main() 