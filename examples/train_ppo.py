#!/usr/bin/env python3
"""Example script for training the PPO agent."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from franka_golf.agents import PPOAgent
from franka_golf.training.trainer import Trainer
from franka_golf.utils.device import print_device_info

def main():
    """Train the PPO agent."""
    print("Training PPO Agent")
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
    
    # Create agent with PPO-specific parameters
    agent = PPOAgent(
        env.observation_space, 
        env.action_space,
        lr=3e-4,
        hidden_dim=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        max_trajectory_length=2048
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        max_episodes=1000,
        max_steps=650,
        eval_freq=50,
        save_freq=100,
        log_dir="logs/ppo"
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main() 