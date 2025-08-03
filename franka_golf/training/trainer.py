"""Training utilities for Franka Golf agents."""

import numpy as np
import torch
from typing import Dict, Any, Optional
from tqdm import tqdm
import json
import os

from ..agents import BaseAgent
from ..utils.device import print_device_info


class Trainer:
    """Main trainer class for training agents."""

    def __init__(self, env, agent: BaseAgent, **kwargs):
        """Initialize trainer."""
        self.env = env
        self.agent = agent
        self.max_episodes = kwargs.get("max_episodes", 1000)
        self.max_steps = kwargs.get("max_steps", 650)
        self.eval_freq = kwargs.get("eval_freq", 50)
        self.save_freq = kwargs.get("save_freq", 100)
        self.log_dir = kwargs.get("log_dir", "logs")
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        print_device_info()

    def train(self, **kwargs):
        """Train the agent."""
        print(f"Starting training for {self.max_episodes} episodes...")
        
        for episode in tqdm(range(self.max_episodes)):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps):
                # Select action
                action = self.agent.act(obs, deterministic=False)
                
                # Take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition (for agents that use replay buffers)
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Learn (for agents that learn online)
                if hasattr(self.agent, 'learn') and step % 10 == 0:
                    self.agent.learn({})
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rates.append(info.get("success", False))
            
            # Evaluation
            if episode % self.eval_freq == 0:
                self._evaluate(episode)
            
            # Save checkpoint
            if episode % self.save_freq == 0:
                self._save_checkpoint(episode)
            
            # Print progress
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_success = self.success_rates[-10:]
                print(f"Episode {episode}: Avg Reward = {np.mean(recent_rewards):.2f}, "
                      f"Success Rate = {np.mean(recent_success):.2f}")

    def _evaluate(self, episode: int):
        """Evaluate the agent."""
        eval_rewards = []
        eval_successes = []
        
        for _ in range(10):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                action = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_successes.append(info.get("success", False))
        
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_successes)
        
        print(f"Evaluation (Episode {episode}): Avg Reward = {avg_reward:.2f}, "
              f"Success Rate = {success_rate:.2f}")

    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, f"checkpoint_episode_{episode}.pth")
        self.agent.save(checkpoint_path)
        
        # Save training metrics
        metrics = {
            "episode": episode,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates": self.success_rates,
        }
        
        metrics_path = os.path.join(self.log_dir, f"metrics_episode_{episode}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}") 