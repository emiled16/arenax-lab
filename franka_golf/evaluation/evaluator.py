"""Evaluation utilities for Franka Golf agents."""

import numpy as np
from typing import Dict, List, Any
import json
import os
from tqdm import tqdm

from ..agents import BaseAgent


class Evaluator:
    """Evaluator class for testing trained agents."""

    def __init__(self, env, agent: BaseAgent, **kwargs):
        """Initialize evaluator."""
        self.env = env
        self.agent = agent
        self.num_episodes = kwargs.get("num_episodes", 100)
        self.max_steps = kwargs.get("max_steps", 650)
        self.render = kwargs.get("render", False)
        self.save_results = kwargs.get("save_results", True)
        self.results_dir = kwargs.get("results_dir", "results")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the agent."""
        print(f"Evaluating agent for {self.num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        final_ball_distances = []
        
        for episode in tqdm(range(self.num_episodes)):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps):
                # Select action
                action = self.agent.act(obs, deterministic=True)
                
                # Take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_rates.append(info.get("success", False))
            
            # Calculate final ball distance to hole
            ball_pos = obs[18:21]  # Ball position in observation
            hole_pos = obs[28:31]  # Hole position in observation
            final_distance = np.linalg.norm(ball_pos - hole_pos)
            final_ball_distances.append(final_distance)
        
        # Compute statistics
        results = {
            "num_episodes": self.num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "success_rate": np.mean(success_rates),
            "avg_final_distance": np.mean(final_ball_distances),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "success_rates": success_rates,
            "final_ball_distances": final_ball_distances,
        }
        
        # Print results
        self._print_results(results)
        
        # Save results
        if self.save_results:
            self._save_results(results)
        
        return results

    def _print_results(self, results: Dict[str, Any]):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of episodes: {results['num_episodes']}")
        print(f"Average reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Reward range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"Average episode length: {results['avg_length']:.1f} steps")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Average final ball distance: {results['avg_final_distance']:.3f} meters")
        print("="*50)

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            "avg_reward": results["avg_reward"],
            "std_reward": results["std_reward"],
            "success_rate": results["success_rate"],
            "avg_final_distance": results["avg_final_distance"],
        }
        
        summary_path = os.path.join(self.results_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {self.results_dir}/")

    def run_single_episode(self, render: bool = True) -> Dict[str, Any]:
        """Run a single episode with optional rendering."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        trajectory = []
        
        for step in range(self.max_steps):
            # Record state
            trajectory.append({
                "step": step,
                "observation": obs.copy(),
                "ball_pos": obs[18:21],
                "hole_pos": obs[28:31],
            })
            
            # Select action
            action = self.agent.act(obs, deterministic=True)
            
            # Take action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update trajectory
            trajectory[-1].update({
                "action": action.copy(),
                "reward": reward,
                "success": info.get("success", False),
            })
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Calculate final distance
        final_ball_pos = obs[18:21]
        final_hole_pos = obs[28:31]
        final_distance = np.linalg.norm(final_ball_pos - final_hole_pos)
        
        results = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": info.get("success", False),
            "final_distance": final_distance,
            "trajectory": trajectory,
        }
        
        print(f"Episode completed: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, Success = {results['success']}, "
              f"Final Distance = {final_distance:.3f}")
        
        return results 