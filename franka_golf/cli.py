"""Command-line interface for Franka Golf."""

import click
import os
from typing import Dict, Any

from .agents import (
    BaselineAgent,
    BehavioralCloningAgent,
    DQNAgent,
    PPOAgent,
    SACAgent,
)
from .training.trainer import Trainer
from .evaluation.evaluator import Evaluator
from .utils.device import print_device_info


AGENT_REGISTRY = {
    "baseline": BaselineAgent,
    "bc": BehavioralCloningAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def create_agent(agent_type: str, observation_space, action_space, **kwargs) -> Any:
    """Create an agent instance."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(observation_space, action_space, **kwargs)


@click.group()
def cli():
    """Franka Golf - ArenaX Labs ML Hiring Challenge CLI."""
    pass


@cli.command()
@click.option("--agent", type=click.Choice(list(AGENT_REGISTRY.keys())), required=True,
              help="Type of agent to train")
@click.option("--episodes", default=1000, help="Number of training episodes")
@click.option("--steps", default=650, help="Maximum steps per episode")
@click.option("--eval-freq", default=50, help="Evaluation frequency")
@click.option("--save-freq", default=100, help="Save checkpoint frequency")
@click.option("--log-dir", default="logs", help="Directory to save logs")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--hidden-dim", default=256, help="Hidden dimension for neural networks")
@click.option("--batch-size", default=64, help="Batch size for training")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--render", is_flag=True, help="Render environment during training")
def train(agent, episodes, steps, eval_freq, save_freq, log_dir, lr, hidden_dim, batch_size, gamma, render):
    """Train an agent."""
    print(f"Training {agent} agent...")
    print_device_info()
    
    # Import environment here to avoid issues with SAI client
    try:
        from sai_rl import SAIClient
        sai = SAIClient(comp_id="franka-ml-hiring")
        env = sai.make_env()
    except (ImportError, Exception):
        import sai_mujoco
        import gymnasium as gym
        env = gym.make("FrankaIkGolfCourseEnv-v0")
    
    # Create agent
    agent_kwargs = {
        "lr": lr,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "gamma": gamma,
    }
    
    agent_instance = create_agent(agent, env.observation_space, env.action_space, **agent_kwargs)
    
    # Create trainer
    trainer_kwargs = {
        "max_episodes": episodes,
        "max_steps": steps,
        "eval_freq": eval_freq,
        "save_freq": save_freq,
        "log_dir": os.path.join(log_dir, agent),
    }
    
    trainer = Trainer(env, agent_instance, **trainer_kwargs)
    
    # Start training
    trainer.train()
    
    print(f"Training completed! Checkpoints saved in {log_dir}/{agent}/")


@cli.command()
@click.option("--agent", type=click.Choice(list(AGENT_REGISTRY.keys())), required=True,
              help="Type of agent to evaluate")
@click.option("--checkpoint", required=True, help="Path to agent checkpoint")
@click.option("--episodes", default=100, help="Number of evaluation episodes")
@click.option("--steps", default=650, help="Maximum steps per episode")
@click.option("--results-dir", default="results", help="Directory to save results")
@click.option("--render", is_flag=True, help="Render environment during evaluation")
@click.option("--single", is_flag=True, help="Run single episode with detailed output")
def evaluate(agent, checkpoint, episodes, steps, results_dir, render, single):
    """Evaluate a trained agent."""
    print(f"Evaluating {agent} agent from {checkpoint}...")
    print_device_info()
    
    # Import environment
    try:
        from sai_rl import SAIClient
        sai = SAIClient(comp_id="franka-ml-hiring")
        env = sai.make_env()
    except (ImportError, Exception):
        import sai_mujoco
        import gymnasium as gym
        env = gym.make("FrankaIkGolfCourseEnv-v0")
    
    # Create agent
    agent_instance = create_agent(agent, env.observation_space, env.action_space)
    
    # Load checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    agent_instance.load(checkpoint)
    print(f"Loaded checkpoint from {checkpoint}")
    
    # Create evaluator
    evaluator_kwargs = {
        "num_episodes": 1 if single else episodes,
        "max_steps": steps,
        "render": render,
        "results_dir": os.path.join(results_dir, agent),
    }
    
    evaluator = Evaluator(env, agent_instance, **evaluator_kwargs)
    
    # Run evaluation
    if single:
        results = evaluator.run_single_episode(render=render)
    else:
        results = evaluator.evaluate()
    
    print(f"Evaluation completed! Results saved in {results_dir}/{agent}/")


@cli.command()
@click.option("--agent", type=click.Choice(list(AGENT_REGISTRY.keys())), required=True,
              help="Type of agent to run")
@click.option("--checkpoint", help="Path to agent checkpoint (optional for baseline)")
@click.option("--episodes", default=1, help="Number of episodes to run")
@click.option("--steps", default=650, help="Maximum steps per episode")
@click.option("--render", is_flag=True, help="Render environment")
def run(agent, checkpoint, episodes, steps, render):
    """Run an agent for demonstration."""
    print(f"Running {agent} agent...")
    print_device_info()
    
    # Import environment
    try:
        from sai_rl import SAIClient
        sai = SAIClient(comp_id="franka-ml-hiring")
        env = sai.make_env(render_mode="human" if render else None)
    except (ImportError, Exception):
        import sai_mujoco
        import gymnasium as gym
        env = gym.make("FrankaIkGolfCourseEnv-v0", render_mode="human" if render else None)
    
    # Create agent
    agent_instance = create_agent(agent, env.observation_space, env.action_space)
    
    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        agent_instance.load(checkpoint)
        print(f"Loaded checkpoint from {checkpoint}")
    
    # Run episodes
    for episode in range(episodes):
        print(f"Running episode {episode + 1}/{episodes}")
        
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(steps):
            action = agent_instance.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: Reward = {episode_reward:.2f}, "
              f"Success = {info.get('success', False)}")


@cli.command()
def list_agents():
    """List available agents."""
    print("Available agents:")
    for agent_name in AGENT_REGISTRY.keys():
        print(f"  - {agent_name}")


if __name__ == "__main__":
    cli() 