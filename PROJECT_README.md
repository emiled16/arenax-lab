# Franka Golf - ArenaX Labs ML Hiring Challenge

A modular implementation of 5 different reinforcement learning approaches for the Franka Golf environment, featuring a clean CLI interface and easy agent swapping.

## 🎯 Overview

This project implements 5 different approaches to solve the Franka Golf challenge:

1. **Baseline (Rule-based)** - Simple heuristic-based approach
2. **Behavioral Cloning (BC)** - Imitation learning from expert demonstrations
3. **Deep Q-Network (DQN)** - Value-based RL with action discretization
4. **Proximal Policy Optimization (PPO)** - Policy gradient method
5. **Soft Actor-Critic (SAC)** - Off-policy actor-critic method

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd franka-golf
   ```

2. **Install dependencies with Poetry:**
   ```bash
   poetry install
   poetry shell
   ```

   Or with pip:
   ```bash
   pip install -e .
   ```

3. **Install the SAI environment:**
   ```bash
   pip install sai-rl
   ```

### Basic Usage

#### List available agents:
```bash
python main.py list-agents
```

#### Train an agent:
```bash
# Train PPO agent
python main.py train --agent ppo --episodes 1000

# Train with custom parameters
python main.py train --agent sac --episodes 2000 --lr 1e-4 --batch-size 128
```

#### Evaluate a trained agent:
```bash
# Evaluate with 100 episodes
python main.py evaluate --agent ppo --checkpoint logs/ppo/checkpoint_episode_1000.pth --episodes 100

# Run single episode with rendering
python main.py evaluate --agent ppo --checkpoint logs/ppo/checkpoint_episode_1000.pth --single --render
```

#### Run agent demonstration:
```bash
# Run baseline agent (no checkpoint needed)
python main.py run --agent baseline --render

# Run trained agent
python main.py run --agent ppo --checkpoint logs/ppo/checkpoint_episode_1000.pth --render
```

## 🏗️ Project Structure

```
franka-golf/
├── franka_golf/
│   ├── agents/           # Agent implementations
│   │   ├── base.py      # Base agent class
│   │   ├── baseline.py  # Rule-based baseline
│   │   ├── bc.py        # Behavioral cloning
│   │   ├── dqn.py       # Deep Q-Network
│   │   ├── ppo.py       # PPO agent
│   │   └── sac.py       # SAC agent
│   ├── training/        # Training utilities
│   │   └── trainer.py   # Main trainer class
│   ├── evaluation/      # Evaluation utilities
│   │   └── evaluator.py # Evaluator class
│   └── utils/           # Utility functions
│       └── device.py    # GPU detection and management
├── examples/            # Example scripts
├── main.py             # CLI entry point
├── pyproject.toml      # Poetry configuration
└── setup.py           # Setup script
```

## 🤖 Agent Implementations

### 1. Baseline Agent
- **Type**: Rule-based
- **Approach**: Simple heuristics with phase-based control
- **Pros**: Fast, interpretable, no training required
- **Cons**: Limited performance, not adaptive

### 2. Behavioral Cloning Agent
- **Type**: Imitation learning
- **Approach**: Supervised learning from expert demonstrations
- **Pros**: Can learn complex behaviors from experts
- **Cons**: Requires expert data, doesn't improve beyond expert

### 3. DQN Agent
- **Type**: Value-based RL
- **Approach**: Q-learning with neural networks and action discretization
- **Pros**: Stable, good for discrete action spaces
- **Cons**: Action discretization limits precision

### 4. PPO Agent
- **Type**: Policy gradient
- **Approach**: Trust region policy optimization
- **Pros**: Stable, good sample efficiency, continuous actions
- **Cons**: On-policy, requires more samples

### 5. SAC Agent
- **Type**: Actor-critic
- **Approach**: Maximum entropy RL with soft updates
- **Pros**: Sample efficient, continuous actions, exploration
- **Cons**: More complex, sensitive to hyperparameters

## 🔧 Configuration

### Training Parameters

All agents support common training parameters:

- `--episodes`: Number of training episodes
- `--steps`: Maximum steps per episode
- `--eval-freq`: Evaluation frequency
- `--save-freq`: Checkpoint save frequency
- `--log-dir`: Directory for logs and checkpoints

### Agent-Specific Parameters

#### PPO Parameters:
- `--lr`: Learning rate (default: 3e-4)
- `--hidden-dim`: Hidden layer dimension (default: 256)
- `--batch-size`: Batch size (default: 64)
- `--gamma`: Discount factor (default: 0.99)

#### SAC Parameters:
- `--lr`: Learning rate (default: 3e-4)
- `--alpha`: Entropy coefficient (default: 0.2)
- `--tau`: Target network update rate (default: 0.005)

## 📊 Results and Evaluation

### Evaluation Metrics

The evaluator tracks several key metrics:

- **Average Reward**: Mean episode reward
- **Success Rate**: Percentage of successful episodes
- **Average Episode Length**: Mean steps per episode
- **Final Ball Distance**: Average distance from ball to hole

### Results Storage

Evaluation results are saved in JSON format:

```
results/
├── agent_name/
│   ├── evaluation_results.json    # Detailed results
│   └── evaluation_summary.json    # Summary statistics
```

## 🖥️ GPU Support

The project automatically detects and uses available GPUs:

- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon GPUs (M1/M2)
- **CPU**: Fallback when no GPU is available

Device information is printed at the start of training/evaluation.

## 🧪 Example Scripts

### Quick Training Examples

```bash
# Train baseline (fast, no GPU needed)
python examples/train_baseline.py

# Train PPO (recommended for good performance)
python examples/train_ppo.py
```

### Custom Training

```python
from franka_golf.agents import PPOAgent
from franka_golf.training.trainer import Trainer

# Create agent
agent = PPOAgent(env.observation_space, env.action_space, lr=1e-4)

# Create trainer
trainer = Trainer(env, agent, max_episodes=1000)

# Train
trainer.train()
```

## 🚨 Troubleshooting

### Common Issues

1. **SAI Environment Import Error**:
   ```bash
   pip install sai-rl
   ```

2. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 32`
   - Use smaller networks: `--hidden-dim 128`

3. **Training Not Converging**:
   - Increase episodes: `--episodes 2000`
   - Adjust learning rate: `--lr 1e-4`
   - Try different agent: PPO or SAC recommended

### Performance Tips

- **GPU Training**: Use CUDA or MPS for 5-10x speedup
- **Hyperparameter Tuning**: Start with PPO, then try SAC
- **Evaluation**: Use 100+ episodes for reliable metrics
- **Checkpoints**: Save frequently to avoid losing progress

## 📝 License

This project is part of the ArenaX Labs ML Hiring Challenge.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Behavioral Cloning](https://arxiv.org/abs/1507.04296) 