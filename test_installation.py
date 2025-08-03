#!/usr/bin/env python3
"""Test script to verify Franka Golf installation."""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from franka_golf.agents import (
            BaselineAgent, 
            BehavioralCloningAgent, 
            DQNAgent, 
            PPOAgent, 
            SACAgent
        )
        print("✓ Agent imports successful")
    except ImportError as e:
        print(f"✗ Agent imports failed: {e}")
        return False
    
    try:
        from franka_golf.training.trainer import Trainer
        print("✓ Trainer import successful")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        return False
    
    try:
        from franka_golf.evaluation.evaluator import Evaluator
        print("✓ Evaluator import successful")
    except ImportError as e:
        print(f"✗ Evaluator import failed: {e}")
        return False
    
    try:
        from franka_golf.utils.device import get_device, print_device_info
        print("✓ Device utilities import successful")
    except ImportError as e:
        print(f"✗ Device utilities import failed: {e}")
        return False
    
    return True

def test_device_detection():
    """Test GPU device detection."""
    print("\nTesting device detection...")
    
    try:
        from franka_golf.utils.device import get_device, print_device_info
        device = get_device()
        print(f"✓ Device detection successful: {device}")
        print_device_info()
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation with mock spaces."""
    print("\nTesting agent creation...")
    
    try:
        import numpy as np
        from gymnasium.spaces import Box
        from franka_golf.agents import (
            BaselineAgent, 
            BehavioralCloningAgent, 
            DQNAgent, 
            PPOAgent, 
            SACAgent
        )
        
        # Mock spaces
        obs_space = Box(low=-np.inf, high=np.inf, shape=(31,))
        action_space = Box(low=-1, high=1, shape=(7,))
        
        # Test each agent
        agents = [
            ("Baseline", BaselineAgent),
            ("BC", BehavioralCloningAgent),
            ("DQN", DQNAgent),
            ("PPO", PPOAgent),
            ("SAC", SACAgent),
        ]
        
        for name, agent_class in agents:
            try:
                agent = agent_class(obs_space, action_space)
                print(f"✓ {name} agent creation successful")
            except Exception as e:
                print(f"✗ {name} agent creation failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Agent creation test failed: {e}")
        return False

def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI...")
    
    try:
        from franka_golf.cli import cli, AGENT_REGISTRY
        print(f"✓ CLI import successful")
        print(f"✓ Available agents: {list(AGENT_REGISTRY.keys())}")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Franka Golf Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_device_detection,
        test_agent_creation,
        test_cli,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Install SAI environment: pip install sai-rl")
        print("2. Try training: python main.py train --agent baseline --episodes 10")
        print("3. Check examples: python examples/train_baseline.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 