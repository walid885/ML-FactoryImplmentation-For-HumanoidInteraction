# quick_test.py - Test your setup quickly
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append("src")

def test_setup():
    """Quick test of the training setup"""
    print("Testing RL setup...")
    
    try:
        # Test imports
        from algorithms.concrete.ppo import PPOAlgorithm
        from utils.config import RLConfig
        print("✓ Imports successful")
        
        # Test configuration
        config = RLConfig(
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            num_workers=4
        )
        print("✓ Config created")
        
        # Test algorithm creation
        agent = PPOAlgorithm(config.to_dict())
        print("✓ PPO algorithm created")
        
        # Quick training test (5 episodes)
        print("Running quick training test...")
        start_time = time.time()
        metrics = agent.train("CartPole-v1", num_episodes=5)
        duration = time.time() - start_time
        
        print(f"✓ Training completed in {duration:.1f}s")
        print(f"✓ Final reward: {metrics['episode_rewards'][-1]:.2f}")
        
        # Test save/load
        agent.save("test_model.pkl")
        print("✓ Model saved")
        
        new_agent = PPOAlgorithm(config.to_dict())
        new_agent.load("test_model.pkl")
        print("✓ Model loaded")
        
        print("\n🎉 Setup test passed! Ready for training.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_setup()