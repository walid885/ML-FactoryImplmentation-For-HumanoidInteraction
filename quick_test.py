# quick_test.py - Test all concrete algorithms
import sys
from pathlib import Path
import time
import os
import gymnasium as gym

# Add src to path - FIXED
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_environment_compatibility():
    """Test if environment works with both discrete and continuous action spaces"""
    print("Testing environment compatibility...")
    
    # Test discrete action space (CartPole)
    try:
        env = gym.make("CartPole-v1")
        state, _ = env.reset()
        print(f"✓ CartPole - State dim: {state.shape[0]}, Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"❌ CartPole failed: {e}")
    
    # Test continuous action space (Pendulum)
    try:
        env = gym.make("Pendulum-v1")
        state, _ = env.reset()
        print(f"✓ Pendulum - State dim: {state.shape[0]}, Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"❌ Pendulum failed: {e}")

def test_algorithm(algorithm_name, algorithm_class, config, env_name="Pendulum-v1"):
    """Test a specific algorithm"""
    print(f"\n{'='*50}")
    print(f"Testing {algorithm_name}")
    print(f"{'='*50}")
    
    try:
        # Test algorithm creation
        agent = algorithm_class(config)
        print(f"✓ {algorithm_name} algorithm created")
        
        # Quick training test (3 episodes for speed)
        print(f"Running quick training test on {env_name}...")
        start_time = time.time()
        metrics = agent.train(env_name, num_episodes=3)
        duration = time.time() - start_time
        
        print(f"✓ Training completed in {duration:.1f}s")
        print(f"✓ Episode rewards: {[f'{r:.2f}' for r in metrics['episode_rewards']]}")
        
        # Test prediction
        env = gym.make(env_name)
        state, _ = env.reset()
        action = agent.predict(state)
        print(f"✓ Prediction works - Action shape: {action.shape}")
        env.close()
        
        # Test save/load
        model_path = f"test_{algorithm_name.lower()}_model.pkl"
        agent.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Test loading
        new_agent = algorithm_class(config)
        new_agent.load(model_path)
        print(f"✓ Model loaded successfully")
        
        # Test prediction after loading
        env = gym.make(env_name)
        state, _ = env.reset()
        action = new_agent.predict(state)
        print(f"✓ Prediction after loading works - Action: {action}")
        env.close()
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        print(f"🎉 {algorithm_name} test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ {algorithm_name} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_algorithms():
    """Test all concrete algorithms"""
    print("Testing all RL algorithms...")
    
    # Test environment compatibility first
    test_environment_compatibility()
    
    # Import algorithms
    try:
        from algorithms.concrete.ppo import PPOAlgorithm
        from algorithms.concrete.ddpg import DDPGAlgorithm
        from algorithms.concrete.TD3 import TD3Algorithm
        from algorithms.concrete.SAC import SACAlgorithm
        from utils.config import RLConfig
        print("✓ All imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test configurations for different algorithms
    configs = {
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'epochs': 2,  # Reduced for quick testing
            'batch_size': 64,
            'num_workers': 2
        },
        'ddpg': {
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 64,
            'buffer_size': 10000,
            'noise_std': 0.1
        },
        'td3': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 64,
            'buffer_size': 10000,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2
        },
        'sac': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 64,
            'buffer_size': 10000,
            'automatic_entropy_tuning': True
        }
    }
    
    # Test algorithms
    algorithms = [
        ('PPO', PPOAlgorithm, configs['ppo'], 'CartPole-v1'),  # PPO works with discrete
        ('DDPG', DDPGAlgorithm, configs['ddpg'], 'Pendulum-v1'),  # DDPG needs continuous
        ('TD3', TD3Algorithm, configs['td3'], 'Pendulum-v1'),   # TD3 needs continuous
        ('SAC', SACAlgorithm, configs['sac'], 'Pendulum-v1')    # SAC needs continuous
    ]
    
    results = {}
    for name, algorithm_class, config, env_name in algorithms:
        success = test_algorithm(name, algorithm_class, config, env_name)
        results[name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{name:15} {status}")
    
    print(f"\nOverall: {passed}/{total} algorithms passed")
    
    if passed == total:
        print("🎉 All algorithms are working correctly!")
        print("Ready for full training experiments!")
    else:
        print("⚠️  Some algorithms need attention before proceeding.")

def test_factory_integration():
    """Test factory pattern integration"""
    print(f"\n{'='*50}")
    print("Testing Factory Integration")
    print(f"{'='*50}")
    
    try:
        from algorithms.factory import RLAlgorithmFactory
        from algorithms.concrete.ppo import PPOAlgorithm
        from algorithms.concrete.ddpg import DDPGAlgorithm
        from algorithms.concrete.TD3 import TD3Algorithm
        from algorithms.concrete.SAC import SACAlgorithm
        
        # Setup factory
        factory = RLAlgorithmFactory()
        factory.register_algorithm("ppo", PPOAlgorithm)
        factory.register_algorithm("ddpg", DDPGAlgorithm)
        factory.register_algorithm("td3", TD3Algorithm)
        factory.register_algorithm("sac", SACAlgorithm)
        
        # Test creation through factory
        config = {'learning_rate': 3e-4, 'gamma': 0.99, 'batch_size': 64}
        
        for algo_name in ["ppo", "ddpg", "td3", "sac"]:
            agent = factory.create_algorithm(algo_name, config)
            print(f"✓ {algo_name.upper()} created via factory")
        
        print("✓ Factory integration works!")
        return True
        
    except Exception as e:
        print(f"❌ Factory integration failed: {e}")
        print("Note: This is expected if factory.py doesn't exist yet")
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive algorithm testing...")
    print("This will test all concrete RL algorithms in your project")
    
    # Run all tests
    test_all_algorithms()
    
    # Test factory integration if available
    test_factory_integration()
    
    print("\n" + "="*60)
    print("Testing complete! Check results above.")
    print("="*60)