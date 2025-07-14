# src/experiments/train.py
import argparse
import os
import time
import mlflow
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.factory import get_algorithm_factory
from utils.config import RLConfig

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("humanoid_rl")

def get_default_config(algorithm: str) -> dict:
    """Get default configuration for each algorithm"""
    configs = {
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'epochs': 4,
            'batch_size': 256,
            'num_workers': 6
        },
        'ddpg': {
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_size': 1000000,
            'noise_std': 0.1,
            'noise_decay': 0.995
        },
        'td3': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_size': 1000000,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            'exploration_noise': 0.1
        },
        'sac': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_size': 1000000,
            'automatic_entropy_tuning': True
        }
    }
    return configs.get(algorithm.lower(), {})

def get_recommended_environment(algorithm: str) -> str:
    """Get recommended environment for each algorithm"""
    recommendations = {
        'ppo': 'CartPole-v1',  # Works with discrete actions
        'ddpg': 'Pendulum-v1',  # Requires continuous actions
        'td3': 'Pendulum-v1',   # Requires continuous actions
        'sac': 'Pendulum-v1'    # Requires continuous actions
    }
    return recommendations.get(algorithm.lower(), 'CartPole-v1')

def train_algorithm(algo_name: str, env_name: str = None, config_path: str = None, num_episodes: int = 100):
    """Train specified algorithm"""
    
    # Get algorithm factory
    factory = get_algorithm_factory()
    
    # Check if algorithm is available
    if not factory.is_registered(algo_name):
        available = factory.get_available_algorithms()
        raise ValueError(f"Algorithm '{algo_name}' not available. Available: {available}")
    
    # Set default environment if not specified
    if env_name is None:
        env_name = get_recommended_environment(algo_name)
        print(f"Using recommended environment for {algo_name}: {env_name}")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = get_default_config(algo_name)
        print(f"Using default configuration for {algo_name}")
    
    # Create algorithm
    agent = factory.create_algorithm(algo_name, config_dict)
    
    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_params(config_dict)
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("environment", env_name)
        mlflow.log_param("num_episodes", num_episodes)
        
        start_time = time.time()
        
        # Train
        print(f"Starting training: {algo_name.upper()} on {env_name}")
        print(f"Episodes: {num_episodes}")
        print("-" * 50)
        
        metrics = agent.train(env_name, num_episodes=num_episodes)
        training_time = time.time() - start_time
        
        # Log metrics
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("final_avg_reward", metrics['episode_rewards'][-1])
        mlflow.log_metric("best_reward", max(metrics['episode_rewards']))
        mlflow.log_metric("avg_reward", sum(metrics['episode_rewards']) / len(metrics['episode_rewards']))
        
        # Log episode rewards
        for i, reward in enumerate(metrics['episode_rewards']):
            mlflow.log_metric("episode_reward", reward, step=i)
        
        # Save model
        timestamp = int(time.time())
        model_path = f"models/{algo_name}_{env_name.replace('-', '_')}_{timestamp}.pkl"
        os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        mlflow.log_artifact(model_path)
        
        print("-" * 50)
        print(f"Training completed in {training_time/60:.1f} minutes")
        print(f"Model saved to {model_path}")
        print(f"Final reward: {metrics['episode_rewards'][-1]:.2f}")
        print(f"Best reward: {max(metrics['episode_rewards']):.2f}")
        print(f"Average reward: {sum(metrics['episode_rewards']) / len(metrics['episode_rewards']):.2f}")
        
        return agent, metrics

def main():
    parser = argparse.ArgumentParser(description="Train RL Algorithm")
    
    # Get available algorithms
    factory = get_algorithm_factory()
    available_algos = factory.get_available_algorithms()
    
    parser.add_argument("--algo", choices=available_algos, 
                        required=True, help="Algorithm to train")
    parser.add_argument("--env", default=None,
                        help="Environment to train on (auto-selected if not specified)")
    parser.add_argument("--config", default=None,
                        help="Config file path")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes")
    
    args = parser.parse_args()
    
    # Setup
    setup_mlflow()
    
    try:
        # Train
        agent, metrics = train_algorithm(args.algo, args.env, args.config, args.episodes)
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Algorithm: {args.algo.upper()}")
        print(f"Environment: {args.env or get_recommended_environment(args.algo)}")
        print(f"Episodes: {args.episodes}")
        print(f"Best reward: {max(metrics['episode_rewards']):.2f}")
        print(f"Final reward: {metrics['episode_rewards'][-1]:.2f}")
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())